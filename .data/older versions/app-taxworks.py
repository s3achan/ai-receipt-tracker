import os
import re
import json
import base64
from datetime import date

import pandas as pd
import streamlit as st
import pdfplumber
import plotly.express as px
from dotenv import load_dotenv
from openai import OpenAI

from sqlalchemy.orm import Session
from sqlalchemy import create_engine, ForeignKey, inspect
from sqlalchemy.orm import DeclarativeBase, sessionmaker, relationship, Mapped, mapped_column
from sqlalchemy import Integer, String, Float, Date as SQLDate

# ----------------------------
# Streamlit config (ONLY ONCE)
# ----------------------------
st.set_page_config(page_title="Receipt Classifier", page_icon="🧾", layout="wide")



load_dotenv()

# ===================== CONFIG =====================
DB_PATH = "receipts.db"

# OpenAI models
MODEL_VISION = "gpt-4.1-mini"   # Vision for photos
MODEL_TEXT = "gpt-4o-mini"      # Text for PDFs

# --- HARDCODED CATEGORIES LIST ---
INITIAL_CATEGORIES = [
    "Meat & Prepared Foods",
    "Produce",
    "Dairy & Eggs",
    "Beverages",
    "Frozen & Packaged Foods",
    "Baby",
    "Household",
    "Clothing",
    "Tax",
    "Other",
]
CATEGORIES = list(INITIAL_CATEGORIES)

# ===================== OPENAI KEY =====================
def get_openai_key() -> str | None:
    # Streamlit secrets (Cloud or local secrets.toml)
    try:
        key = st.secrets.get("OPENAI_API_KEY")
        if key:
            return key
    except Exception:
        pass

    # Fallback to env/.env (local)
    return os.getenv("OPENAI_API_KEY")

OPENAI_API_KEY = get_openai_key()
if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY. Add it to Streamlit Cloud Secrets or your .env file.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)



# ===================== SQLALCHEMY =====================
class Base(DeclarativeBase):
    pass

engine = create_engine(f"sqlite:///{DB_PATH}", echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

class Receipt(Base):
    __tablename__ = "receipts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    store: Mapped[str] = mapped_column(String, default="Costco")
    date: Mapped[date] = mapped_column(SQLDate, default=date.today)
    subtotal: Mapped[float | None] = mapped_column(Float)
    tax: Mapped[float | None] = mapped_column(Float)
    total: Mapped[float | None] = mapped_column(Float)
    pdf_path: Mapped[str | None] = mapped_column(String)
    items: Mapped[list["ReceiptItem"]] = relationship(
        "ReceiptItem", back_populates="receipt", cascade="all, delete-orphan"
    )

class ReceiptItem(Base):
    __tablename__ = "receipt_items"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    receipt_id: Mapped[int] = mapped_column(ForeignKey("receipts.id"), index=True)
    date: Mapped[date] = mapped_column(SQLDate, nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    category: Mapped[str] = mapped_column(String, nullable=False)
    price: Mapped[float | None] = mapped_column(Float)
    receipt: Mapped[Receipt] = relationship("Receipt", back_populates="items")

Base.metadata.create_all(engine)

# ===================== DYNAMIC CATEGORY LOADING & MAPPING =====================
def load_dynamic_categories():
    global CATEGORIES
    session = SessionLocal()
    try:
        if inspect(engine).has_table("receipt_items"):
            db_categories = session.query(ReceiptItem.category).distinct().all()
            db_category_names = {c[0] for c in db_categories}
            all_categories = set(INITIAL_CATEGORIES) | db_category_names
            CATEGORIES = sorted(list(all_categories))
    except Exception as e:
        print(f"Error loading dynamic categories: {e}")
        CATEGORIES = list(INITIAL_CATEGORIES)
    finally:
        session.close()

def get_item_category_mapping() -> dict:
    session = SessionLocal()
    mapping = {}
    try:
        if inspect(engine).has_table("receipt_items"):
            query = (
                session.query(ReceiptItem.name, ReceiptItem.category, Receipt.date)
                .join(Receipt)
                .order_by(Receipt.date.desc())
            )
            for name, category, _d in query.all():
                normalized = str(name).strip().lower()
                if normalized and normalized not in mapping:
                    mapping[normalized] = category
    except Exception as e:
        print(f"Error loading item category mapping: {e}")
    finally:
        session.close()
    return mapping

load_dynamic_categories()

# ===================== SESSION STATE =====================
if "parsed_data" not in st.session_state:
    st.session_state.parsed_data = None
if "df_to_save" not in st.session_state:
    st.session_state.df_to_save = None

# ===================== HELPERS =====================
def extract_text_from_pdf(uploaded_file) -> str:
    with pdfplumber.open(uploaded_file) as pdf:
        pages = []
        for page in pdf.pages:
            t = page.extract_text() or ""
            if t.strip():
                pages.append(t)
        return "\n\n".join(pages).strip()

def get_item_count(text: str) -> int | None:
    match = re.search(r"TOTAL NUMBER OF ITEMS SOLD\s*=\s*(\d+)", text, re.IGNORECASE)
    return int(match.group(1)) if match else None

def base64_data_url(image_bytes: bytes, mime: str) -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def apply_category_memory(parsed_data: dict) -> list[dict]:
    items = parsed_data.get("items", [])
    if not items:
        return []
    category_map = get_item_category_mapping()
    if not category_map:
        return items

    st.info("Applying user-defined category memory...")
    for item in items:
        name = str(item.get("name", "")).strip()
        key = name.lower()
        if key in category_map:
            item["category"] = category_map[key]
            st.write(f"Memory Hit: **{name}** → **{category_map[key]}**")
    return items

# ===================== PROMPT & PARSING =====================
def build_prompt(category_list: str, item_count: int | None, receipt_text: str | None = None) -> str:
    count_constraint = ""
    if item_count and item_count > 0:
        count_constraint = f"""
5. **COUNT CHECK (CRITICAL):** The total number of items in the final "items" list MUST match "TOTAL NUMBER OF ITEMS SOLD" (which is **{item_count}**).
"""

    receipt_block = ""
    if receipt_text is not None:
        receipt_block = f"""
Receipt text:
---BEGIN---
{receipt_text}
---END---
"""

    return f"""
Extract the store name, items purchased, subtotal, tax, total, and the **purchase date**.

Costco Tax Rule (CRITICAL):
- If the store is Costco: an item is taxable ONLY if the receipt shows an "A" marker next to that item line.
- Set each item's "taxable" = true/false based on the "A" marker.
- Even if a tax-looking number exists, you must still mark taxable correctly based on "A".

Instructions:
1. **Item Definition:** Extract every line that represents a unique purchased product.
2. **Discount Exclusion (CRITICAL):** DO NOT include discounts/instant savings/negative lines in "items".
3. **Price Rule:** Use the full, undiscounted price for each item.
4. **Categorization:** Use one exact category per item, prioritizing this list: {category_list}.
{count_constraint}

Return ONLY valid JSON:
{{
  "store": "Costco",
  "items": [
    {{"name":"", "category":"", "price":0.0, "taxable": false}}
  ],
  "subtotal": 0.0,
  "tax": 0.0,
  "total": 0.0,
  "receipt_date": "2025-11-13"
}}

{receipt_block}
""".strip()

def parse_receipt_from_text(text: str, item_count: int | None) -> dict:
    category_list = ", ".join(CATEGORIES)
    prompt = build_prompt(category_list=category_list, item_count=item_count, receipt_text=text)

    resp = client.responses.create(
        model=MODEL_TEXT,
        input=[{"role": "user", "content": prompt}],
        text={"format": {"type": "json_object"}},
        temperature=0.1,
    )
    return json.loads(resp.output_text)

def parse_receipt_from_image(image_bytes: bytes, mime: str = "image/jpeg") -> dict:
    category_list = ", ".join(CATEGORIES)
    prompt = build_prompt(category_list=category_list, item_count=None, receipt_text=None)
    img_url = base64_data_url(image_bytes, mime)

    resp = client.responses.create(
        model=MODEL_VISION,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt + "\n\nThe receipt is in the attached image. Read it carefully."},
                    {"type": "input_image", "image_url": img_url},
                ],
            }
        ],
        text={"format": {"type": "json_object"}},
        temperature=0.1,
    )
    return json.loads(resp.output_text)

# ===================== TAX LOGIC =====================
def compute_costco_tax_from_taxable_items(parsed_data: dict) -> dict:
    """
    Costco rule:
    - Only items with taxable=True (A marker) contribute to taxable subtotal.
    - Compute tax using taxable subtotal:
        - Prefer receipt tax if present -> tax_rate = tax / taxable_sum
        - Else use (total - subtotal) if available
    - Writes:
        parsed_data["tax"] (recomputed)
        each item["item_tax"] (per-item tax)
    """
    store = str(parsed_data.get("store", "")).strip().lower()
    if store != "costco":
        # For non-Costco, keep tax as parsed
        parsed_data["tax"] = float(parsed_data.get("tax", 0.0) or 0.0)
        items = parsed_data.get("items", []) or []
        for it in items:
            it["item_tax"] = 0.0
        return parsed_data

    items = parsed_data.get("items", []) or []
    if not items:
        parsed_data["tax"] = float(parsed_data.get("tax", 0.0) or 0.0)
        return parsed_data

    taxable_sum = 0.0
    for it in items:
        price = float(it.get("price", 0.0) or 0.0)
        if bool(it.get("taxable")) and price > 0:
            taxable_sum += price

    # If no taxable items, tax must be 0 for Costco
    if taxable_sum <= 0:
        parsed_data["tax"] = 0.0
        for it in items:
            it["item_tax"] = 0.0
        return parsed_data

    receipt_tax = float(parsed_data.get("tax", 0.0) or 0.0)
    subtotal = float(parsed_data.get("subtotal", 0.0) or 0.0)
    total = float(parsed_data.get("total", 0.0) or 0.0)

    # Infer a tax rate
    if receipt_tax > 0:
        tax_rate = receipt_tax / taxable_sum
    else:
        implied_tax = max(total - subtotal, 0.0)
        tax_rate = implied_tax / taxable_sum if implied_tax > 0 else 0.0

    computed_tax_total = round(taxable_sum * tax_rate, 2)
    parsed_data["tax"] = computed_tax_total

    # Allocate per item
    for it in items:
        price = float(it.get("price", 0.0) or 0.0)
        if bool(it.get("taxable")) and price > 0 and tax_rate > 0:
            it["item_tax"] = round(price * tax_rate, 2)
        else:
            it["item_tax"] = 0.0

    return parsed_data

def add_tax_as_item(df_items: pd.DataFrame, tax_value: float) -> pd.DataFrame:
    """Append a Sales Tax line as its own item/category, if tax_value > 0."""
    if tax_value and tax_value > 0:
        tax_row = pd.DataFrame([{"name": "Sales Tax", "category": "Tax", "price": float(tax_value), "taxable": False, "item_tax": 0.0}])
        return pd.concat([df_items, tax_row], ignore_index=True)
    return df_items

def split_subtotal_tax(edited_df: pd.DataFrame) -> tuple[float, float, float]:
    """Compute subtotal (excluding Tax), tax (Tax category), total."""
    df = edited_df.copy()
    df["price"] = pd.to_numeric(df.get("price"), errors="coerce")

    tax_mask = df["category"].astype(str).str.strip().str.lower() == "tax"

    tax_value = float(df.loc[tax_mask, "price"].sum() or 0.0)
    subtotal = float(df.loc[~tax_mask, "price"].sum() or 0.0)
    total = subtotal + tax_value
    return subtotal, tax_value, total

# ===================== DB SAVE/EDIT =====================
def save_to_database(items_df: pd.DataFrame, receipt_date: date):
    st.subheader("Saving to database")
    if st.session_state.parsed_data is None:
        st.error("Parse a receipt first!")
        return

    items_df = items_df.copy()
    items_df["price"] = pd.to_numeric(items_df["price"], errors="coerce")
    items_df = items_df.dropna(subset=["price"])
    items_df = items_df[items_df["price"] >= 0]

    new_subtotal, tax_value, new_total = split_subtotal_tax(items_df)

    session = SessionLocal()
    try:
        store_name = str(st.session_state.parsed_data.get("store", "Costco")).strip() or "Costco"
        receipt = Receipt(
            store=store_name,
            date=receipt_date,
            subtotal=float(new_subtotal),
            tax=float(tax_value),
            total=float(new_total),
            pdf_path="uploaded",
        )
        session.add(receipt)
        session.flush()

        for _, item in items_df.iterrows():
            session.add(
                ReceiptItem(
                    receipt_id=receipt.id,
                    date=receipt_date,
                    name=str(item.get("name", "")).strip(),
                    category=str(item.get("category", "Other")).strip() or "Other",
                    price=float(item.get("price")),
                )
            )

        session.commit()
        st.success(f"Receipt #{receipt.id} saved! Total: ${new_total:,.2f} on {receipt_date.isoformat()}")
        st.balloons()

        st.session_state.parsed_data = None
        st.session_state.df_to_save = None
        load_dynamic_categories()
        st.rerun()

    except Exception as e:
        session.rollback()
        st.error(f"Save failed: {e}")
    finally:
        session.close()

def delete_all_data():
    session = SessionLocal()
    try:
        session.query(ReceiptItem).delete()
        session.query(Receipt).delete()
        session.commit()
        st.sidebar.success("✅ All data deleted successfully!")
        load_dynamic_categories()
        st.rerun()
    except Exception as e:
        session.rollback()
        st.sidebar.error(f"Error during cleanup: {e}")
    finally:
        session.close()

def update_receipt_metadata(receipt_id: int, new_store: str, new_date: date, new_total: float):
    session: Session = SessionLocal()
    try:
        r = session.get(Receipt, receipt_id)
        if not r:
            st.sidebar.error(f"Could not find Receipt with ID: {receipt_id}")
            return

        r.store = new_store
        r.date = new_date
        r.total = float(new_total)

        for item in r.items:
            item.date = new_date

        session.commit()
        st.sidebar.success(f"Updated Receipt ID {receipt_id}.")
        st.rerun()
    except Exception as e:
        session.rollback()
        st.sidebar.error(f"Metadata update error: {e}")
    finally:
        session.close()

def update_item_category(item_id: int, new_category: str):
    session: Session = SessionLocal()
    try:
        item = session.get(ReceiptItem, item_id)
        cleaned = new_category.strip()
        if not cleaned:
            st.sidebar.error("New category cannot be empty.")
            return
        if not item:
            st.sidebar.error(f"Could not find Receipt Item with ID: {item_id}")
            return

        item.category = cleaned
        session.commit()
        st.sidebar.success(f"Item ID {item_id} recategorized to {cleaned}.")
        load_dynamic_categories()
        st.rerun()
    except Exception as e:
        session.rollback()
        st.sidebar.error(f"Category update error: {e}")
    finally:
        session.close()

# ===================== SIDEBAR: INPUT + MGMT =====================
st.sidebar.header("🧾 Add Receipt")

tab_photo, tab_upload = st.sidebar.tabs(["📷 Take Photo", "📄 Upload File"])

with tab_photo:
    st.caption("Best on phone. Allow camera permission in your browser.")
    camera_file = st.camera_input("Take a photo of the receipt")

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload a receipt (PDF or image)",
        type=["pdf", "png", "jpg", "jpeg"],
    )

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Data Management & Editing")

if os.path.exists(DB_PATH) and inspect(engine).has_table("receipts"):
    receipts_df = pd.read_sql_table("receipts", engine, columns=["id", "store", "date", "total"])
    if not receipts_df.empty:
        receipts_df["date"] = pd.to_datetime(receipts_df["date"]).dt.date
        receipts_df["label"] = (
            "ID " + receipts_df["id"].astype(str)
            + " | " + receipts_df["store"].astype(str)
            + " | " + receipts_df["date"].astype(str)
            + " | $" + receipts_df["total"].round(2).astype(str)
        )

        st.sidebar.subheader("1. Edit Receipt Header")
        selected_label = st.sidebar.selectbox(
            "Select Receipt",
            receipts_df["label"],
            index=None,
            placeholder="Select a receipt...",
            key="select_receipt",
        )
        if selected_label:
            selected_id = int(selected_label.split(" | ")[0].replace("ID ", ""))
            receipt_to_edit = receipts_df[receipts_df["id"] == selected_id][["store", "date", "total"]].copy()

            edited_receipt_df = st.sidebar.data_editor(
                receipt_to_edit,
                hide_index=True,
                column_config={
                    "store": st.column_config.TextColumn("Store", required=True),
                    "date": st.column_config.DateColumn("Date", required=True),
                    "total": st.column_config.NumberColumn("Total", required=True, format="$%.2f"),
                },
                key=f"edit_receipt_{selected_id}",
            )

            if st.sidebar.button("Update Receipt", type="primary", use_container_width=True):
                update_receipt_metadata(
                    receipt_id=selected_id,
                    new_store=edited_receipt_df.iloc[0]["store"],
                    new_date=edited_receipt_df.iloc[0]["date"],
                    new_total=float(edited_receipt_df.iloc[0]["total"]),
                )

        st.sidebar.markdown("---")

    items_df = pd.read_sql_table(
        "receipt_items", engine, columns=["id", "receipt_id", "name", "category", "price", "date"]
    )
    if not items_df.empty:
        items_df["label"] = (
            "Item ID " + items_df["id"].astype(str)
            + " | " + items_df["name"].astype(str).str[:30] + "..."
            + " | Cat: " + items_df["category"].astype(str)
        )

        st.sidebar.subheader("2. Edit Item Category")
        selected_item_label = st.sidebar.selectbox(
            "Select Item",
            items_df["label"],
            index=None,
            placeholder="Select an item...",
            key="select_item",
        )
        if selected_item_label:
            selected_item_id = int(selected_item_label.split(" | ")[0].replace("Item ID ", ""))
            current_category = items_df[items_df["id"] == selected_item_id]["category"].iloc[0]

            new_cat_select = st.sidebar.selectbox(
                "Existing Category",
                CATEGORIES,
                index=CATEGORIES.index(current_category)
                if current_category in CATEGORIES
                else (CATEGORIES.index("Other") if "Other" in CATEGORIES else 0),
                key=f"cat_select_{selected_item_id}",
            )
            st.sidebar.caption("OR create a new one:")
            new_cat_text = st.sidebar.text_input("Custom Category", value="", key=f"cat_text_{selected_item_id}")

            final_category = new_cat_text.strip() or new_cat_select
            if st.sidebar.button("Update Item Category", use_container_width=True):
                update_item_category(selected_item_id, final_category)

    st.sidebar.markdown("---")
    st.sidebar.subheader("3. ⚠️ Clear Database")
    if st.sidebar.button("⚠️ DELETE ALL RECEIPTS", use_container_width=True):
        delete_all_data()
else:
    st.sidebar.info("No receipts saved yet.")

# ===================== MAIN UI =====================
st.title("Receipt Classifier")
st.caption("Photo/PDF → Parse → Edit → Save | Analytics")

# Decide which input we got
input_kind = None
image_bytes = None
image_mime = None
pdf_file = None
upload_preview_name = None

if camera_file is not None:
    input_kind = "camera"
    image_bytes = camera_file.getvalue()
    image_mime = camera_file.type or "image/jpeg"
    upload_preview_name = "Camera photo"
elif uploaded_file is not None:
    ft = (uploaded_file.type or "").lower()
    fn = (uploaded_file.name or "").lower()
    upload_preview_name = uploaded_file.name

    if ft == "application/pdf" or fn.endswith(".pdf"):
        input_kind = "pdf"
        pdf_file = uploaded_file
    elif ft.startswith("image/") or fn.endswith((".png", ".jpg", ".jpeg")):
        input_kind = "image"
        image_bytes = uploaded_file.getvalue()
        image_mime = uploaded_file.type or "image/jpeg"

if input_kind in ("camera", "image") and image_bytes:
    st.image(image_bytes, caption=upload_preview_name or "Receipt image", use_container_width=True)

if input_kind:
    if st.button("Parse & Categorize", type="primary"):
        st.session_state.df_to_save = None
        st.session_state.parsed_data = None

        try:
            if input_kind == "pdf":
                with st.spinner("Reading PDF..."):
                    text = extract_text_from_pdf(pdf_file)
                    if not text.strip():
                        st.error("This PDF has no selectable text. Use photo mode or upload an image instead.")
                        st.stop()
                    item_count = get_item_count(text)

                with st.spinner("Asking AI (text)..."):
                    parsed_data = parse_receipt_from_text(text, item_count)

            else:
                with st.spinner("Asking AI (vision) from photo..."):
                    parsed_data = parse_receipt_from_image(image_bytes=image_bytes, mime=image_mime)

            # Apply Costco A-tax rule -> compute tax from taxable items
            parsed_data = compute_costco_tax_from_taxable_items(parsed_data)

            # Apply category memory AFTER tax computation (doesn't affect taxable flags)
            parsed_data["items"] = apply_category_memory(parsed_data)

            # Build DF and add tax as its own category row
            df_items = pd.DataFrame(parsed_data.get("items", []))
            tax_value = float(parsed_data.get("tax", 0.0) or 0.0)
            df_items = add_tax_as_item(df_items, tax_value)

            st.session_state.parsed_data = parsed_data
            st.session_state.df_to_save = df_items

        except Exception as e:
            st.error(f"Parsing failed: {e}")
            st.stop()

    # Editing UI
    if (
        st.session_state.parsed_data is not None
        and st.session_state.df_to_save is not None
        and not st.session_state.df_to_save.empty
    ):
        st.markdown("---")
        st.subheader("Confirm & Edit Parsed Data")

        df = st.session_state.df_to_save.copy()
        df["price"] = pd.to_numeric(df.get("price"), errors="coerce")
        df = df.dropna(subset=["price"])
        df = df[df["price"] >= 0]
        st.session_state.df_to_save = df

        parsed_date_str = st.session_state.parsed_data.get("receipt_date")
        try:
            parsed_date = pd.to_datetime(parsed_date_str).date()
        except Exception:
            parsed_date = date.today()
            st.warning("Could not extract date. Using today.")

        receipt_date = st.date_input("Confirm/Edit Receipt Date", parsed_date, key="receipt_date_input")

        st.subheader("Edit Items")
        edited_df = st.data_editor(
            st.session_state.df_to_save,
            num_rows="dynamic",
            column_config={
                "name": st.column_config.TextColumn("Item", required=True),
                "category": st.column_config.TextColumn("Category", required=True),
                "price": st.column_config.NumberColumn("Price", required=True, format="$%.2f"),
                "taxable": st.column_config.CheckboxColumn("Taxable (A)", default=False),
                "item_tax": st.column_config.NumberColumn("Item Tax", format="$%.2f"),
            },
            key="receipt_editor",
        )
        st.session_state.df_to_save = edited_df

        # Compute subtotal/tax/total from table (Tax category included as its own row)
        current_subtotal, current_tax, current_total = split_subtotal_tax(edited_df)

        c1, c2, c3 = st.columns(3)
        c1.metric("Subtotal (Excl Tax)", f"${current_subtotal:,.2f}")
        c2.metric("Tax (Auto from A)", f"${current_tax:,.2f}")
        c3.metric("Total (Recalc)", f"${current_total:,.2f}")

        st.download_button(
            "Download CSV",
            edited_df.to_csv(index=False).encode("utf-8"),
            file_name=f"receipt_{date.today().isoformat()}.csv",
        )

        st.markdown("---")
        if st.button("SAVE TO DATABASE (Permanent)", type="primary", use_container_width=True):
            save_to_database(st.session_state.df_to_save, receipt_date)

else:
    st.info("Use the sidebar to take a photo (phone) or upload a PDF/image.")

# ===================== ANALYTICS =====================
st.markdown("---")
st.subheader("Spending Analytics")

if os.path.exists(DB_PATH) and inspect(engine).has_table("receipts"):
    items = pd.read_sql_table("receipt_items", engine)
    receipts = pd.read_sql_table("receipts", engine)

    if not items.empty:
        items["price"] = pd.to_numeric(items["price"], errors="coerce")
        items["date"] = pd.to_datetime(items["date"])
        items["month_label"] = items["date"].dt.strftime("%Y-%m")

        total = items["price"].sum()
        a1, a2, a3 = st.columns(3)
        a1.metric("Total Spent (Incl Tax)", f"${total:,.2f}")
        a2.metric("Receipts", len(receipts))
        a3.metric("Items (Incl Tax row)", len(items))

        st.markdown("---")
        col1, col2 = st.columns(2)
        cat = items.groupby("category")["price"].sum().sort_values(ascending=False)
        col1.subheader("Category Totals")
        col1.dataframe(cat.reset_index().style.format({"price": "${:,.2f}"}), use_container_width=True)
        col2.plotly_chart(px.pie(cat, names=cat.index, values=cat.values, title="Category Breakdown"), use_container_width=True)

        st.markdown("---")
        monthly = items.groupby("month_label")["price"].sum().reset_index()
        st.subheader("Total Spending by Month")
        st.dataframe(
            monthly.rename(columns={"month_label": "Month", "price": "Total"}).style.format({"Total": "${:,.2f}"}),
            use_container_width=True,
        )
        fig = px.bar(monthly, x="month_label", y="price", title="Monthly Spending", text="price")
        fig.update_traces(texttemplate="$%{text:,.2f}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

st.caption(f"Database: {os.path.abspath(DB_PATH)}")