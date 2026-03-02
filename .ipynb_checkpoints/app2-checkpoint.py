import os
import io
import re
import json
import uuid
import base64
from datetime import date

import pandas as pd
import streamlit as st
import plotly.express as px
import pdfplumber

from dotenv import load_dotenv
from openai import OpenAI

import boto3
from botocore.exceptions import ClientError

import fitz  # PyMuPDF
from PIL import Image

from sqlalchemy.orm import Session
from sqlalchemy import create_engine, ForeignKey, inspect
from sqlalchemy.orm import DeclarativeBase, sessionmaker, relationship, Mapped, mapped_column
from sqlalchemy import Integer, String, Float, Date as SQLDate

# ------------------ ENV ------------------
load_dotenv()

def _get_secret(name: str, default: str | None = None) -> str | None:
    # Streamlit Cloud secrets first
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv(name, default)

OPENAI_API_KEY = _get_secret("OPENAI_API_KEY")

# ------------------ CATEGORIES ------------------
INITIAL_CATEGORIES = [
    "Meat & Prepared Foods",
    "Produce",
    "Dairy & Eggs",
    "Beverages",
    "Frozen & Packaged Foods",
    "Baby",
    "Household",
    "Clothing",
    "Other"
]
CATEGORIES = list(INITIAL_CATEGORIES)

# ------------------ DB (SQLite) ------------------
class Base(DeclarativeBase):
    pass

DB_PATH = "receipts.db"
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
    # Store S3 URI here (pdf or image)
    pdf_path: Mapped[str | None] = mapped_column(String)
    items: Mapped[list["ReceiptItem"]] = relationship(
        "ReceiptItem",
        back_populates="receipt",
        cascade="all, delete-orphan"
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

# ------------------ CATEGORY MEMORY ------------------
def load_dynamic_categories():
    global CATEGORIES
    session = SessionLocal()
    try:
        if inspect(engine).has_table("receipt_items"):
            db_categories = session.query(ReceiptItem.category).distinct().all()
            db_category_names = {c[0] for c in db_categories if c and c[0]}
            all_categories = set(INITIAL_CATEGORIES) | db_category_names
            CATEGORIES = sorted(list(all_categories))
    except Exception:
        CATEGORIES = list(INITIAL_CATEGORIES)
    finally:
        session.close()

def get_item_category_mapping() -> dict:
    """Map item name -> most recent category."""
    session = SessionLocal()
    mapping = {}
    try:
        if inspect(engine).has_table("receipt_items"):
            q = session.query(ReceiptItem.name, ReceiptItem.category, Receipt.date).join(Receipt).order_by(Receipt.date.desc())
            for name, category, _d in q.all():
                normalized = (name or "").strip().lower()
                if normalized and normalized not in mapping:
                    mapping[normalized] = category
    except Exception:
        pass
    finally:
        session.close()
    return mapping

load_dynamic_categories()

# ------------------ S3 HELPERS ------------------
def get_s3_client():
    region = _get_secret("AWS_DEFAULT_REGION") or _get_secret("AWS_REGION") or "us-east-1"
    access_key = _get_secret("AWS_ACCESS_KEY_ID")
    secret_key = _get_secret("AWS_SECRET_ACCESS_KEY")
    session_token = _get_secret("AWS_SESSION_TOKEN")

    if access_key and secret_key:
        return boto3.client(
            "s3",
            region_name=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            aws_session_token=session_token,
        )

    # fallback: IAM role / default chain
    return boto3.client("s3", region_name=region)

def upload_bytes_to_s3(data: bytes, *, bucket: str, key: str, content_type: str) -> str:
    s3 = get_s3_client()
    try:
        s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
        return f"s3://{bucket}/{key}"
    except ClientError as e:
        raise RuntimeError(f"S3 upload failed: {e}")

def upload_streamlit_file_to_s3(uploaded_file, *, bucket: str, prefix: str = "receipts") -> str:
    uploaded_file.seek(0)
    data = uploaded_file.read()

    filename = uploaded_file.name or "upload"
    ext = os.path.splitext(filename)[1].lower() or ""
    if ext not in [".pdf", ".png", ".jpg", ".jpeg"]:
        ext = ".bin"

    content_type = {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }.get(ext, "application/octet-stream")

    key = f"{prefix}/{date.today().isoformat()}/{uuid.uuid4().hex}{ext}"
    return upload_bytes_to_s3(data, bucket=bucket, key=key, content_type=content_type)

def upload_image_bytes_to_s3(image_bytes: bytes, *, bucket: str, prefix: str = "receipts", ext: str = ".jpg") -> str:
    ext = ext.lower()
    if ext not in [".jpg", ".jpeg", ".png"]:
        ext = ".jpg"
    content_type = "image/png" if ext == ".png" else "image/jpeg"
    key = f"{prefix}/{date.today().isoformat()}/{uuid.uuid4().hex}{ext}"
    return upload_bytes_to_s3(image_bytes, bucket=bucket, key=key, content_type=content_type)

# ------------------ OCR + TEXT EXTRACTION ------------------
def looks_like_garbage(text: str) -> bool:
    if not text or not text.strip():
        return True
    printable = sum(ch.isprintable() for ch in text)
    ratio = printable / max(len(text), 1)
    return ratio < 0.85

def pdf_to_images_pymupdf(pdf_bytes: bytes, zoom: float = 2.0) -> list[Image.Image]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images: list[Image.Image] = []
    mat = fitz.Matrix(zoom, zoom)

    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        images.append(img)

    return images

def ocr_image_openai(pil_img: Image.Image) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing (set in Secrets or .env).")

    client = OpenAI(api_key=OPENAI_API_KEY)

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    data_url = f"data:image/png;base64,{b64}"

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract all readable receipt/invoice text from this image. Return ONLY the raw text."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
    )
    return (resp.choices[0].message.content or "").strip()

def extract_text_from_pdf(uploaded_file) -> str:
    """Try pdfplumber first; fallback to OCR using PyMuPDF+OpenAI Vision."""
    uploaded_file.seek(0)
    pdf_bytes = uploaded_file.read()

    # 1) text-based extraction
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            text = "\n\n".join((p.extract_text() or "") for p in pdf.pages).strip()
        if text and not looks_like_garbage(text):
            return text
    except Exception:
        text = ""

    # 2) OCR fallback
    images = pdf_to_images_pymupdf(pdf_bytes, zoom=2.0)
    out = []
    for i, img in enumerate(images, start=1):
        page_text = ocr_image_openai(img)
        if page_text:
            out.append(f"--- PAGE {i} ---\n{page_text}")
    return "\n\n".join(out).strip()

def extract_text_from_image_file(uploaded_image_file) -> str:
    uploaded_image_file.seek(0)
    img = Image.open(uploaded_image_file).convert("RGB")
    return ocr_image_openai(img)

def extract_text_from_camera_image(camera_file) -> tuple[str, bytes]:
    """
    camera_file is a Streamlit UploadedFile-like object (png bytes).
    Returns: (text, raw_bytes)
    """
    camera_file.seek(0)
    raw = camera_file.getvalue()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    text = ocr_image_openai(img)
    return text, raw

# ------------------ RECEIPT PARSING ------------------
def get_item_count(text: str) -> int | None:
    match = re.search(r"TOTAL NUMBER OF ITEMS SOLD\s*=\s*(\d+)", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

def parse_receipt(text: str, item_count: int | None) -> dict:
    client = OpenAI(api_key=OPENAI_API_KEY)

    count_constraint = ""
    if item_count is not None and item_count > 0:
        count_constraint = f"""
5. **COUNT CHECK (CRITICAL):** The total number of items in the final "items" list MUST match the "TOTAL NUMBER OF ITEMS SOLD" count found on the receipt text (which is **{item_count}**).
"""

    category_list = ", ".join(CATEGORIES)

    prompt = f"""
From the receipt text, extract the items purchased, the subtotals, tax, total, and the purchase date.

Instructions:
1) Extract every line that represents a unique purchased product.
2) **DO NOT** include discounts/instant savings/negative lines in items.
3) Use the full, undiscounted item price.
4) Assign one exact category, prioritizing this list: {category_list}.
{count_constraint}

Return ONLY valid JSON, including receipt_date in YYYY-MM-DD format:
{{
  "items": [{{"name":"", "category":"", "price": 0.0}}],
  "subtotal": 0.0,
  "tax": 0.0,
  "total": 0.0,
  "receipt_date": "2025-11-13"
}}

Receipt:
---BEGIN---
{text}
---END---
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    return json.loads(resp.choices[0].message.content.strip())

def apply_category_memory(parsed_data: dict) -> list[dict]:
    items = parsed_data.get("items", [])
    if not items:
        return []

    category_map = get_item_category_mapping()
    if not category_map:
        return items

    st.info("Applying user-defined category memory...")
    for item in items:
        nm = str(item.get("name", "")).strip().lower()
        if nm in category_map:
            item["category"] = category_map[nm]
    return items

# ------------------ DB SAVE / EDIT ------------------
def save_to_database(items_df: pd.DataFrame, receipt_date: date, source_uri: str | None):
    if st.session_state.parsed_data is None:
        st.error("Parse a receipt first!")
        return

    items_df = items_df.copy()
    items_df["price"] = pd.to_numeric(items_df["price"], errors="coerce")
    items_df = items_df.dropna(subset=["price"])
    items_df = items_df[items_df["price"] >= 0]

    new_subtotal = float(items_df["price"].sum())
    tax_value = float(st.session_state.parsed_data.get("tax") or 0.0)
    new_total = new_subtotal + tax_value

    session = SessionLocal()
    try:
        receipt = Receipt(
            store="Costco",
            date=receipt_date,
            subtotal=new_subtotal,
            tax=tax_value,
            total=new_total,
            pdf_path=source_uri,  # <-- S3 URI saved here
        )
        session.add(receipt)
        session.flush()

        for _, row in items_df.iterrows():
            session.add(
                ReceiptItem(
                    receipt_id=receipt.id,
                    date=receipt_date,
                    name=str(row.get("name", "")).strip(),
                    category=str(row.get("category", "Other")).strip() or "Other",
                    price=float(row.get("price") or 0.0),
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
        st.sidebar.success("✅ All data deleted successfully.")
        load_dynamic_categories()
        st.rerun()
    except Exception as e:
        session.rollback()
        st.sidebar.error(f"Error during data cleanup: {e}")
    finally:
        session.close()

def update_receipt_metadata(receipt_id: int, new_store: str, new_date: date, new_total: float):
    session: Session = SessionLocal()
    try:
        receipt_to_update = session.get(Receipt, receipt_id)
        if receipt_to_update:
            receipt_to_update.store = new_store
            receipt_to_update.date = new_date
            receipt_to_update.total = float(new_total)

            for item in receipt_to_update.items:
                item.date = new_date

            session.commit()
            st.sidebar.success(f"Updated Receipt {receipt_id}.")
            st.rerun()
        else:
            st.sidebar.error(f"Receipt ID not found: {receipt_id}")
    except Exception as e:
        session.rollback()
        st.sidebar.error(f"Update failed: {e}")
    finally:
        session.close()

def update_item_category(item_id: int, new_category: str):
    session: Session = SessionLocal()
    try:
        item_to_update = session.get(ReceiptItem, item_id)
        cleaned = (new_category or "").strip()
        if not cleaned:
            st.sidebar.error("New category cannot be empty.")
            return

        if item_to_update:
            item_to_update.category = cleaned
            session.commit()
            st.sidebar.success(f"Item {item_id} recategorized to {cleaned}.")
            load_dynamic_categories()
            st.rerun()
        else:
            st.sidebar.error(f"Item ID not found: {item_id}")
    except Exception as e:
        session.rollback()
        st.sidebar.error(f"Category update failed: {e}")
    finally:
        session.close()

# ------------------ STREAMLIT UI ------------------
st.set_page_config(page_title="Receipt Classifier", layout="wide")

if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY (set in Streamlit Secrets or .env).")
    st.stop()

S3_BUCKET = _get_secret("S3_BUCKET_NAME")

# Session state
if "parsed_data" not in st.session_state:
    st.session_state.parsed_data = None
if "df_to_save" not in st.session_state:
    st.session_state.df_to_save = None
if "last_s3_uri" not in st.session_state:
    st.session_state.last_s3_uri = None

# Sidebar upload
st.sidebar.header("🧾 Upload Receipt")

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF / Image",
    type=["pdf", "png", "jpg", "jpeg"],
    help="PDF receipts or image receipts work. Scanned PDFs will use OCR."
)

camera_file = st.sidebar.camera_input("Or take a photo (camera)")

st.sidebar.markdown("---")
st.sidebar.caption("S3 is optional. If S3 secrets are set, the original file/image will be uploaded and the S3 URI stored in the DB.")
if not S3_BUCKET:
    st.sidebar.warning("S3_BUCKET_NAME not set → S3 upload disabled.")

# Data management
st.sidebar.header("⚙️ Data Management & Editing")

if os.path.exists(DB_PATH) and inspect(engine).has_table("receipts"):
    receipts_df = pd.read_sql_table("receipts", engine, columns=["id", "store", "date", "total", "pdf_path"])
    if not receipts_df.empty:
        receipts_df["date"] = pd.to_datetime(receipts_df["date"]).dt.date
        receipts_df["label"] = (
            "ID " + receipts_df["id"].astype(str)
            + " | " + receipts_df["store"]
            + " | " + receipts_df["date"].astype(str)
            + " | $" + receipts_df["total"].round(2).astype(str)
        )

        st.sidebar.subheader("1) Edit Receipt Header")
        selected_label = st.sidebar.selectbox(
            "Select Receipt to Edit",
            receipts_df["label"],
            index=None,
            placeholder="Select a receipt...",
            key="select_receipt_date",
        )

        if selected_label:
            selected_id = int(selected_label.split(" | ")[0].replace("ID ", ""))
            receipt_to_edit = receipts_df[receipts_df["id"] == selected_id][["store", "date", "total"]].copy()

            edited_receipt_df = st.sidebar.data_editor(
                receipt_to_edit,
                hide_index=True,
                column_config={
                    "store": st.column_config.TextColumn("Store Name", required=True),
                    "date": st.column_config.DateColumn("Purchase Date", required=True),
                    "total": st.column_config.NumberColumn("Total Price", required=True, format="$%.2f"),
                },
                key=f"edit_receipt_{selected_id}",
            )

            if st.sidebar.button("Update Receipt Metadata", type="primary", use_container_width=True):
                update_receipt_metadata(
                    receipt_id=selected_id,
                    new_store=edited_receipt_df.iloc[0]["store"],
                    new_date=edited_receipt_df.iloc[0]["date"],
                    new_total=edited_receipt_df.iloc[0]["total"],
                )

        st.sidebar.markdown("---")

    items_df = pd.read_sql_table("receipt_items", engine, columns=["id", "receipt_id", "name", "category", "price", "date"])
    if not items_df.empty and not receipts_df.empty:
        receipt_cols = ["id", "store", "date", "total"]
        items_df = pd.merge(
            items_df, receipts_df[receipt_cols],
            left_on="receipt_id", right_on="id",
            suffixes=("_item", "_receipt"),
            how="left",
        )
        items_df["label"] = (
            "Item ID " + items_df["id_item"].astype(str)
            + " | " + items_df["name"].str[:30] + "..."
            + " | Current Cat: " + items_df["category"]
        )

        st.sidebar.subheader("2) Edit Item Categories")
        selected_item_label = st.sidebar.selectbox(
            "Select Item to Recategorize",
            items_df["label"],
            index=None,
            placeholder="Select an item...",
            key="select_item_cat",
        )

        if selected_item_label:
            selected_item_id = int(selected_item_label.split(" | ")[0].replace("Item ID ", ""))
            current_category = items_df[items_df["id_item"] == selected_item_id]["category"].iloc[0]

            new_category_from_select = st.sidebar.selectbox(
                "Select Existing Category",
                CATEGORIES,
                index=CATEGORIES.index(current_category) if current_category in CATEGORIES else (CATEGORIES.index("Other") if "Other" in CATEGORIES else 0),
                key=f"edit_cat_select_{selected_item_id}",
            )

            st.sidebar.caption("OR create a new one:")
            new_category_from_text = st.sidebar.text_input("Custom Category Name", value="", key=f"edit_cat_text_{selected_item_id}")
            final_category = new_category_from_text.strip() or new_category_from_select

            if st.sidebar.button("Update Item Category", type="secondary", use_container_width=True):
                update_item_category(selected_item_id, final_category)

    st.sidebar.markdown("---")
    st.sidebar.subheader("3) ⚠️ Clear Database")
    if st.sidebar.button("⚠️ DELETE ALL RECEIPTS", use_container_width=True):
        delete_all_data()
else:
    st.sidebar.info("No receipts saved yet. Upload a PDF/image to begin.")

# Main area
st.title("Receipt Classifier")
st.caption("Upload/Camera → (Optional) S3 Upload → Parse → Edit → Save → Analytics")

# Determine input source
source_kind = None
if camera_file is not None:
    source_kind = "camera"
elif uploaded_file is not None:
    ext = os.path.splitext(uploaded_file.name or "")[1].lower()
    if ext == ".pdf":
        source_kind = "pdf"
    else:
        source_kind = "image"

if source_kind:
    if st.button("Parse & Categorize", type="primary"):
        st.session_state.df_to_save = None
        st.session_state.parsed_data = None
        st.session_state.last_s3_uri = None

        # ---- Upload original to S3 (optional) ----
        if S3_BUCKET:
            try:
                with st.spinner("Uploading original to S3..."):
                    if source_kind == "camera":
                        camera_file.seek(0)
                        raw = camera_file.getvalue()
                        # camera_input is typically PNG
                        st.session_state.last_s3_uri = upload_image_bytes_to_s3(raw, bucket=S3_BUCKET, prefix="receipts", ext=".png")
                    else:
                        st.session_state.last_s3_uri = upload_streamlit_file_to_s3(uploaded_file, bucket=S3_BUCKET, prefix="receipts")
                st.success(f"Uploaded to S3: {st.session_state.last_s3_uri}")
            except Exception as e:
                st.warning(f"S3 upload failed (continuing without S3): {e}")

        # ---- Extract text ----
        with st.spinner("Extracting text..."):
            try:
                if source_kind == "pdf":
                    text = extract_text_from_pdf(uploaded_file)
                elif source_kind == "image":
                    text = extract_text_from_image_file(uploaded_file)
                else:  # camera
                    text, _raw = extract_text_from_camera_image(camera_file)
            except Exception as e:
                st.error(f"Text extraction failed: {e}")
                st.stop()

            if not text:
                st.error("Could not extract any text. Try a clearer image/PDF.")
                st.stop()

            item_count = get_item_count(text)
            if item_count is not None and item_count > 0:
                st.info(f"Detected item count = {item_count} (will enforce count).")
            else:
                st.caption("Item count not detected; AI will infer items and exclude discounts.")

        # ---- LLM parse ----
        with st.spinner("Parsing receipt with GPT..."):
            parsed_data = parse_receipt(text, item_count)

        parsed_data["items"] = apply_category_memory(parsed_data)
        st.session_state.parsed_data = parsed_data
        st.session_state.df_to_save = pd.DataFrame(parsed_data.get("items", []))

    # ---- Edit + Save ----
    if (
        st.session_state.parsed_data is not None
        and st.session_state.df_to_save is not None
        and not st.session_state.df_to_save.empty
    ):
        st.markdown("---")
        st.subheader("Confirm & Edit Parsed Data")

        parsed_date_str = st.session_state.parsed_data.get("receipt_date")
        try:
            parsed_date = pd.to_datetime(parsed_date_str).date()
        except Exception:
            parsed_date = date.today()

        receipt_date = st.date_input("Confirm/Edit Receipt Date", parsed_date, key="receipt_date_input")

        st.subheader("Edit Items")
        edited_df = st.data_editor(
            st.session_state.df_to_save,
            num_rows="dynamic",
            column_config={"category": st.column_config.TextColumn("Category", required=True)},
            key="receipt_editor",
        )
        edited_df["price"] = pd.to_numeric(edited_df["price"], errors="coerce")
        edited_df = edited_df.dropna(subset=["price"])
        edited_df = edited_df[edited_df["price"] >= 0]
        st.session_state.df_to_save = edited_df

        current_subtotal = float(edited_df["price"].sum()) if not edited_df.empty else 0.0
        current_tax = float(st.session_state.parsed_data.get("tax") or 0.0)
        current_total = current_subtotal + current_tax

        c1, c2, c3 = st.columns(3)
        c1.metric("Subtotal (Recalc)", f"${current_subtotal:,.2f}")
        c2.metric("Tax (Parsed)", f"${current_tax:,.2f}")
        c3.metric("Total (Recalc)", f"${current_total:,.2f}")

        st.download_button(
            "Download CSV",
            edited_df.to_csv(index=False).encode(),
            file_name=f"receipt_{date.today().isoformat()}.csv",
        )

        st.markdown("---")
        if st.button("SAVE TO DATABASE (Permanent)", type="primary", use_container_width=True):
            save_to_database(
                st.session_state.df_to_save,
                receipt_date,
                source_uri=st.session_state.last_s3_uri,
            )

else:
    st.info("Upload a PDF/image or use camera input to begin.")

# ------------------ ANALYTICS ------------------
st.markdown("---")
st.subheader("Spending Analytics")

if os.path.exists(DB_PATH) and inspect(engine).has_table("receipts"):
    try:
        items = pd.read_sql_table("receipt_items", engine)
        receipts = pd.read_sql_table("receipts", engine)

        if not items.empty:
            items["price"] = pd.to_numeric(items["price"], errors="coerce")
            items = items.dropna(subset=["price"])
            items["date"] = pd.to_datetime(items["date"])
            items["month_label"] = items["date"].dt.strftime("%Y-%m")

            total_spent = float(items["price"].sum())

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Spent", f"${total_spent:,.2f}")
            c2.metric("Receipts", len(receipts))
            c3.metric("Items", len(items))

            st.markdown("---")
            cat = items.groupby("category")["price"].sum().sort_values(ascending=False)
            col1, col2 = st.columns(2)
            col1.subheader("Category Totals")
            col1.dataframe(cat.reset_index().style.format({"price": "${:,.2f}"}), use_container_width=True)
            col2.plotly_chart(px.pie(cat, names=cat.index, values=cat.values, title="Category Breakdown"), use_container_width=True)

            st.markdown("---")
            st.subheader("Spending Trends")
            monthly = items.groupby("month_label")["price"].sum().reset_index()
            st.dataframe(
                monthly.rename(columns={"month_label": "Month", "price": "Total Spent"}).style.format({"Total Spent": "${:,.2f}"}),
                use_container_width=True,
            )
            fig = px.bar(
                monthly,
                x="month_label",
                y="price",
                title="Total Spending by Month",
                labels={"month_label": "Month (YYYY-MM)", "price": "Total Spent"},
                text="price",
            )
            fig.update_traces(texttemplate="$%{text:,.2f}", textposition="outside")
            fig.update_layout(uniformtext_minsize=8, uniformtext_mode="hide")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.subheader("Monthly Breakdown by Category")
            monthly_cat = items.groupby(["month_label", "category"])["price"].sum().reset_index()
            fig_stacked = px.bar(
                monthly_cat,
                x="month_label",
                y="price",
                color="category",
                title="Monthly Spending Stacked by Category",
                labels={"month_label": "Month (YYYY-MM)", "price": "Total Spent", "category": "Category"},
            )
            fig_stacked.update_layout(xaxis={"categoryorder": "category ascending"})
            st.plotly_chart(fig_stacked, use_container_width=True)

        else:
            st.caption("No items saved yet.")

    except Exception as e:
        st.warning(f"Analytics unavailable: {e}")

st.caption(f"Database: {os.path.abspath(DB_PATH)}")