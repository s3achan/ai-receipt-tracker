import os
import re
import io
import json
import base64
import uuid
from datetime import date

import pandas as pd
import streamlit as st
import pdfplumber
import plotly.express as px
from dotenv import load_dotenv
from openai import OpenAI

import boto3
from botocore.exceptions import ClientError

from sqlalchemy.orm import Session
from sqlalchemy import create_engine, ForeignKey, inspect
from sqlalchemy.orm import DeclarativeBase, sessionmaker, relationship, Mapped, mapped_column
from sqlalchemy import Integer, String, Float, Date as SQLDate

# ----------------------------
# Streamlit config (ONLY ONCE)
# ----------------------------
st.set_page_config(page_title="Receipt Classifier", page_icon="", layout="wide")
load_dotenv()

# ===================== CONFIG =====================
DB_PATH = "receipts.db"

# OpenAI models
MODEL_VISION = "gpt-4.1-mini"   # Vision for photos/images
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

# ===================== SECRETS / KEYS =====================
def get_openai_key() -> str | None:
    # Streamlit secrets (Cloud or local secrets.toml)
    try:
        key = st.secrets.get("OPENAI_API_KEY")
        if key:
            return str(key)
    except Exception:
        pass
    # Fallback to env/.env (local)
    return os.getenv("OPENAI_API_KEY")

OPENAI_API_KEY = get_openai_key()
if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY. Add it to Streamlit Cloud Secrets or your .env file.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

def get_secret(name: str, default: str | None = None) -> str | None:
    try:
        v = st.secrets.get(name, None)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    except Exception:
        pass

    v = os.getenv(name, default)
    if v is None:
        return None
    v = str(v).strip()
    return v if v != "" else default

AWS_ACCESS_KEY_ID = get_secret("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = get_secret("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = get_secret("AWS_DEFAULT_REGION", "us-east-1")

# IMPORTANT: This code expects S3_BUCKET in secrets/.env
S3_BUCKET = get_secret("S3_BUCKET")  # e.g. ai-receipt-classifer

def s3_enabled() -> bool:
    return bool(AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and AWS_DEFAULT_REGION and S3_BUCKET)

def get_s3_client():
    return boto3.client(
        "s3",
        region_name=AWS_DEFAULT_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )

def guess_ext_from_mime(mime: str) -> str:
    m = (mime or "").lower()
    if "pdf" in m:
        return ".pdf"
    if "png" in m:
        return ".png"
    if "jpeg" in m or "jpg" in m:
        return ".jpg"
    if "webp" in m:
        return ".webp"
    return ".bin"

def s3_upload_bytes(file_bytes: bytes, content_type: str, receipt_date: date, source: str) -> str:
    """
    Upload bytes to S3 and return the key.
    receipts/YYYY-MM-DD/<source>-<uuid>.<ext>
    """
    if not s3_enabled():
        raise ValueError("S3 not configured (missing AWS_* or S3_BUCKET).")

    ext = guess_ext_from_mime(content_type)
    safe_source = re.sub(r"[^a-z0-9_-]+", "-", (source or "upload").lower()).strip("-") or "upload"
    key = f"receipts/{receipt_date.isoformat()}/{safe_source}-{uuid.uuid4().hex}{ext}"

    s3 = get_s3_client()
    bio = io.BytesIO(file_bytes)
    s3.upload_fileobj(
        Fileobj=bio,
        Bucket=S3_BUCKET,
        Key=key,
        ExtraArgs={"ContentType": content_type or "application/octet-stream"},
    )
    return key

def s3_presigned_get_url(key: str, expires_seconds: int = 3600) -> str:
    s3 = get_s3_client()
    return s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": S3_BUCKET, "Key": key},
        ExpiresIn=expires_seconds,
    )

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
    # stores S3 key (or "uploaded" if not using S3)
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
    price: Mapped[float | None] = mapped_column(Float)  # NET price (after discounts)
    receipt: Mapped[Receipt] = relationship("Receipt", back_populates="items")

Base.metadata.create_all(engine)

# ===================== DYNAMIC CATEGORY LOADING & MAPPING =====================
def load_dynamic_categories():
    global CATEGORIES
    session = SessionLocal()
    try:
        if inspect(engine).has_table("receipt_items"):
            db_categories = session.query(ReceiptItem.category).distinct().all()
            db_category_names = {c[0] for c in db_categories if c and c[0]}
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
if "raw_receipt_bytes" not in st.session_state:
    st.session_state.raw_receipt_bytes = None
if "raw_receipt_mime" not in st.session_state:
    st.session_state.raw_receipt_mime = None
if "raw_receipt_source" not in st.session_state:
    st.session_state.raw_receipt_source = None

# ===================== HELPERS =====================
def extract_text_from_pdf(uploaded_file) -> str:
    """
    Text-only extraction. (Scanned PDFs will return empty.)
    """
    uploaded_file.seek(0)
    with pdfplumber.open(uploaded_file) as pdf:
        pages = []
        for page in pdf.pages:
            t = page.extract_text() or ""
            if t.strip():
                pages.append(t)
        return "\n\n".join(pages).strip()

def is_garbled_text(text: str) -> bool:
    """
    Returns True if the extracted text is garbled due to custom/encoded fonts.
    Detects high ratio of (cid:N) sequences which indicate undecodable glyphs.
    """
    if not text:
        return False
    cid_count = text.count("(cid:")
    total_chars = max(len(text), 1)
    # If more than 5% of content is (cid:) sequences, it's garbled
    return (cid_count * 6) / total_chars > 0.05

def pdf_to_image_bytes(pdf_bytes: bytes, zoom: float = 2.0) -> bytes:
    """
    Convert the first page of a scanned PDF to a PNG image using PyMuPDF.
    Returns raw PNG bytes.
    """
    import fitz  # pymupdf
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    doc.close()
    return img_bytes

def get_item_count(text: str) -> int | None:
    match = re.search(r"TOTAL NUMBER OF ITEMS SOLD\s*=\s*(\d+)", text, re.IGNORECASE)
    return int(match.group(1)) if match else None

def base64_data_url(image_bytes: bytes, mime: str) -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def apply_category_memory(items: list[dict]) -> list[dict]:
    if not items:
        return items
    category_map = get_item_category_mapping()
    if not category_map:
        return items

    st.info("Applying user-defined category memory...")
    for item in items:
        nm = str(item.get("name", "")).strip()
        key = nm.lower()
        if key in category_map:
            item["category"] = category_map[key]
            st.write(f"Memory Hit: **{nm}** → **{category_map[key]}**")
    return items

# ===================== PROMPT & PARSING =====================
def build_prompt(category_list: str, item_count: int | None, receipt_text: str | None = None) -> str:
    """
    Return ordered "lines" including discount lines so we can apply:
    next line like '4.00-A' discounts previous item.
    """
    count_constraint = ""
    if item_count and item_count > 0:
        count_constraint = f"""
COUNT CHECK (Costco): The number of PURCHASED ITEMS must match "TOTAL NUMBER OF ITEMS SOLD" (which is {item_count}).
Discount lines do NOT count as items sold.
""".strip()

    receipt_block = ""
    if receipt_text is not None:
        receipt_block = f"""
Receipt text:
---BEGIN---
{receipt_text}
---END---
""".strip()

    return f"""
Extract receipt details and return ONLY valid JSON.

You must return:
- store (string)
- receipt_date (YYYY-MM-DD)
- subtotal (number)
- tax (number)
- total (number)
- lines (ordered list from top-to-bottom of the purchase section)

Each element of "lines" must be either:

ITEM line:
{{"type":"item","name":"...","category":"...","price":0.0,"taxable":false}}

DISCOUNT line:
{{"type":"discount","amount":0.0,"raw":"4.00-A"}}

Costco tax + discount rules (CRITICAL):
- If store is Costco: an item is taxable ONLY if the receipt shows an "A" marker next to that item line -> set taxable=true.
- Costco discount rule: if a DISCOUNT line immediately follows an ITEM line and looks like "4.00-A" (or similar),
  apply that discount amount to the PRECEDING ITEM (reduce its net price by that amount).
- Discount lines are NOT items.
- Keep item "price" as the base price (before discount). We'll compute net in code.

Categorization: choose ONE category from this list when possible:
{category_list}

{count_constraint}

Return ONLY JSON in this schema:
{{
  "store": "Costco",
  "receipt_date": "2025-11-13",
  "subtotal": 0.0,
  "tax": 0.0,
  "total": 0.0,
  "lines": [
    {{"type":"item","name":"", "category":"Other", "price":0.0, "taxable": false}},
    {{"type":"discount","amount":0.0,"raw":"4.00-A"}}
  ]
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

# ===================== COSTCO: DISCOUNT + TAX LOGIC =====================
def normalize_discount_amount(x) -> float:
    try:
        v = float(x or 0.0)
    except Exception:
        v = 0.0
    return abs(v)

def build_items_from_lines_with_costco_discounts(parsed_data: dict) -> list[dict]:
    """
    Convert parsed_data["lines"] -> items with:
      - base_price
      - discount
      - price (net = base - discount)
    Rule: if next line is discount like 4.00-A, discount applies to preceding item.
    """
    store = str(parsed_data.get("store", "")).strip().lower()
    lines = parsed_data.get("lines", []) or []

    items_out: list[dict] = []
    last_item_idx: int | None = None

    for ln in lines:
        t = str(ln.get("type", "")).strip().lower()

        if t == "item":
            base_price = float(ln.get("price", 0.0) or 0.0)
            item = {
                "name": str(ln.get("name", "")).strip(),
                "category": str(ln.get("category", "Other")).strip() or "Other",
                "base_price": round(base_price, 2),
                "discount": 0.0,
                "price": round(base_price, 2),  # net after discount
                "taxable": bool(ln.get("taxable", False)),
                "item_tax": 0.0,
            }
            items_out.append(item)
            last_item_idx = len(items_out) - 1
            continue

        if t == "discount" and store == "costco" and last_item_idx is not None:
            amt = normalize_discount_amount(ln.get("amount", 0.0))
            if amt <= 0:
                continue

            prev = items_out[last_item_idx]
            prev["discount"] = round(float(prev.get("discount", 0.0) or 0.0) + amt, 2)

            base_price = float(prev.get("base_price", 0.0) or 0.0)
            prev["price"] = round(max(base_price - float(prev["discount"]), 0.0), 2)

    return items_out

def compute_costco_tax_from_taxable_items(parsed_data: dict, items: list[dict]) -> tuple[dict, list[dict]]:
    """
    Costco:
    - only taxable=True items contribute (A marker)
    - tax computed from taxable NET prices (after discounts)
    - prefer parsed receipt tax to infer rate; else use (total - subtotal)
    """
    store = str(parsed_data.get("store", "")).strip().lower()

    if store != "costco":
        parsed_data["tax"] = float(parsed_data.get("tax", 0.0) or 0.0)
        for it in items:
            it["item_tax"] = 0.0
        return parsed_data, items

    taxable_sum = 0.0
    for it in items:
        p = float(it.get("price", 0.0) or 0.0)
        if bool(it.get("taxable")) and p > 0:
            taxable_sum += p

    if taxable_sum <= 0:
        parsed_data["tax"] = 0.0
        for it in items:
            it["item_tax"] = 0.0
        return parsed_data, items

    receipt_tax = float(parsed_data.get("tax", 0.0) or 0.0)
    subtotal = float(parsed_data.get("subtotal", 0.0) or 0.0)
    total = float(parsed_data.get("total", 0.0) or 0.0)

    if receipt_tax > 0:
        tax_rate = receipt_tax / taxable_sum
    else:
        implied_tax = max(total - subtotal, 0.0)
        tax_rate = implied_tax / taxable_sum if implied_tax > 0 else 0.0

    computed_tax_total = round(taxable_sum * tax_rate, 2)
    parsed_data["tax"] = computed_tax_total

    for it in items:
        p = float(it.get("price", 0.0) or 0.0)
        if bool(it.get("taxable")) and p > 0 and tax_rate > 0:
            it["item_tax"] = round(p * tax_rate, 2)
        else:
            it["item_tax"] = 0.0

    return parsed_data, items

def add_tax_as_item(df_items: pd.DataFrame, tax_value: float) -> pd.DataFrame:
    if tax_value and tax_value > 0:
        tax_row = pd.DataFrame([{
            "name": "Sales Tax",
            "category": "Tax",
            "base_price": 0.0,
            "discount": 0.0,
            "price": float(tax_value),
            "taxable": False,
            "item_tax": 0.0,
        }])
        return pd.concat([df_items, tax_row], ignore_index=True)
    return df_items

def split_subtotal_tax(edited_df: pd.DataFrame) -> tuple[float, float, float]:
    df = edited_df.copy()
    df["price"] = pd.to_numeric(df.get("price"), errors="coerce")
    tax_mask = df["category"].astype(str).str.strip().str.lower() == "tax"
    tax_value = float(df.loc[tax_mask, "price"].sum() or 0.0)
    subtotal = float(df.loc[~tax_mask, "price"].sum() or 0.0)
    total = subtotal + tax_value
    return subtotal, tax_value, total

# ===================== DB SAVE/EDIT (WITH S3 UPLOAD) =====================
def upload_receipt_to_s3_if_configured(receipt_date: date) -> str | None:
    """
    Uploads the raw uploaded file (photo/image/pdf) to S3 if configured.
    Returns S3 key or None.
    """
    raw_bytes = st.session_state.get("raw_receipt_bytes")
    raw_mime = st.session_state.get("raw_receipt_mime")
    raw_source = st.session_state.get("raw_receipt_source") or "upload"

    if not raw_bytes or not raw_mime:
        return None

    if not s3_enabled():
        st.warning("S3 not configured. Skipping upload (set AWS_* + S3_BUCKET in secrets).")
        return None

    try:
        key = s3_upload_bytes(
            file_bytes=raw_bytes,
            content_type=raw_mime,
            receipt_date=receipt_date,
            source=raw_source,
        )
        st.success("Uploaded receipt file to S3.")
        try:
            url = s3_presigned_get_url(key, expires_seconds=3600)
            st.link_button("View uploaded file (1 hour)", url)
        except Exception:
            pass
        return key
    except ClientError as e:
        st.error(f"S3 upload failed: {e}")
        return None
    except Exception as e:
        st.error(f"S3 upload failed: {e}")
        return None

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

    # Upload original file to S3 (photo/pdf) and keep its key
    s3_key = upload_receipt_to_s3_if_configured(receipt_date)

    session = SessionLocal()
    try:
        store_name = str(st.session_state.parsed_data.get("store", "Costco")).strip() or "Costco"
        receipt = Receipt(
            store=store_name,
            date=receipt_date,
            subtotal=float(new_subtotal),
            tax=float(tax_value),
            total=float(new_total),
            pdf_path=s3_key or "uploaded",
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

        # Clear state
        st.session_state.parsed_data = None
        st.session_state.df_to_save = None
        st.session_state.raw_receipt_bytes = None
        st.session_state.raw_receipt_mime = None
        st.session_state.raw_receipt_source = None

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
        st.sidebar.success("All data deleted successfully!")
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
        cleaned = (new_category or "").strip()
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
st.sidebar.header("Add Receipt")

tab_photo, tab_upload = st.sidebar.tabs(["Take Photo", "Upload File"])

with tab_photo:
    st.caption("Best on phone. Allow camera permission in your browser.")
    camera_file = st.camera_input("Take a photo of the receipt")

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload a receipt (PDF or image)",
        type=["pdf", "png", "jpg", "jpeg"],
    )

st.sidebar.markdown("---")
st.sidebar.header("Data Management & Editing")

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

        st.sidebar.subheader("1) Edit Receipt Header")
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

        st.sidebar.subheader("2) Edit Item Category")
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
    st.sidebar.subheader("3) Clear Database")
    if st.sidebar.button("DELETE ALL RECEIPTS", use_container_width=True):
        delete_all_data()
else:
    st.sidebar.info("No receipts saved yet.")

# ===================== MAIN UI =====================
st.title("Receipt Classifier")
st.caption("Photo/PDF → Parse → Discounts+Tax (Costco) → Edit → Save | Analytics")

# Decide input
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

    # Save raw for S3
    st.session_state.raw_receipt_bytes = image_bytes
    st.session_state.raw_receipt_mime = image_mime
    st.session_state.raw_receipt_source = "camera"

elif uploaded_file is not None:
    ft = (uploaded_file.type or "").lower()
    fn = (uploaded_file.name or "").lower()
    upload_preview_name = uploaded_file.name

    raw_bytes = uploaded_file.getvalue()
    st.session_state.raw_receipt_bytes = raw_bytes
    st.session_state.raw_receipt_mime = uploaded_file.type or "application/octet-stream"
    st.session_state.raw_receipt_source = "upload"

    if ft == "application/pdf" or fn.endswith(".pdf"):
        input_kind = "pdf"
        pdf_file = uploaded_file
    elif ft.startswith("image/") or fn.endswith((".png", ".jpg", ".jpeg")):
        input_kind = "image"
        image_bytes = raw_bytes
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
                    pdf_bytes = uploaded_file.getvalue()
                    text = extract_text_from_pdf(uploaded_file)

                if not text.strip() or is_garbled_text(text):
                    # Scanned or encoded-font PDF — convert first page to image and use vision model
                    with st.spinner("PDF text unreadable, converting to image for vision model..."):
                        try:
                            img_bytes = pdf_to_image_bytes(pdf_bytes, zoom=2.0)
                        except ImportError:
                            st.error(
                                "PyMuPDF is not installed. Run `pip install pymupdf` to support scanned PDFs."
                            )
                            st.stop()

                    st.image(img_bytes, caption="Scanned PDF (page 1 preview)", use_container_width=True)

                    # Store rendered image for S3 (overrides original PDF bytes)
                    st.session_state.raw_receipt_bytes = img_bytes
                    st.session_state.raw_receipt_mime = "image/png"

                    with st.spinner("Asking AI (vision) from scanned PDF..."):
                        parsed_data = parse_receipt_from_image(image_bytes=img_bytes, mime="image/png")
                else:
                    item_count = get_item_count(text)
                    with st.spinner("Asking AI (text)..."):
                        parsed_data = parse_receipt_from_text(text, item_count)

            else:
                with st.spinner("Asking AI (vision) from photo..."):
                    parsed_data = parse_receipt_from_image(image_bytes=image_bytes, mime=image_mime)

            # Build items and apply Costco discounts
            items = build_items_from_lines_with_costco_discounts(parsed_data)

            # Apply category memory
            items = apply_category_memory(items)

            # Compute Costco tax from taxable items
            parsed_data, items = compute_costco_tax_from_taxable_items(parsed_data, items)

            df_items = pd.DataFrame(items)
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

    st.subheader("Edit Items (base_price / discount / net price)")
    edited_df = st.data_editor(
        st.session_state.df_to_save,
        num_rows="dynamic",
        column_config={
            "name": st.column_config.TextColumn("Item", required=True),
            "category": st.column_config.TextColumn("Category", required=True),
            "base_price": st.column_config.NumberColumn("Base Price", format="$%.2f"),
            "discount": st.column_config.NumberColumn("Discount", format="$%.2f"),
            "price": st.column_config.NumberColumn("Net Price", required=True, format="$%.2f"),
            "taxable": st.column_config.CheckboxColumn("Taxable (A)", default=False),
            "item_tax": st.column_config.NumberColumn("Item Tax", format="$%.2f"),
        },
        key="receipt_editor",
    )
    st.session_state.df_to_save = edited_df

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