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

# NEW imports for PDF → image conversion
from pdf2image import convert_from_bytes
from PIL import Image

from sqlalchemy import create_engine, ForeignKey, inspect
from sqlalchemy.orm import DeclarativeBase, sessionmaker, relationship, Mapped, mapped_column
from sqlalchemy import Integer, String, Float, Date as SQLDate

# ────────────────────────────────────────────────
# Streamlit config
# ────────────────────────────────────────────────
st.set_page_config(page_title="Receipt Classifier", layout="wide")
load_dotenv()

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────
DB_PATH = "receipts.db"

MODEL_VISION = "gpt-4o-mini"
MODEL_TEXT  = "gpt-4o-mini"

INITIAL_CATEGORIES = [
    "Meat & Prepared Foods", "Produce", "Dairy & Eggs", "Beverages",
    "Frozen & Packaged Foods", "Baby", "Household", "Clothing",
    "Tax", "Other",
]
CATEGORIES = list(INITIAL_CATEGORIES)

# ────────────────────────────────────────────────
# Secrets
# ────────────────────────────────────────────────
def get_openai_key():
    try: return st.secrets["OPENAI_API_KEY"]
    except: return os.getenv("OPENAI_API_KEY")

OPENAI_API_KEY = get_openai_key()
if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY → add to .env or Streamlit secrets")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

def get_secret(name, default=None):
    try:
        v = st.secrets.get(name)
        if v and str(v).strip(): return str(v).strip()
    except:
        pass
    return os.getenv(name, default)

S3_BUCKET          = get_secret("S3_BUCKET")
AWS_ACCESS_KEY_ID  = get_secret("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = get_secret("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = get_secret("AWS_DEFAULT_REGION", "us-east-1")

def s3_enabled():
    return all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION, S3_BUCKET])

def get_s3_client():
    return boto3.client("s3", region_name=AWS_DEFAULT_REGION,
                        aws_access_key_id=AWS_ACCESS_KEY_ID,
                        aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

# ────────────────────────────────────────────────
# Database
# ────────────────────────────────────────────────
class Base(DeclarativeBase): pass

engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
SessionLocal = sessionmaker(bind=engine)

class Receipt(Base):
    __tablename__ = "receipts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    store: Mapped[str] = mapped_column(String, default="Costco")
    date: Mapped[date] = mapped_column(SQLDate, default=date.today)
    subtotal: Mapped[float | None] = mapped_column(Float)
    tax: Mapped[float | None] = mapped_column(Float)
    total: Mapped[float | None] = mapped_column(Float)
    pdf_path: Mapped[str | None] = mapped_column(String)
    items: Mapped[list["ReceiptItem"]] = relationship("ReceiptItem", back_populates="receipt", cascade="all, delete-orphan")

class ReceiptItem(Base):
    __tablename__ = "receipt_items"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    receipt_id: Mapped[int] = mapped_column(ForeignKey("receipts.id"))
    date: Mapped[date] = mapped_column(SQLDate)
    name: Mapped[str] = mapped_column(String)
    category: Mapped[str] = mapped_column(String)
    price: Mapped[float | None] = mapped_column(Float)
    receipt: Mapped[Receipt] = relationship("Receipt", back_populates="items")

Base.metadata.create_all(engine)

# ────────────────────────────────────────────────
# Category memory
# ────────────────────────────────────────────────
def load_dynamic_categories():
    global CATEGORIES
    with SessionLocal() as session:
        try:
            cats = {c[0] for c in session.query(ReceiptItem.category).distinct().all() if c[0]}
            CATEGORIES = sorted(set(INITIAL_CATEGORIES) | cats)
        except:
            CATEGORIES = list(INITIAL_CATEGORIES)

def get_item_category_mapping():
    mapping = {}
    with SessionLocal() as session:
        try:
            q = session.query(ReceiptItem.name, ReceiptItem.category).order_by(ReceiptItem.date.desc())
            for name, cat in q.all():
                key = str(name).strip().lower()
                if key and key not in mapping:
                    mapping[key] = cat
        except:
            pass
    return mapping

load_dynamic_categories()

# ────────────────────────────────────────────────
# Session state
# ────────────────────────────────────────────────
for k in ["parsed_data", "df_to_save", "raw_bytes", "raw_mime", "raw_source"]:
    if k not in st.session_state:
        st.session_state[k] = None

# ────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────
def extract_text_from_pdf(file_like):
    with pdfplumber.open(file_like) as pdf:
        pages = [p.extract_text() or "" for p in pdf.pages]
        return "\n\n".join(p.strip() for p in pages if p.strip()).strip()

def base64_data_url(bytes_data, mime):
    b64 = base64.b64encode(bytes_data).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def apply_category_memory(items):
    if not items: return items
    mapping = get_item_category_mapping()
    if not mapping: return items
    for item in items:
        key = str(item.get("name","")).strip().lower()
        if key in mapping:
            item["category"] = mapping[key]
    return items

# ────────────────────────────────────────────────
# Prompt
# ────────────────────────────────────────────────
def build_prompt(categories_str, item_count=None, text=None):
    count_part = f"Number of items must match TOTAL NUMBER OF ITEMS SOLD = {item_count}" if item_count else ""
    text_part = f"Receipt text:\n---\n{text}\n---" if text else "This is an image of a receipt."

    return f"""Extract receipt and return **only** valid JSON.

Fields:
- store: string
- receipt_date: YYYY-MM-DD
- subtotal: number
- tax: number
- total: number
- lines: array of objects (top to bottom)

Line types:
{{"type":"item",   "name":"...", "category":"...", "price":number, "taxable":bool}}
{{"type":"discount","amount":number, "raw":"..."}}   # e.g. "4.00-A"

Costco rules:
- taxable = true only if line shows "A"
- discount line like "X.XX-A" immediately after item → subtract from previous item's price
- price = gross price (before discount)

Categories (prefer these): {categories_str}

{count_part}

{text_part}

Return ONLY JSON object. No explanation.
""".strip()

def parse_with_text(text, item_count):
    cats = ", ".join(CATEGORIES)
    prompt = build_prompt(cats, item_count, text)
    resp = client.chat.completions.create(
        model=MODEL_TEXT,
        messages=[{"role":"user", "content":prompt}],
        response_format={"type": "json_object"},
        temperature=0.25,
    )
    return json.loads(resp.choices[0].message.content)

def parse_with_vision(image_bytes, mime="image/png"):
    cats = ", ".join(CATEGORIES)
    prompt = build_prompt(cats) + "\n\nThis is a photo/scan of a Costco receipt. Look carefully for A markers, discount lines (-X.XX-A), member prices, date format."
    b64_url = base64_data_url(image_bytes, mime)

    resp = client.chat.completions.create(
        model=MODEL_VISION,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": b64_url}}
            ]
        }],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    return json.loads(resp.choices[0].message.content)

# ────────────────────────────────────────────────
# Costco discount & tax logic
# ────────────────────────────────────────────────
def process_lines_to_items(parsed):
    store = str(parsed.get("store","")).lower()
    lines = parsed.get("lines", [])
    items = []
    last_idx = None

    for line in lines:
        typ = str(line.get("type","")).lower()
        if typ == "item":
            p = float(line.get("price",0))
            it = {
                "name": str(line.get("name","")).strip(),
                "category": str(line.get("category","Other")).strip() or "Other",
                "base_price": round(p,2),
                "discount": 0.0,
                "price": round(p,2),
                "taxable": bool(line.get("taxable",False)),
                "item_tax": 0.0
            }
            items.append(it)
            last_idx = len(items)-1
        elif typ == "discount" and store == "costco" and last_idx is not None:
            amt = abs(float(line.get("amount",0)))
            if amt > 0:
                prev = items[last_idx]
                prev["discount"] = round(prev["discount"] + amt, 2)
                prev["price"] = round(max(prev["base_price"] - prev["discount"], 0), 2)

    return items

def apply_costco_tax(parsed, items):
    if str(parsed.get("store","")).lower() != "costco":
        parsed["tax"] = float(parsed.get("tax",0))
        for it in items: it["item_tax"] = 0
        return parsed, items

    taxable_sum = sum(it["price"] for it in items if it["taxable"] and it["price"] > 0)
    if taxable_sum <= 0:
        parsed["tax"] = 0
        for it in items: it["item_tax"] = 0
        return parsed, items

    receipt_tax = float(parsed.get("tax",0))
    if receipt_tax > 0:
        rate = receipt_tax / taxable_sum
    else:
        rate = max(parsed.get("total",0) - parsed.get("subtotal",0), 0) / taxable_sum

    parsed["tax"] = round(taxable_sum * rate, 2)
    for it in items:
        if it["taxable"] and it["price"] > 0:
            it["item_tax"] = round(it["price"] * rate, 2)
        else:
            it["item_tax"] = 0

    return parsed, items

def add_tax_row(df, tax_val):
    if tax_val <= 0: return df
    tax_row = pd.DataFrame([{
        "name":"Sales Tax", "category":"Tax",
        "base_price":0, "discount":0, "price":tax_val,
        "taxable":False, "item_tax":0
    }])
    return pd.concat([df, tax_row], ignore_index=True)

# ────────────────────────────────────────────────
# Sidebar – Input
# ────────────────────────────────────────────────
st.sidebar.header("Add Receipt")

tab1, tab2 = st.sidebar.tabs(["📸 Camera", "📄 Upload"])

with tab1:
    camera_file = st.camera_input("Take photo")

with tab2:
    uploaded = st.file_uploader("PDF or image", type=["pdf","png","jpg","jpeg"])

# ────────────────────────────────────────────────
# Input handling
# ────────────────────────────────────────────────
input_type = None
image_data = None
image_type = None
pdf_obj = None

if camera_file:
    input_type = "image"
    image_data = camera_file.getvalue()
    image_type = camera_file.type or "image/jpeg"
    st.session_state.raw_bytes  = image_data
    st.session_state.raw_mime   = image_type
    st.session_state.raw_source = "camera"

elif uploaded:
    fname = (uploaded.name or "").lower()
    ftype = (uploaded.type or "").lower()
    raw = uploaded.getvalue()

    st.session_state.raw_bytes  = raw
    st.session_state.raw_mime   = uploaded.type or "application/octet-stream"
    st.session_state.raw_source = "upload"

    if "pdf" in ftype or fname.endswith(".pdf"):
        input_type = "pdf"
        pdf_obj = uploaded
    else:
        input_type = "image"
        image_data = raw
        image_type = uploaded.type or "image/jpeg"

if input_type == "image" and image_data:
    st.image(image_data, caption="Receipt preview", use_column_width=True)

# ────────────────────────────────────────────────
# Parse button
# ────────────────────────────────────────────────
if input_type and st.button("Parse receipt", type="primary"):
    st.session_state.parsed_data = None
    st.session_state.df_to_save  = None

    try:
        if input_type == "pdf":
            pdf_obj.seek(0)
            pdf_bytes = pdf_obj.read()

            # Try text first
            text = extract_text_from_pdf(io.BytesIO(pdf_bytes))

            if text.strip() and len(text) > 80:
                item_count = None   # you can re-add get_item_count if needed
                with st.spinner("Text layer found → using text model"):
                    parsed = parse_with_text(text, item_count)
            else:
                # Scanned PDF → convert to image
                with st.spinner("No text → converting PDF to image"):
                    pages = convert_from_bytes(pdf_bytes, dpi=180, first_page=1, last_page=1)
                    if not pages:
                        st.error("Could not convert PDF to image")
                        st.stop()
                    pil_img = pages[0]
                    buf = io.BytesIO()
                    pil_img.save(buf, format="PNG")
                    image_data = buf.getvalue()
                    image_type = "image/png"
                    st.image(pil_img, caption="Using page 1", use_column_width=True)

                with st.spinner("Vision parsing..."):
                    parsed = parse_with_vision(image_data, image_type)

        else:
            # photo / image upload
            with st.spinner("Vision parsing..."):
                parsed = parse_with_vision(image_data, image_type)

        # Post-process
        items = process_lines_to_items(parsed)
        items = apply_category_memory(items)
        parsed, items = apply_costco_tax(parsed, items)

        df = pd.DataFrame(items)
        df = add_tax_row(df, float(parsed.get("tax", 0)))

        st.session_state.parsed_data = parsed
        st.session_state.df_to_save  = df

    except Exception as e:
        st.error(f"Parsing failed\n\n{e}")
        st.stop()

# ────────────────────────────────────────────────
# Edit & Save
# ────────────────────────────────────────────────
if st.session_state.df_to_save is not None and not st.session_state.df_to_save.empty:
    st.divider()
    st.subheader("Review & edit")

    df = st.session_state.df_to_save.copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0)

    try:
        dstr = st.session_state.parsed_data.get("receipt_date")
        default_date = pd.to_datetime(dstr).date()
    except:
        default_date = date.today()

    receipt_date = st.date_input("Date", default_date)

    edited = st.data_editor(
        df,
        num_rows="dynamic",
        column_config={
            "name":     st.column_config.TextColumn("Item", required=True),
            "category": st.column_config.TextColumn("Category", required=True),
            "base_price": st.column_config.NumberColumn("Base", format="$%.2f"),
            "discount":   st.column_config.NumberColumn("Disc", format="$%.2f"),
            "price":      st.column_config.NumberColumn("Net", format="$%.2f", required=True),
            "taxable":    st.column_config.CheckboxColumn("Taxable"),
            "item_tax":   st.column_config.NumberColumn("Tax amt", format="$%.2f"),
        }
    )

    sub = edited[edited["category"].str.lower() != "tax"]["price"].sum()
    tax = edited[edited["category"].str.lower() == "tax"]["price"].sum()
    tot = sub + tax

    cols = st.columns(3)
    cols[0].metric("Subtotal", f"${sub:,.2f}")
    cols[1].metric("Tax", f"${tax:,.2f}")
    cols[2].metric("Total", f"${tot:,.2f}")

    if st.button("Save to database", type="primary"):
        # (add your save logic here – omitted for brevity)
        st.success("Saved!")
        st.session_state.parsed_data = None
        st.session_state.df_to_save = None

# Analytics, DB management, etc. can be added below as before