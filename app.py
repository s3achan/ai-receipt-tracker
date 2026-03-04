import os
import re
import io
import json
import base64
import uuid
import hashlib
import logging
from datetime import date, datetime
from typing import Optional
import pandas as pd
import streamlit as st
import pdfplumber
import plotly.express as px
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from sqlalchemy import text


import boto3
from botocore.exceptions import ClientError

from sqlalchemy.orm import Session
from sqlalchemy import create_engine, ForeignKey, inspect, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker, relationship, Mapped, mapped_column
from sqlalchemy import Integer, String, Float, Date as SQLDate, DateTime, Boolean

# ===================== LOGGING =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ===================== PAGE CONFIG =====================
st.set_page_config(page_title="Receipt Classifier", page_icon="🧾", layout="wide")
load_dotenv()

# ===================== CONFIG =====================
MODEL_VISION = "gpt-4.1-mini"
MODEL_TEXT   = "gpt-4o-mini"

# Cost estimates per 1M tokens (USD) — update as needed
COST_PER_1M_INPUT  = {"gpt-4.1-mini": 0.40, "gpt-4o-mini": 0.15}
COST_PER_1M_OUTPUT = {"gpt-4.1-mini": 1.60, "gpt-4o-mini": 0.60}

MONTHLY_COST_LIMIT_USD = 20.0   # warn if exceeded
MAX_FILE_SIZE_MB       = 10

# ===================== DEMO MODE =====================
DEMO_MODE = True  # Set to False to enable full data management features

INITIAL_CATEGORIES = [
    "Meat & Seafood",
    "Produce",
    "Dairy & Eggs",
    "Bakery & Bread",
    "Frozen Foods",
    "Pantry & Dry Goods",
    "Snacks & Candy",
    "Beverages & Coffee",
    "Household & Cleaning",
    "Paper & Laundry",
    "Health & Beauty",
    "Vitamins & Supplements",
    "Baby",
    "Clothing & Apparel",
    "Electronics & Office",
    "Garden & Outdoor",
    "Auto & Hardware",
    "Tax",
    "Refund",
    "Other",
]
CATEGORIES = list(INITIAL_CATEGORIES)

# ===================== SECRETS =====================
def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    try:
        v = st.secrets.get(name)
        if v is not None and str(v).strip():
            return str(v).strip()
    except Exception:
        pass
    v = os.getenv(name, default)
    return str(v).strip() if v and str(v).strip() else default

OPENAI_API_KEY     = get_secret("OPENAI_API_KEY")
AWS_ACCESS_KEY_ID  = get_secret("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = get_secret("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = get_secret("AWS_DEFAULT_REGION", "us-east-1")
S3_BUCKET          = get_secret("S3_BUCKET")
APP_PASSWORD       = get_secret("APP_PASSWORD")        # optional password gate
DB_URL             = get_secret("DATABASE_URL")         # PostgreSQL on prod, SQLite fallback

# ===================== AUTHENTICATION =====================
def check_auth() -> bool:
    """Simple password gate. Skip if APP_PASSWORD not set."""
    if not APP_PASSWORD:
        return True
    if st.session_state.get("authenticated"):
        return True
    st.title("🧾 Receipt Classifier")
    pwd = st.text_input("Enter password", type="password")
    if st.button("Login"):
        if pwd == APP_PASSWORD:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False

if not check_auth():
    st.stop()

# ===================== OPENAI CLIENT =====================
if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY. Add it to Streamlit Cloud Secrets.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# ===================== S3 =====================
def s3_enabled() -> bool:
    return bool(AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and S3_BUCKET)

def get_s3_client():
    return boto3.client(
        "s3",
        region_name=AWS_DEFAULT_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )

def guess_ext_from_mime(mime: str) -> str:
    m = (mime or "").lower()
    if "pdf"  in m: return ".pdf"
    if "png"  in m: return ".png"
    if "jpeg" in m or "jpg" in m: return ".jpg"
    if "webp" in m: return ".webp"
    return ".bin"

def s3_upload_bytes(file_bytes: bytes, content_type: str, receipt_date: date, source: str) -> str:
    if not s3_enabled():
        raise ValueError("S3 not configured.")
    ext        = guess_ext_from_mime(content_type)
    safe_src   = re.sub(r"[^a-z0-9_-]+", "-", (source or "upload").lower()).strip("-") or "upload"
    key        = f"receipts/{receipt_date.isoformat()}/{safe_src}-{uuid.uuid4().hex}{ext}"
    s3         = get_s3_client()
    s3.upload_fileobj(
        Fileobj=io.BytesIO(file_bytes),
        Bucket=S3_BUCKET,
        Key=key,
        ExtraArgs={"ContentType": content_type or "application/octet-stream"},
    )
    logger.info("S3 upload: %s", key)
    return key

def s3_presigned_get_url(key: str, expires_seconds: int = 3600) -> str:
    return get_s3_client().generate_presigned_url(
        "get_object", Params={"Bucket": S3_BUCKET, "Key": key}, ExpiresIn=expires_seconds
    )

# ===================== DATABASE =====================
class Base(DeclarativeBase):
    pass

@st.cache_resource
def get_engine():
    url = DB_URL or "sqlite:///receipts.db"

    # Supabase sometimes provides postgres://
    url = url.replace("postgres://", "postgresql://", 1)

    # Make sure SQLAlchemy uses psycopg2 driver (recommended on Streamlit Cloud)
    if url.startswith("postgresql://") and "+psycopg2" not in url:
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)

    if url.startswith("postgresql"):
        return create_engine(
            url,
            pool_size=3,
            max_overflow=2,
            pool_timeout=30,
            pool_recycle=1800,
            pool_pre_ping=True,
            future=True,
        )

    return create_engine(url, echo=False, future=True)

engine = get_engine()

try:
    with engine.connect() as conn:
        conn.execute(text("select 1"))
    st.sidebar.success("✅ Connected to database")
except Exception as e:
    st.sidebar.error(f"❌ DB connection failed: {e}")
    st.stop()

@st.cache_resource
def get_sessionmaker():
    return sessionmaker(bind=engine, expire_on_commit=False)

SessionLocal = get_sessionmaker()

class Receipt(Base):
    __tablename__ = "receipts"
    id:        Mapped[int]            = mapped_column(Integer, primary_key=True)
    store:     Mapped[str]            = mapped_column(String, default="Costco")
    date:      Mapped[date]           = mapped_column(SQLDate, default=date.today)
    subtotal:  Mapped[Optional[float]]= mapped_column(Float)
    tax:       Mapped[Optional[float]]= mapped_column(Float)
    total:     Mapped[Optional[float]]= mapped_column(Float)
    pdf_path:  Mapped[Optional[str]]  = mapped_column(String)
    file_hash: Mapped[Optional[str]]  = mapped_column(String, index=True)
    created_at:Mapped[Optional[datetime]] = mapped_column(DateTime, default=datetime.utcnow)
    items: Mapped[list["ReceiptItem"]] = relationship(
        "ReceiptItem", back_populates="receipt", cascade="all, delete-orphan"
    )

class ReceiptItem(Base):
    __tablename__ = "receipt_items"
    id:           Mapped[int]            = mapped_column(Integer, primary_key=True)
    receipt_id:   Mapped[int]            = mapped_column(ForeignKey("receipts.id"), index=True)
    date:         Mapped[date]           = mapped_column(SQLDate, nullable=False)
    name:         Mapped[str]            = mapped_column(String, nullable=False)
    category:     Mapped[str]            = mapped_column(String, nullable=False)
    sub_category: Mapped[Optional[str]]  = mapped_column(String, nullable=True)
    price:        Mapped[Optional[float]]= mapped_column(Float)
    receipt: Mapped[Receipt] = relationship("Receipt", back_populates="items")

class ApiLog(Base):
    __tablename__ = "api_logs"
    id:           Mapped[int]            = mapped_column(Integer, primary_key=True)
    created_at:   Mapped[datetime]       = mapped_column(DateTime, default=datetime.utcnow)
    model:        Mapped[str]            = mapped_column(String)
    input_tokens: Mapped[int]            = mapped_column(Integer, default=0)
    output_tokens:Mapped[int]            = mapped_column(Integer, default=0)
    cost_usd:     Mapped[float]          = mapped_column(Float, default=0.0)
    purpose:      Mapped[Optional[str]]  = mapped_column(String)

Base.metadata.create_all(engine)

def _run_migrations():
    migrations = [
        ("receipts",      "file_hash",    "VARCHAR"),
        ("receipts",      "created_at",   "TIMESTAMP"),
        ("api_logs",      "purpose",      "VARCHAR"),
        ("receipt_items", "sub_category", "VARCHAR"),
    ]
    with engine.connect() as conn:
        for table, column, col_type in migrations:
            try:
                if str(engine.url).startswith("postgresql"):
                    conn.execute(text(
                        f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {col_type}"
                    ))
                else:
                    conn.execute(text(
                        f"ALTER TABLE {table} ADD COLUMN {column} {col_type}"
                    ))
                conn.commit()
                logger.info("Migration: added %s.%s", table, column)
            except Exception as e:
                err = str(e).lower()
                if "duplicate column" in err or "already exists" in err:
                    pass
                else:
                    logger.warning("Migration warning (%s.%s): %s", table, column, e)

_run_migrations()

# ===================== COST TRACKING =====================
def log_api_call(model: str, usage, purpose: str = ""):
    if not usage:
        return
    inp  = getattr(usage, "input_tokens",  0) or 0
    out  = getattr(usage, "output_tokens", 0) or 0
    cost = (inp / 1_000_000) * COST_PER_1M_INPUT.get(model, 0.5) + \
           (out / 1_000_000) * COST_PER_1M_OUTPUT.get(model, 1.5)
    session = SessionLocal()
    try:
        session.add(ApiLog(model=model, input_tokens=inp, output_tokens=out,
                           cost_usd=round(cost, 6), purpose=purpose))
        session.commit()
        logger.info("API call [%s] in=%d out=%d cost=$%.4f", model, inp, out, cost)
    except Exception as e:
        logger.warning("Failed to log API call: %s", e)
        session.rollback()
    finally:
        session.close()

@st.cache_data(ttl=300)
def get_monthly_cost() -> float:
    session = SessionLocal()
    try:
        first_of_month = date.today().replace(day=1)
        rows = session.query(ApiLog).filter(ApiLog.created_at >= first_of_month).all()
        return round(sum(r.cost_usd for r in rows), 4)
    except Exception:
        return 0.0
    finally:
        session.close()

# ===================== DUPLICATE DETECTION =====================
def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def find_duplicate_receipt(fhash: str) -> Optional[int]:
    session = SessionLocal()
    try:
        r = session.query(Receipt).filter(Receipt.file_hash == fhash).first()
        return r.id if r else None
    except Exception:
        return None
    finally:
        session.close()

# ===================== DYNAMIC CATEGORIES =====================
def load_dynamic_categories():
    global CATEGORIES
    session = SessionLocal()
    try:
        if inspect(engine).has_table("receipt_items"):
            rows = session.query(ReceiptItem.category).distinct().all()
            db_cats = {c[0] for c in rows if c and c[0]}
            CATEGORIES = sorted(list(set(INITIAL_CATEGORIES) | db_cats))
    except Exception as e:
        logger.warning("load_dynamic_categories: %s", e)
        CATEGORIES = list(INITIAL_CATEGORIES)
    finally:
        session.close()

def get_item_category_mapping() -> dict:
    session = SessionLocal()
    mapping: dict = {}
    try:
        if inspect(engine).has_table("receipt_items"):
            rows = (
                session.query(ReceiptItem.name, ReceiptItem.category, ReceiptItem.sub_category)
                .all()
            )
            from collections import Counter
            counts: dict = {}
            for name, category, sub_category in rows:
                key = str(name).strip().lower()
                if not key:
                    continue
                if key not in counts:
                    counts[key] = Counter()
                counts[key][(category or "Other", sub_category or "")] += 1

            for key, counter in counts.items():
                best_cat, best_sub = counter.most_common(1)[0][0]
                mapping[key] = {"category": best_cat, "sub_category": best_sub}

    except Exception as e:
        logger.warning("get_item_category_mapping: %s", e)
    finally:
        session.close()
    return mapping

load_dynamic_categories()

# ===================== SESSION STATE =====================
for _k, _v in {
    "parsed_data": None, "df_to_save": None,
    "raw_receipt_bytes": None, "raw_receipt_mime": None,
    "raw_receipt_source": None, "raw_receipt_hash": None,
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ===================== PDF / IMAGE HELPERS =====================
def extract_text_from_pdf(uploaded_file) -> str:
    uploaded_file.seek(0)
    with pdfplumber.open(uploaded_file) as pdf:
        pages = [p.extract_text() or "" for p in pdf.pages]
    return "\n\n".join(t for t in pages if t.strip()).strip()

def is_garbled_text(text: str) -> bool:
    if not text:
        return False
    return (text.count("(cid:") * 6) / max(len(text), 1) > 0.05

def compress_image(image_bytes: bytes, max_dim: int = 1600, quality: int = 85) -> bytes:
    from PIL import Image as PILImage
    img = PILImage.open(io.BytesIO(image_bytes))
    img.thumbnail((max_dim, max_dim), PILImage.LANCZOS)
    buf = io.BytesIO()
    fmt = "JPEG" if img.mode == "RGB" else "PNG"
    img.save(buf, format=fmt, quality=quality if fmt == "JPEG" else None, optimize=True)
    return buf.getvalue()

def pdf_to_image_bytes(pdf_bytes: bytes, zoom: float = 2.0) -> tuple[bytes, list[bytes]]:
    import fitz
    from PIL import Image as PILImage

    doc    = fitz.open(stream=pdf_bytes, filetype="pdf")
    mat    = fitz.Matrix(zoom, zoom)
    images, page_bytes_list = [], []

    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        img = PILImage.open(io.BytesIO(pix.tobytes("png")))
        images.append(img)
        buf = io.BytesIO(); img.save(buf, format="PNG")
        page_bytes_list.append(buf.getvalue())
    doc.close()

    if len(images) == 1:
        return page_bytes_list[0], page_bytes_list

    total_w = max(i.width  for i in images)
    total_h = sum(i.height for i in images)
    combined = PILImage.new("RGB", (total_w, total_h), (255, 255, 255))
    y = 0
    for img in images:
        combined.paste(img, (0, y)); y += img.height
    buf = io.BytesIO(); combined.save(buf, format="PNG")
    return buf.getvalue(), page_bytes_list

def extract_date_from_pdf_bytes(pdf_bytes: bytes) -> Optional[str]:
    DATE_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b")

    def _parse(match) -> Optional[str]:
        m, d, y = int(match.group(1)), int(match.group(2)), int(match.group(3))
        if y < 100: y += 2000
        if not (1 <= m <= 12 and 1 <= d <= 31 and 2000 <= y <= 2100):
            return None
        return f"{y:04d}-{m:02d}-{d:02d}"

    try:
        import fitz
        doc      = fitz.open(stream=pdf_bytes, filetype="pdf")
        all_text = "\n".join(p.get_text() for p in doc)
        doc.close()

        for line in all_text.splitlines():
            if re.search(r"APPROVED|AMOUNT|\d{4}\s+\d{3}\s+\d{3}", line, re.IGNORECASE):
                m = DATE_RE.search(line)
                if m:
                    r = _parse(m)
                    if r: return r

        for m in DATE_RE.finditer(all_text):
            r = _parse(m)
            if r: return r
    except Exception as e:
        logger.warning("extract_date_from_pdf_bytes: %s", e)
    return None

def get_item_count(text: str) -> Optional[int]:
    m = re.search(r"TOTAL NUMBER OF ITEMS SOLD\s*=\s*(\d+)", text, re.IGNORECASE)
    return int(m.group(1)) if m else None

def base64_data_url(image_bytes: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(image_bytes).decode()}"

# ===================== OPENAI — RETRY WRAPPER =====================
@retry(
    retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)
def _call_openai_vision(prompt: str, img_url: str) -> tuple[dict, object]:
    resp = client.responses.create(
        model=MODEL_VISION,
        input=[{"role": "user", "content": [
            {"type": "input_text",  "text": prompt},
            {"type": "input_image", "image_url": img_url},
        ]}],
        text={"format": {"type": "json_object"}},
        temperature=0.1,
        max_output_tokens=2000,
    )
    return json.loads(resp.output_text), getattr(resp, "usage", None)

@retry(
    retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)
def _call_openai_text(prompt: str) -> tuple[dict, object]:
    resp = client.responses.create(
        model=MODEL_TEXT,
        input=[{"role": "user", "content": prompt}],
        text={"format": {"type": "json_object"}},
        temperature=0.1,
        max_output_tokens=2000,
    )
    return json.loads(resp.output_text), getattr(resp, "usage", None)

@retry(
    retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _call_openai_date(img_url: str) -> Optional[str]:
    resp = client.responses.create(
        model=MODEL_VISION,
        input=[{"role": "user", "content": [
            {"type": "input_text",  "text": "What is the purchase date? Return ONLY YYYY-MM-DD. M/D/YY like '2/27/26' = 2026-02-27. MM/DD/YYYY like '02/23/2026' = 2026-02-23."},
            {"type": "input_image", "image_url": img_url},
        ]}],
        temperature=0.0,
        max_output_tokens=20,
    )
    m = re.search(r"\d{4}-\d{2}-\d{2}", resp.output_text.strip())
    return m.group(0) if m else None

# ===================== PROMPT =====================
def build_prompt(category_list: str, item_count: Optional[int], receipt_text: Optional[str] = None) -> str:
    count_note = ""
    if item_count:
        count_note = f'COUNT CHECK: items sold = {item_count}. Discount lines do NOT count.'

    receipt_block = ""
    if receipt_text:
        receipt_block = f"Receipt text:\n---BEGIN---\n{receipt_text}\n---END---"

    return f"""
Extract receipt details and return ONLY valid JSON.

Fields required:
- store (string)
- receipt_date (YYYY-MM-DD):
  * WEB receipt: top-left corner, M/D/YY (e.g. "2/27/26" = 2026-02-27)
  * IN-STORE receipt: near bottom next to APPROVED, MM/DD/YYYY (e.g. "02/23/2026")
  Do NOT invent a date.
- subtotal, tax, total (numbers)
- lines: ordered list of ITEM or DISCOUNT lines

ITEM: {{"type":"item","name":"...","category":"...","sub_category":"...","price":0.0,"taxable":false}}
DISCOUNT: {{"type":"discount","amount":0.0,"raw":"4.00-"}}

Rules:
- TAX MARKERS:
  * Costco: Y → taxable:true, N → taxable:false
  * Target/Walmart: T → taxable:true, N or N+ → taxable:false
- Discount line like "/2189436 4.00-" → DISCOUNT type, immediately after that item.
- Keep item price as BASE price; discounts applied in code.
- category: pick best fit from this list: {category_list}
- sub_category: be specific about what the item actually is based on its name.
  Examples by category:
  Meat & Seafood → Chicken, Beef, Pork, Salmon, Shrimp, Deli Meat, Turkey
  Produce → Berries, Apples, Bananas, Salad Mix, Spinach, Broccoli, Avocado
  Dairy & Eggs → Milk, Shredded Cheese, String Cheese, Greek Yogurt, Butter, Eggs
  Bakery & Bread → Bread, Muffins, Croissants, Tortillas, Bagels
  Frozen Foods → Frozen Meals, Frozen Vegetables, Ice Cream, Pizza, Waffles
  Pantry & Dry Goods → Pasta, Rice, Canned Tomatoes, Olive Oil, Hot Sauce, Spices
  Snacks & Candy → Chips, Mixed Nuts, Protein Bars, Cookies, Chocolate, Crackers, Trail Mix
  Beverages & Coffee → Water, Orange Juice, Coffee Pods, Energy Drinks, Sports Drinks, Wine
  Household & Cleaning → Dish Soap, Trash Bags, Ziploc Bags, Batteries, Sponges
  Paper & Laundry → Paper Towels, Toilet Paper, Laundry Pods, Dryer Sheets
  Health & Beauty → Shampoo, Lotion, Toothpaste, Razors, Face Wash, Deodorant
  Vitamins & Supplements → Vitamin D, Fish Oil, Protein Powder, Melatonin, Collagen
  Baby → Diapers, Baby Wipes, Formula, Baby Food, Baby Lotion
  Clothing & Apparel → T-Shirt, Jeans, Socks, Jacket, Kids Clothing, Shoes
  Electronics & Office → Headphones, USB Cable, TV, Printer Ink, Keyboard, Mouse
  Garden & Outdoor → Potting Soil, Plant, Garden Tools, Outdoor Furniture
  Auto & Hardware → Car Mat, Windshield Washer, Motor Oil, Power Strip
  Tax → Tax
  Refund → Refund
  Other → Other
{count_note}

TARGET / WALMART RECEIPT FORMAT:
Target and Walmart print a DEPARTMENT HEADER line (e.g. "APPAREL", "HEALTH AND BEAUTY", "HOME")
above each item. These headers are NOT items — ignore them.
The actual item line is the next line with a barcode number, item name, tax marker (T/N/N+), and price.
Example Target receipt section:
  APPAREL                          ← department header, IGNORE
  330030494 MARIO TEE SH   T $5.60  ← this is the item: name="MARIO TEE SH", taxable=true, price=5.60
  Regular Price $8.00              ← original price note, IGNORE
  HEALTH AND BEAUTY                ← department header, IGNORE
  007080144 DESITIN        N+ $6.99 ← item: name="DESITIN", taxable=false, price=6.99
Tax line format: "T = TX TAX 8.2500 on $3.60  $0.71" → tax = 0.71 (the LAST number, not the rate or taxable amount)

REFUND RECEIPTS (shows "APPROVED - REFUND" or negative TOTAL):
- subtotal, tax, total must ALL be NEGATIVE (e.g. -69.46, -3.95, -73.41)
- Every item price must be NEGATIVE (e.g. 34.99- on receipt → price: -34.99)
- Discount lines on refund receipts: amount stays POSITIVE

Return ONLY this JSON (no markdown, no explanation):
{{
  "store":"Target","receipt_date":"2026-02-24",
  "subtotal":15.59,"tax":0.71,"total":16.30,
  "lines":[
    {{"type":"item","name":"MARIO TEE SH","category":"Clothing & Apparel","sub_category":"T-Shirt","price":5.60,"taxable":true}},
    {{"type":"item","name":"DESITIN","category":"Baby","sub_category":"Baby Cream","price":6.99,"taxable":false}},
    {{"type":"item","name":"No Brand","category":"Household & Cleaning","sub_category":"Home Goods","price":3.00,"taxable":true}}
  ]
}}
{receipt_block}
""".strip()

def parse_receipt_from_text(text: str, item_count: Optional[int]) -> dict:
    prompt = build_prompt(", ".join(CATEGORIES), item_count, text)
    data, usage = _call_openai_text(prompt)
    log_api_call(MODEL_TEXT, usage, "parse_text")
    return data

def parse_receipt_from_image(image_bytes: bytes, mime: str = "image/jpeg") -> dict:
    compressed  = compress_image(image_bytes)
    img_url     = base64_data_url(compressed, "image/png")
    prompt      = build_prompt(", ".join(CATEGORIES), None) + "\n\nThe receipt is in the attached image. Read it carefully."
    data, usage = _call_openai_vision(prompt, img_url)
    log_api_call(MODEL_VISION, usage, "parse_image")
    return data

def extract_date_via_vision(pdf_bytes: bytes) -> Optional[str]:
    try:
        import fitz
        from PIL import Image as PILImage
        doc  = fitz.open(stream=pdf_bytes, filetype="pdf")
        pix  = doc[0].get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        full = PILImage.open(io.BytesIO(pix.tobytes("png")))
        doc.close()
        h, w = full.height, full.width
        top  = full.crop((0, 0, w, int(h * 0.15)))
        bot  = full.crop((0, int(h * 0.70), w, h))
        strip = PILImage.new("RGB", (w, top.height + bot.height), (255, 255, 255))
        strip.paste(top, (0, 0)); strip.paste(bot, (0, top.height))
        buf = io.BytesIO(); strip.save(buf, format="PNG")
        img_url = base64_data_url(buf.getvalue(), "image/png")
        return _call_openai_date(img_url)
    except Exception as e:
        logger.warning("extract_date_via_vision: %s", e)
        return None

# ===================== COSTCO DISCOUNT + TAX =====================
def is_refund_receipt(parsed_data: dict) -> bool:
    total = float(parsed_data.get("total") or 0)
    subtotal = float(parsed_data.get("subtotal") or 0)
    return total < 0 or subtotal < 0

def build_items_from_lines(parsed_data: dict) -> list[dict]:
    store      = parsed_data.get("store", "").strip().lower()
    is_refund  = is_refund_receipt(parsed_data)
    items_out: list[dict] = []
    last_idx = None

    for ln in (parsed_data.get("lines") or []):
        t = str(ln.get("type", "")).strip().lower()
        if t == "item":
            bp = float(ln.get("price") or 0)
            if is_refund and bp > 0:
                bp = -bp
            default_cat = "Refund" if is_refund else (str(ln.get("category", "Other")).strip() or "Other")
            ai_sub = str(ln.get("sub_category") or "").strip()
            items_out.append({
                "name":         str(ln.get("name", "")).strip(),
                "category":     default_cat,
                "sub_category": ai_sub,
                "base_price":   round(bp, 2),
                "discount":     0.0,
                "price":        round(bp, 2),
                "taxable":      bool(ln.get("taxable")),
                "item_tax":     0.0,
            })
            last_idx = len(items_out) - 1
        elif t == "discount" and store == "costco" and last_idx is not None:
            amt = abs(float(ln.get("amount") or 0))
            if amt > 0:
                prev = items_out[last_idx]
                prev["discount"] = round(prev["discount"] + amt, 2)
                prev["price"]    = round(max(prev["base_price"] - prev["discount"], 0), 2)
    return items_out

def _is_costco_store(store_val: str) -> bool:
    s = (store_val or "").strip().lower()
    return "costco" in s

def compute_tax(parsed_data: dict, items: list[dict]) -> tuple[dict, list[dict]]:
    store_raw   = parsed_data.get("store", "")
    is_costco   = _is_costco_store(store_raw)

    receipt_tax = float(parsed_data.get("tax") or 0)

    if receipt_tax <= 0:
        parsed_data["tax"] = 0.0
        for it in items:
            it["item_tax"] = 0.0
        return parsed_data, items

    priced_items = [it for it in items if float(it.get("price") or 0) > 0]

    if not priced_items:
        parsed_data["tax"] = 0.0
        for it in items:
            it["item_tax"] = 0.0
        return parsed_data, items

    if is_costco:
        taxable_items = [it for it in priced_items if bool(it.get("taxable"))]
        alloc_items   = taxable_items if taxable_items else priced_items
    else:
        has_any_taxable_flag = any("taxable" in it for it in items)
        if has_any_taxable_flag:
            taxable_items = [it for it in priced_items if bool(it.get("taxable"))]
            alloc_items   = taxable_items if taxable_items else priced_items
        else:
            alloc_items = priced_items

    alloc_sum = sum(float(it.get("price") or 0) for it in alloc_items)
    if alloc_sum <= 0:
        parsed_data["tax"] = 0.0
        for it in items:
            it["item_tax"] = 0.0
        return parsed_data, items

    tax_rate = receipt_tax / alloc_sum

    for it in items:
        it["item_tax"] = 0.0

    for it in alloc_items:
        p = float(it.get("price") or 0)
        it["item_tax"] = round(p * tax_rate, 2)

    parsed_data["tax"] = round(receipt_tax, 2)
    return parsed_data, items

def add_tax_row(df: pd.DataFrame, tax: float) -> pd.DataFrame:
    if tax != 0:
        label = "Tax Refund" if tax < 0 else "Sales Tax"
        row = pd.DataFrame([{"name": label, "category":"Tax","sub_category":"",
                             "base_price":0.0,"discount":0.0,"price":float(tax),"taxable":False,"item_tax":0.0}])
        return pd.concat([df, row], ignore_index=True)
    return df

def split_subtotal_tax(df: pd.DataFrame) -> tuple[float, float, float]:
    df = df.copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    tax_mask  = df["category"].str.strip().str.lower() == "tax"
    tax       = float(df.loc[tax_mask,  "price"].sum() or 0)
    subtotal  = float(df.loc[~tax_mask, "price"].sum() or 0)
    return subtotal, tax, subtotal + tax

def apply_category_memory(items: list[dict]) -> list[dict]:
    mapping = get_item_category_mapping()
    if not mapping:
        return items
    hits = 0
    for item in items:
        key = str(item.get("name", "")).strip().lower()
        if key in mapping:
            hist = mapping[key]
            item["category"] = hist["category"]
            if hist["sub_category"]:
                item["sub_category"] = hist["sub_category"]
            hits += 1
    if hits:
        st.info(f"✅ Category memory applied to {hits} item(s) from history.")
    return items

# ===================== STYLED TABLE HELPER =====================
CATEGORY_COLORS = {
    "Meat & Seafood":         ("#fff0f0", "#c0392b"),
    "Produce":                ("#f0fff4", "#27ae60"),
    "Dairy & Eggs":           ("#fffde7", "#f39c12"),
    "Bakery & Bread":         ("#fff8f0", "#e67e22"),
    "Frozen Foods":           ("#f0f8ff", "#2980b9"),
    "Pantry & Dry Goods":     ("#fdf5e6", "#8e6c3e"),
    "Snacks & Candy":         ("#fef9f0", "#e74c3c"),
    "Beverages & Coffee":     ("#f0f4ff", "#5b6abf"),
    "Household & Cleaning":   ("#f5f0ff", "#8e44ad"),
    "Paper & Laundry":        ("#f0faff", "#16a085"),
    "Health & Beauty":        ("#fff0f8", "#c0392b"),
    "Vitamins & Supplements": ("#f0fff8", "#1abc9c"),
    "Baby":                   ("#fff5fb", "#d35400"),
    "Clothing & Apparel":     ("#f5f5ff", "#2c3e50"),
    "Electronics & Office":   ("#f0f5ff", "#2980b9"),
    "Garden & Outdoor":       ("#f4fff0", "#27ae60"),
    "Auto & Hardware":        ("#f5f5f5", "#7f8c8d"),
    "Tax":                    ("#fafafa", "#95a5a6"),
    "Refund":                 ("#f0fff4", "#27ae60"),
    "Other":                  ("#f9f9f9", "#95a5a6"),
    "Groceries":              ("#f0fff4", "#27ae60"),
}

def build_styled_table(df: pd.DataFrame, columns: list[str], headers: list[str],
                       price_col: str = "price", show_footer: bool = True) -> str:
    """
    Renders a styled HTML table.
    columns: list of df column names to display
    headers: matching display header labels
    price_col: which column holds the numeric price (for colour + footer)
    """
    rows_html = ""
    for _, row in df.iterrows():
        cells = ""
        for i, col in enumerate(columns):
            val = row[col]
            align = "left"
            style_extra = ""
            content = str(val)

            if col == price_col:
                align = "right"
                price = float(val)
                color = "#27ae60" if price >= 0 else "#e74c3c"
                content = f'<span style="color:{color}; font-weight:700;">${abs(price):,.2f}</span>'
            elif col == "category":
                bg, fg = CATEGORY_COLORS.get(str(val), ("#f0f0f0", "#555"))
                content = f'<span style="background:{bg}; color:{fg}; padding:3px 10px; border-radius:12px; font-size:12px; font-weight:600; white-space:nowrap;">{val}</span>'
            elif col == "date":
                content = f'<span style="color:#888; font-size:12px;">{val}</span>'
            elif col == "total":
                align = "right"
                price = float(val)
                color = "#27ae60" if price >= 0 else "#e74c3c"
                content = f'<span style="color:{color}; font-weight:700;">${abs(price):,.2f}</span>'
            elif col == "receipts":
                align = "right"
                content = f'<span style="color:#555; font-weight:600;">{int(val)}</span>'

            cell_align = f"text-align:{align};"
            cells += f'<td style="padding:10px 14px; font-size:14px; {cell_align} border-bottom:1px solid #f0f0f0; color:#1a1a1a;">{content}</td>'

        rows_html += f"<tr>{cells}</tr>"

    header_cells = "".join(
        f'<th style="padding:11px 14px; text-align:{"right" if h in ("Price","Total","Receipts") else "left"}; font-size:11px; font-weight:700; color:#888; text-transform:uppercase; letter-spacing:0.6px; border-bottom:2px solid #e8e8e8;">{h}</th>'
        for h in headers
    )

    footer_html = ""
    if show_footer and price_col in df.columns:
        total_val = df[price_col].sum()
        total_color = "#27ae60" if total_val >= 0 else "#e74c3c"
        footer_html = f"""
        <tfoot>
          <tr style="background:#f8f9fa;">
            <td colspan="{len(columns)-1}" style="padding:11px 14px; font-size:13px; font-weight:700; color:#555; border-top:2px solid #e8e8e8;">Total ({len(df)} items)</td>
            <td style="padding:11px 14px; text-align:right; font-size:15px; font-weight:800; color:{total_color}; border-top:2px solid #e8e8e8;">${abs(total_val):,.2f}</td>
          </tr>
        </tfoot>"""

    return f"""
    <div style="border-radius:12px; overflow:hidden; border:1px solid #e8e8e8; box-shadow:0 2px 8px rgba(0,0,0,0.06); margin-top:8px;">
      <table style="width:100%; border-collapse:collapse; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
        <thead><tr style="background:#f8f9fa;">{header_cells}</tr></thead>
        <tbody>{rows_html}</tbody>
        {footer_html}
      </table>
    </div>"""

# ===================== ANALYTICS LOADER =====================
@st.cache_data(ttl=60)
def load_analytics():
    items    = pd.read_sql_table("receipt_items", engine)
    receipts = pd.read_sql_table("receipts", engine)
    return items, receipts

# ===================== DB HELPERS =====================
def upload_to_s3(receipt_date: date) -> Optional[str]:
    raw_bytes  = st.session_state.get("raw_receipt_bytes")
    raw_mime   = st.session_state.get("raw_receipt_mime")
    raw_source = st.session_state.get("raw_receipt_source") or "upload"
    if not raw_bytes or not raw_mime or not s3_enabled():
        return None
    try:
        key = s3_upload_bytes(raw_bytes, raw_mime, receipt_date, raw_source)
        try:
            url = s3_presigned_get_url(key)
            st.success("Receipt uploaded to S3.")
            st.link_button("View file (1h)", url)
        except Exception:
            pass
        return key
    except Exception as e:
        logger.error("S3 upload error: %s", e)
        st.warning(f"S3 upload failed: {e}")
        return None

def check_duplicate(fhash: str) -> Optional[int]:
    return find_duplicate_receipt(fhash)

def save_to_database(items_df: pd.DataFrame, receipt_date: date):
    if st.session_state.parsed_data is None:
        st.error("Parse a receipt first!")
        return

    items_df = items_df.copy()
    items_df["price"] = pd.to_numeric(items_df["price"], errors="coerce")
    items_df = items_df.dropna(subset=["price"])

    subtotal, tax, total = split_subtotal_tax(items_df)
    parsed_total = float(st.session_state.parsed_data.get("total") or 0)
    if parsed_total > 0 and abs(total - parsed_total) > 1.0:
        st.warning(f"⚠️ Recalculated total ${total:,.2f} differs from receipt total ${parsed_total:,.2f} by more than $1. Please review items.")

    s3_key     = upload_to_s3(receipt_date)
    fhash      = st.session_state.get("raw_receipt_hash")
    store_name = str(st.session_state.parsed_data.get("store", "Costco")).strip() or "Costco"

    session = SessionLocal()
    try:
        receipt = Receipt(
            store=store_name, date=receipt_date,
            subtotal=float(subtotal), tax=float(tax), total=float(total),
            pdf_path=s3_key or "uploaded", file_hash=fhash,
        )
        session.add(receipt); session.flush()
        for _, row in items_df.iterrows():
            session.add(ReceiptItem(
                receipt_id=receipt.id, date=receipt_date,
                name=str(row.get("name", "")).strip(),
                category=str(row.get("category", "Other")).strip() or "Other",
                sub_category=str(row.get("sub_category", "")).strip() or None,
                price=float(row["price"]),
            ))
        session.commit()
        st.success(f"✅ Receipt #{receipt.id} saved — ${total:,.2f} on {receipt_date.isoformat()}")
        st.balloons()
        for k in ["parsed_data","df_to_save","raw_receipt_bytes","raw_receipt_mime","raw_receipt_source","raw_receipt_hash"]:
            st.session_state[k] = None
        load_dynamic_categories()
        get_monthly_cost.clear()
        load_analytics.clear()
        st.rerun()
    except Exception as e:
        session.rollback()
        logger.error("save_to_database: %s", e)
        st.error(f"Save failed: {e}")
    finally:
        session.close()

# ===================== SIDEBAR =====================
st.sidebar.title("🧾 Receipt Classifier")

if DEMO_MODE:
    st.sidebar.info("🔒 **Demo Mode** — Data management features are visible but disabled. Full functionality available in production.")

# Cost monitor
monthly_cost = get_monthly_cost()
cost_color   = "🔴" if monthly_cost >= MONTHLY_COST_LIMIT_USD else ("🟡" if monthly_cost >= MONTHLY_COST_LIMIT_USD * 0.8 else "🟢")
st.sidebar.caption(f"{cost_color} API cost this month: **${monthly_cost:.4f}** / ${MONTHLY_COST_LIMIT_USD:.0f}")
if monthly_cost >= MONTHLY_COST_LIMIT_USD:
    st.sidebar.warning("⚠️ Monthly API cost limit reached!")

st.sidebar.markdown("---")
st.sidebar.header("Add Receipt")

uploaded_file = st.sidebar.file_uploader(
    "Upload a receipt (PDF or image)",
    type=["pdf", "png", "jpg", "jpeg"],
    help=f"Max {MAX_FILE_SIZE_MB} MB",
)

# ===================== FILE VALIDATION + PARSE =====================
input_kind        = None
image_bytes: Optional[bytes] = None
image_mime        = None

if uploaded_file is not None:
    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        st.sidebar.error(f"File too large ({file_size_mb:.1f} MB). Max {MAX_FILE_SIZE_MB} MB.")
        st.stop()

    ft        = (uploaded_file.type or "").lower()
    fn        = (uploaded_file.name or "").lower()
    raw_bytes = uploaded_file.getvalue()
    fhash     = file_hash(raw_bytes)

    dup_id = check_duplicate(fhash)
    if dup_id:
        st.sidebar.warning(f"⚠️ Possible duplicate of Receipt #{dup_id}.")

    st.session_state.raw_receipt_bytes  = raw_bytes
    st.session_state.raw_receipt_mime   = uploaded_file.type or "application/octet-stream"
    st.session_state.raw_receipt_source = "upload"
    st.session_state.raw_receipt_hash   = fhash

    if ft == "application/pdf" or fn.endswith(".pdf"):
        input_kind = "pdf"
    elif ft.startswith("image/") or fn.endswith((".png",".jpg",".jpeg")):
        input_kind = "image"
        image_bytes = raw_bytes
        image_mime  = uploaded_file.type or "image/jpeg"
        st.sidebar.image(image_bytes, caption=uploaded_file.name, width=260)

if input_kind and st.sidebar.button("🔍 Parse & Categorize", type="primary", width="stretch"):
    st.session_state.df_to_save  = None
    st.session_state.parsed_data = None

    try:
        if input_kind == "pdf":
            with st.spinner("Reading PDF…"):
                pdf_bytes_val = uploaded_file.getvalue()
                text = extract_text_from_pdf(uploaded_file)

            if not text.strip() or is_garbled_text(text):
                with st.spinner("Converting PDF pages to images…"):
                    try:
                        img_bytes, page_images = pdf_to_image_bytes(pdf_bytes_val)
                    except ImportError:
                        st.sidebar.error("PyMuPDF not installed. Run `pip install pymupdf`.")
                        st.stop()

                cols = st.columns(len(page_images))
                for i, (col, pg) in enumerate(zip(cols, page_images)):
                    col.image(pg, caption=f"Page {i+1}", width=300)

                st.session_state.raw_receipt_bytes = img_bytes
                st.session_state.raw_receipt_mime  = "image/png"

                with st.spinner("Asking AI (vision)…"):
                    parsed_data = parse_receipt_from_image(img_bytes, "image/png")

                reliable_date = extract_date_from_pdf_bytes(pdf_bytes_val)
                if not reliable_date:
                    with st.spinner("Extracting date from PDF header…"):
                        reliable_date = extract_date_via_vision(pdf_bytes_val)
                if reliable_date:
                    parsed_data["receipt_date"] = reliable_date
                    st.sidebar.info(f"📅 Date: **{reliable_date}**")
            else:
                item_count = get_item_count(text)
                with st.spinner("Asking AI (text)…"):
                    parsed_data = parse_receipt_from_text(text, item_count)

        else:
            with st.spinner("Asking AI (vision)…"):
                parsed_data = parse_receipt_from_image(image_bytes, image_mime)

        items = build_items_from_lines(parsed_data)
        items = apply_category_memory(items)
        parsed_data, items = compute_tax(parsed_data, items)

        df_items = pd.DataFrame(items)
        df_items = add_tax_row(df_items, float(parsed_data.get("tax") or 0))

        st.session_state.parsed_data = parsed_data
        st.session_state.df_to_save  = df_items
        get_monthly_cost.clear()

    except Exception as e:
        logger.error("Parsing error: %s", e, exc_info=True)
        st.sidebar.error(f"Parsing failed: {e}")
        st.stop()

st.sidebar.markdown("---")
st.sidebar.header("Data Management")

if inspect(engine).has_table("receipts"):
    receipts_sidebar_df = pd.read_sql_table("receipts", engine, columns=["id","store","date","total"])
    if not receipts_sidebar_df.empty:
        receipts_sidebar_df["date"]  = pd.to_datetime(receipts_sidebar_df["date"]).dt.date
        receipts_sidebar_df["label"] = (
            "ID " + receipts_sidebar_df["id"].astype(str) + " | " +
            receipts_sidebar_df["store"].astype(str) + " | " +
            receipts_sidebar_df["date"].astype(str) + " | $" +
            receipts_sidebar_df["total"].round(2).astype(str)
        )

        st.sidebar.subheader("Edit Receipt")
        sel_label = st.sidebar.selectbox("Select", receipts_sidebar_df["label"],
                                         index=None, placeholder="Select receipt…", key="sel_receipt",
                                         disabled=DEMO_MODE)
        if sel_label and not DEMO_MODE:
            sel_id    = int(sel_label.split(" | ")[0].replace("ID ", ""))
            edit_df   = receipts_sidebar_df[receipts_sidebar_df["id"] == sel_id][["store","date","total"]].copy()
            edited_r  = st.sidebar.data_editor(edit_df, hide_index=True, key=f"edit_r_{sel_id}",
                           column_config={
                               "store": st.column_config.TextColumn("Store", required=True),
                               "date":  st.column_config.DateColumn("Date",  required=True),
                               "total": st.column_config.NumberColumn("Total", format="$%.2f"),
                           })
            if st.sidebar.button("Update Receipt", type="primary", width="stretch"):
                pass  # disabled in demo
        else:
            st.sidebar.button("Update Receipt", type="primary", width="stretch", disabled=True)
            st.sidebar.button("🗑 Delete This Receipt", width="stretch", disabled=True)

    st.sidebar.markdown("---")
    if inspect(engine).has_table("receipt_items"):
        items_sidebar_df = pd.read_sql_table("receipt_items", engine, columns=["id","name","category"])
        if not items_sidebar_df.empty:
            items_sidebar_df["label"] = (
                "Item " + items_sidebar_df["id"].astype(str) + " | " +
                items_sidebar_df["name"].str[:25] + " | " +
                items_sidebar_df["category"]
            )
            st.sidebar.subheader("Edit Item Category")
            st.sidebar.selectbox("Select", items_sidebar_df["label"],
                                 index=None, placeholder="Select item…", key="sel_item",
                                 disabled=DEMO_MODE)
            st.sidebar.selectbox("Category", CATEGORIES, key="demo_cat_sel", disabled=DEMO_MODE)
            st.sidebar.text_input("Or custom category", key="demo_cat_txt", disabled=DEMO_MODE)
            st.sidebar.button("Update Category", width="stretch", disabled=DEMO_MODE)

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚠️ Danger Zone")
    st.sidebar.button("Delete ALL Receipts", width="stretch", disabled=DEMO_MODE)
    if DEMO_MODE:
        st.sidebar.caption("🔒 Disabled in demo mode")

# ===================== MAIN =====================
st.title("🧾 Receipt Classifier")
st.caption("Upload → Parse → Review → Save | Analytics")

if DEMO_MODE:
    st.warning("🔒 **Demo Mode** — Parsing and analytics are fully live. Saving receipts and editing data are disabled to protect the shared database.")

# ---- Review & Edit ----
if (st.session_state.parsed_data is not None
        and st.session_state.df_to_save is not None
        and not st.session_state.df_to_save.empty):

    st.markdown("---")
    st.subheader("Review & Edit")

    if is_refund_receipt(st.session_state.parsed_data):
        st.warning("↩️ **Refund receipt detected.** Items will be saved with negative prices and categorized as *Refund*. Edit categories if needed before saving.")

    df = st.session_state.df_to_save.copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])
    st.session_state.df_to_save = df

    try:
        parsed_date = pd.to_datetime(st.session_state.parsed_data.get("receipt_date")).date()
    except Exception:
        parsed_date = date.today()
        st.warning("Could not extract date — defaulting to today.")

    receipt_date = st.date_input("Receipt Date", parsed_date, key="receipt_date_input")

    edited_df = st.data_editor(
        st.session_state.df_to_save, num_rows="dynamic",
        column_config={
            "name":         st.column_config.TextColumn("Item",         required=True),
            "category":     st.column_config.SelectboxColumn("Category", options=CATEGORIES, required=True),
            "sub_category": st.column_config.TextColumn("Sub-category"),
            "base_price":   st.column_config.NumberColumn("Base $",     format="$%.2f"),
            "discount":     st.column_config.NumberColumn("Discount",   format="$%.2f"),
            "price":        st.column_config.NumberColumn("Net $",      required=True, format="$%.2f"),
            "taxable":      st.column_config.CheckboxColumn("Taxable"),
            "item_tax":     st.column_config.NumberColumn("Item Tax",   format="$%.2f"),
        },
        key="receipt_editor",
    )
    st.session_state.df_to_save = edited_df

    sub, tax, total = split_subtotal_tax(edited_df)
    parsed_total    = float(st.session_state.parsed_data.get("total") or 0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Subtotal", f"${sub:,.2f}")
    c2.metric("Tax",      f"${tax:,.2f}")
    c3.metric("Total",    f"${total:,.2f}")
    delta = total - parsed_total
    c4.metric("vs Receipt", f"${parsed_total:,.2f}", delta=f"${delta:+.2f}",
              delta_color="off" if abs(delta) < 0.02 else "inverse")

    st.download_button("⬇️ Download CSV",
        edited_df.to_csv(index=False).encode(),
        file_name=f"receipt_{date.today().isoformat()}.csv",
    )

    st.markdown("---")
    if DEMO_MODE:
        st.button("💾 Save to Database", type="primary", width="stretch", disabled=True,
                  help="🔒 Disabled in demo mode — saving is available in the full version")
    else:
        if st.button("💾 Save to Database", type="primary", width="stretch"):
            save_to_database(st.session_state.df_to_save, receipt_date)

else:
    if not uploaded_file:
        st.info("Upload a receipt PDF or image using the sidebar to get started.")

# ===================== ANALYTICS =====================
st.markdown("---")
st.subheader("📊 Spending Analytics")

if inspect(engine).has_table("receipt_items"):
    items, receipts = load_analytics()

    if not items.empty:
        items["price"]       = pd.to_numeric(items["price"], errors="coerce")
        items["date"]        = pd.to_datetime(items["date"])
        items["month_label"] = items["date"].dt.strftime("%Y-%m")

        # ---- KPI row ----
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Total Spent",    f"${items['price'].sum():,.2f}")
        a2.metric("Receipts",        len(receipts))
        a3.metric("Line Items",      len(items[items["category"].str.lower() != "tax"]))
        a4.metric("API Cost (mo.)",  f"${monthly_cost:.4f}")

        st.markdown("---")

        # ---- Tabs ----
        tab_overview, tab_category, tab_monthly, tab_store, tab_items, tab_api = st.tabs([
            "📈 Overview", "🗂 Categories", "📅 Monthly", "🏪 By Store", "🔍 Drill Down", "⚙️ API Log"
        ])

        items_no_tax = items[items["category"].str.lower() != "tax"].copy()

        # ==================== OVERVIEW ====================
        with tab_overview:
            col1, col2 = st.columns(2)

            cat = items_no_tax.groupby("category")["price"].sum().sort_values(ascending=False)

            fig_cat = px.bar(
                cat.reset_index(), x="price", y="category", orientation="h",
                title="Spend by Category", text="price",
                color="price", color_continuous_scale="Blues",
            )
            fig_cat.update_traces(
                texttemplate="$%{text:,.2f}",
                textposition="outside",
                cliponaxis=False,
            )
            fig_cat.update_layout(
                yaxis=dict(categoryorder="total ascending"), xaxis_title="", yaxis_title="",
                coloraxis_showscale=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=140, t=40, b=10), font=dict(size=13),
                uniformtext=dict(mode="hide", minsize=10),
                xaxis=dict(showticklabels=False, showgrid=False),
            )
            col1.plotly_chart(fig_cat, width="stretch")

            fig_pie = px.pie(
                cat.reset_index(), names="category", values="price",
                title="Category Share", hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Set3,
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            fig_pie.update_layout(showlegend=False, margin=dict(l=10, r=10, t=40, b=10), font=dict(size=13))
            col2.plotly_chart(fig_pie, width="stretch")

            monthly = items.groupby("month_label")["price"].sum().reset_index().sort_values("month_label")
            fig_m = px.bar(
                monthly, x="month_label", y="price", title="Monthly Spending", text="price",
                color="price", color_continuous_scale="Blues",
                category_orders={"month_label": sorted(monthly["month_label"].tolist())},
            )
            fig_m.update_traces(texttemplate="$%{text:,.2f}", textposition="outside")
            fig_m.update_layout(
                xaxis=dict(type="category", title="Month"), yaxis_title="Amount ($)",
                coloraxis_showscale=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=40, b=10), font=dict(size=13), bargap=0.3,
            )
            fig_m.update_xaxes(showgrid=False)
            fig_m.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
            st.plotly_chart(fig_m, width="stretch")

        # ==================== CATEGORIES ====================
        with tab_category:
            all_cats = sorted(items_no_tax["category"].unique().tolist())
            selected_cat = st.selectbox("Select a category to inspect", all_cats,
                                        index=all_cats.index("Other") if "Other" in all_cats else 0)

            cat_items = items_no_tax[items_no_tax["category"] == selected_cat].copy()
            cat_items = cat_items.sort_values("price", ascending=False)

            st.caption(f"**{len(cat_items)} items** totalling **${cat_items['price'].sum():,.2f}** in *{selected_cat}*")

            top_items = cat_items.groupby("name")["price"].sum().sort_values(ascending=False).head(15)
            fig_items = px.bar(
                top_items.reset_index(), x="price", y="name", orientation="h",
                title=f"Top items in {selected_cat}", text="price",
                color="price", color_continuous_scale="Teal",
            )
            fig_items.update_traces(
                texttemplate="$%{text:,.2f}",
                textposition="outside",
                cliponaxis=False,
            )
            fig_items.update_layout(
                yaxis=dict(categoryorder="total ascending"), xaxis_title="", yaxis_title="",
                coloraxis_showscale=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=140, t=40, b=10), font=dict(size=13),
                uniformtext=dict(mode="hide", minsize=10),
                xaxis=dict(showticklabels=False, showgrid=False),
            )
            st.plotly_chart(fig_items, width="stretch")

            # Editable table — only shown in non-demo mode
            if DEMO_MODE:
                st.caption("🔒 *Inline recategorization is available in the full version.*")
            else:
                st.markdown("**Recategorize items — edit the Category column then click Save Changes**")
                edit_cat_df = cat_items[["id", "name", "category", "sub_category", "price", "date"]].copy() if "sub_category" in cat_items.columns else cat_items[["id", "name", "category", "price", "date"]].copy()
                edit_cat_df["date"] = edit_cat_df["date"].dt.date

                edited_cat = st.data_editor(
                    edit_cat_df,
                    column_config={
                        "id":       st.column_config.NumberColumn("ID",    disabled=True),
                        "name":     st.column_config.TextColumn("Item",    disabled=True),
                        "date":     st.column_config.DateColumn("Date",    disabled=True),
                        "price":    st.column_config.NumberColumn("Price", disabled=True, format="$%.2f"),
                        "category": st.column_config.SelectboxColumn("Category", options=CATEGORIES, required=True),
                    },
                    hide_index=True,
                    num_rows="fixed",
                    key=f"cat_editor_{selected_cat}",
                    width="stretch",
                )

                if st.button("💾 Save Category Changes", type="primary"):
                    changed = edited_cat[edited_cat["category"] != cat_items["category"].values]
                    if changed.empty:
                        st.info("No changes detected.")
                    else:
                        session = SessionLocal()
                        try:
                            for _, row in changed.iterrows():
                                item = session.get(ReceiptItem, int(row["id"]))
                                if item:
                                    item.category = row["category"]
                            session.commit()
                            load_dynamic_categories()
                            load_analytics.clear()
                            st.success(f"Updated {len(changed)} item(s).")
                            st.rerun()
                        except Exception as e:
                            session.rollback()
                            st.error(f"Save failed: {e}")
                        finally:
                            session.close()

            # Styled preview table
            st.markdown("**All items in this category**")
            cat_display = cat_items[["name","category","price"]].copy() if "sub_category" not in cat_items.columns else cat_items[["name","category","sub_category","price"]].copy()
            if "sub_category" in cat_display.columns:
                tbl = build_styled_table(cat_display, ["name","category","sub_category","price"], ["Item","Category","Sub-category","Price"])
            else:
                tbl = build_styled_table(cat_display, ["name","category","price"], ["Item","Category","Price"])
            import streamlit.components.v1 as components
            components.html(tbl, height=min(60 + len(cat_display) * 46, 800), scrolling=True)

        # ==================== MONTHLY ====================
        with tab_monthly:
            months = sorted(items["month_label"].unique().tolist(), reverse=True)
            sel_month = st.selectbox("Select month", months)

            month_items = items_no_tax[items_no_tax["month_label"] == sel_month].copy()

            col1, col2, col3 = st.columns(3)
            col1.metric("Total",    f"${month_items['price'].sum():,.2f}")
            col2.metric("Items",     len(month_items))
            col3.metric("Receipts",  month_items["receipt_id"].nunique() if "receipt_id" in month_items.columns else "—")

            m_cat = month_items.groupby("category")["price"].sum().sort_values(ascending=False)
            fig_mcat = px.bar(
                m_cat.reset_index(), x="price", y="category", orientation="h",
                title=f"Category breakdown — {sel_month}", text="price",
                color="price", color_continuous_scale="Purples",
            )
            fig_mcat.update_traces(
                texttemplate="$%{text:,.2f}",
                textposition="outside",
                cliponaxis=False,
            )
            fig_mcat.update_layout(
                yaxis=dict(categoryorder="total ascending"), xaxis_title="", yaxis_title="",
                coloraxis_showscale=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=140, t=40, b=10), font=dict(size=13),
                uniformtext=dict(mode="hide", minsize=10),
                xaxis=dict(showticklabels=False, showgrid=False),
            )
            st.plotly_chart(fig_mcat, width="stretch")

            st.markdown("**All items this month**")
            display_df = month_items[["name","category","price"]].sort_values("price", ascending=False).copy()
            tbl = build_styled_table(display_df, ["name","category","price"], ["Item","Category","Price"])
            import streamlit.components.v1 as components
            components.html(tbl, height=min(60 + len(display_df) * 46, 800), scrolling=True)

        # ==================== BY STORE ====================
        with tab_store:
            if "receipt_id" in items_no_tax.columns and not receipts.empty:
                receipts_slim = receipts[["id","store"]].rename(columns={"id":"receipt_id"})
                items_with_store = items_no_tax.merge(receipts_slim, on="receipt_id", how="left")
            else:
                items_with_store = items_no_tax.copy()
                items_with_store["store"] = "Unknown"

            store_summary = (
                items_with_store.groupby("store")["price"]
                .agg(total="sum", visits=lambda x: x.index.nunique())
                .reset_index()
                .sort_values("total", ascending=False)
            )
            receipt_counts = items_with_store.groupby("store")["receipt_id"].nunique().reset_index()
            receipt_counts.columns = ["store","receipts"]
            store_summary = store_summary.merge(receipt_counts, on="store")
            store_summary = store_summary[["store","total","receipts"]].sort_values("total", ascending=False)

            fig_store = px.bar(
                store_summary, x="total", y="store", orientation="h",
                title="Total Spend by Store", text="total",
                color="total", color_continuous_scale="Teal",
            )
            fig_store.update_traces(texttemplate="$%{text:,.2f}", textposition="outside", cliponaxis=False)
            fig_store.update_layout(
                yaxis=dict(categoryorder="total ascending"), xaxis_title="", yaxis_title="",
                coloraxis_showscale=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=140, t=40, b=10), font=dict(size=13),
                uniformtext=dict(mode="hide", minsize=10),
                xaxis=dict(showticklabels=False, showgrid=False),
            )
            st.plotly_chart(fig_store, width="stretch")

            store_tbl = build_styled_table(
                store_summary, ["store","receipts","total"], ["Store","Receipts","Total"],
                price_col="total", show_footer=False,
            )
            import streamlit.components.v1 as components
            components.html(store_tbl, height=min(60 + len(store_summary) * 46, 400), scrolling=False)

            st.markdown("---")

            stores = sorted(items_with_store["store"].unique().tolist())
            sel_store = st.selectbox("Drill into store", stores)
            store_items = items_with_store[items_with_store["store"] == sel_store]

            sc1, sc2 = st.columns(2)
            sc1.metric("Total", f"${store_items['price'].sum():,.2f}")
            sc2.metric("Receipts", store_items["receipt_id"].nunique())

            store_cat = store_items.groupby("category")["price"].sum().sort_values(ascending=False)
            fig_sc = px.bar(
                store_cat.reset_index(), x="price", y="category", orientation="h",
                title=f"Category breakdown — {sel_store}", text="price",
                color="price", color_continuous_scale="Oranges",
            )
            fig_sc.update_traces(texttemplate="$%{text:,.2f}", textposition="outside", cliponaxis=False)
            fig_sc.update_layout(
                yaxis=dict(categoryorder="total ascending"), xaxis_title="", yaxis_title="",
                coloraxis_showscale=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=140, t=40, b=10), font=dict(size=13),
                uniformtext=dict(mode="hide", minsize=10),
                xaxis=dict(showticklabels=False, showgrid=False),
            )
            st.plotly_chart(fig_sc, width="stretch")

            store_monthly = store_items.groupby("month_label")["price"].sum().reset_index().sort_values("month_label")
            if not store_monthly.empty:
                fig_sm = px.bar(
                    store_monthly, x="month_label", y="price",
                    title=f"Monthly spend — {sel_store}", text="price",
                    color="price", color_continuous_scale="Oranges",
                    category_orders={"month_label": sorted(store_monthly["month_label"].tolist())},
                )
                fig_sm.update_traces(texttemplate="$%{text:,.2f}", textposition="outside", cliponaxis=False)
                fig_sm.update_layout(
                    xaxis=dict(type="category", title="Month"), yaxis_title="Amount ($)",
                    coloraxis_showscale=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=10, r=10, t=40, b=10), font=dict(size=13), bargap=0.3,
                )
                fig_sm.update_xaxes(showgrid=False)
                fig_sm.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
                st.plotly_chart(fig_sm, width="stretch")

        # ==================== DRILL DOWN ====================
        with tab_items:
            if DEMO_MODE:
                st.caption("🔒 *Recategorization saving is disabled in demo mode.*")
            st.markdown("**Search and recategorize any item across all receipts**")

            search = st.text_input("🔎 Search item name", placeholder="e.g. chicken, detergent…")
            filter_cat = st.multiselect("Filter by category", CATEGORIES, default=[])

            drill_df = items_no_tax.copy()
            if search:
                drill_df = drill_df[drill_df["name"].str.contains(search, case=False, na=False)]
            if filter_cat:
                drill_df = drill_df[drill_df["category"].isin(filter_cat)]

            drill_df = drill_df[["id","name","category","price","date"]].copy()
            drill_df["date"] = drill_df["date"].dt.date
            drill_df = drill_df.sort_values("price", ascending=False)

            st.caption(f"{len(drill_df)} items found | Total: ${drill_df['price'].sum():,.2f}")

            if not DEMO_MODE:
                edited_drill = st.data_editor(
                    drill_df,
                    column_config={
                        "id":       st.column_config.NumberColumn("ID",    disabled=True),
                        "name":     st.column_config.TextColumn("Item",    disabled=True),
                        "date":     st.column_config.DateColumn("Date",    disabled=True),
                        "price":    st.column_config.NumberColumn("Price", disabled=True, format="$%.2f"),
                        "category": st.column_config.SelectboxColumn("Category", options=CATEGORIES, required=True),
                    },
                    hide_index=True,
                    num_rows="fixed",
                    key="drill_editor",
                    width="stretch",
                )
                if st.button("💾 Save Changes", type="primary", key="drill_save"):
                    original_cats = drill_df.set_index("id")["category"]
                    changed = edited_drill[edited_drill.apply(
                        lambda r: r["category"] != original_cats.get(r["id"], r["category"]), axis=1
                    )]
                    if changed.empty:
                        st.info("No changes detected.")
                    else:
                        session = SessionLocal()
                        try:
                            for _, row in changed.iterrows():
                                item = session.get(ReceiptItem, int(row["id"]))
                                if item:
                                    item.category = row["category"]
                            session.commit()
                            load_dynamic_categories()
                            load_analytics.clear()
                            st.success(f"Updated {len(changed)} item(s).")
                            st.rerun()
                        except Exception as e:
                            session.rollback()
                            st.error(f"Save failed: {e}")
                        finally:
                            session.close()

            # Styled table (always shown; only interactive in non-demo mode)
            if not drill_df.empty:
                drill_display = drill_df[["name","category","price","date"]].copy()
                drill_display["date"] = drill_display["date"].astype(str)
                tbl = build_styled_table(drill_display, ["name","category","date","price"], ["Item","Category","Date","Price"])
                import streamlit.components.v1 as components
                components.html(tbl, height=min(60 + len(drill_display) * 46, 800), scrolling=True)

        # ==================== API LOG ====================
        with tab_api:
            if inspect(engine).has_table("api_logs"):
                logs_df = pd.read_sql_table("api_logs", engine)
                if not logs_df.empty:
                    logs_df["created_at"] = pd.to_datetime(logs_df["created_at"])
                    logs_df = logs_df.sort_values("created_at", ascending=False).head(50)

                    MODEL_COLORS = {
                        "gpt-4.1-mini": ("#eef4ff", "#2563eb"),
                        "gpt-4o-mini":  ("#f0fdf4", "#16a34a"),
                    }
                    PURPOSE_COLORS = {
                        "parse_text":   ("#fff7ed", "#c2410c"),
                        "parse_image":  ("#fdf4ff", "#7e22ce"),
                    }

                    rows_html = ""
                    for _, row in logs_df.iterrows():
                        ts     = str(row["created_at"])[:19].replace("T", " ")
                        model  = str(row["model"])
                        purpose= str(row.get("purpose") or "—")
                        inp    = int(row.get("input_tokens") or 0)
                        out    = int(row.get("output_tokens") or 0)
                        cost   = float(row.get("cost_usd") or 0)

                        mbg, mfg = MODEL_COLORS.get(model, ("#f0f0f0", "#555"))
                        pbg, pfg = PURPOSE_COLORS.get(purpose, ("#f5f5f5", "#666"))

                        cost_color = "#dc2626" if cost > 0.005 else "#16a34a"

                        rows_html += f"""
                        <tr>
                          <td style="padding:10px 14px; font-size:13px; color:#555; border-bottom:1px solid #f0f0f0; white-space:nowrap;">{ts}</td>
                          <td style="padding:10px 14px; border-bottom:1px solid #f0f0f0;">
                            <span style="background:{mbg}; color:{mfg}; padding:3px 10px; border-radius:12px; font-size:12px; font-weight:600;">{model}</span>
                          </td>
                          <td style="padding:10px 14px; border-bottom:1px solid #f0f0f0;">
                            <span style="background:{pbg}; color:{pfg}; padding:3px 10px; border-radius:12px; font-size:12px; font-weight:600;">{purpose}</span>
                          </td>
                          <td style="padding:10px 14px; text-align:right; font-size:13px; color:#444; border-bottom:1px solid #f0f0f0;">{inp:,}</td>
                          <td style="padding:10px 14px; text-align:right; font-size:13px; color:#444; border-bottom:1px solid #f0f0f0;">{out:,}</td>
                          <td style="padding:10px 14px; text-align:right; font-size:13px; font-weight:700; color:{cost_color}; border-bottom:1px solid #f0f0f0;">${cost:.5f}</td>
                        </tr>"""

                    total_cost = logs_df["cost_usd"].sum()
                    total_inp  = int(logs_df["input_tokens"].sum())
                    total_out  = int(logs_df["output_tokens"].sum())

                    api_table_html = f"""
                    <div style="border-radius:12px; overflow:hidden; border:1px solid #e8e8e8; box-shadow:0 2px 8px rgba(0,0,0,0.06); margin-top:8px;">
                      <table style="width:100%; border-collapse:collapse; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
                        <thead>
                          <tr style="background:#f8f9fa;">
                            {"".join(f'<th style="padding:11px 14px; text-align:{"right" if h in ("In Tokens","Out Tokens","Cost (USD)") else "left"}; font-size:11px; font-weight:700; color:#888; text-transform:uppercase; letter-spacing:0.6px; border-bottom:2px solid #e8e8e8;">{h}</th>' for h in ["Timestamp","Model","Purpose","In Tokens","Out Tokens","Cost (USD)"])}
                          </tr>
                        </thead>
                        <tbody>{rows_html}</tbody>
                        <tfoot>
                          <tr style="background:#f8f9fa;">
                            <td colspan="3" style="padding:11px 14px; font-size:13px; font-weight:700; color:#555; border-top:2px solid #e8e8e8;">Total ({len(logs_df)} calls)</td>
                            <td style="padding:11px 14px; text-align:right; font-size:13px; font-weight:700; color:#444; border-top:2px solid #e8e8e8;">{total_inp:,}</td>
                            <td style="padding:11px 14px; text-align:right; font-size:13px; font-weight:700; color:#444; border-top:2px solid #e8e8e8;">{total_out:,}</td>
                            <td style="padding:11px 14px; text-align:right; font-size:15px; font-weight:800; color:#dc2626; border-top:2px solid #e8e8e8;">${total_cost:.5f}</td>
                          </tr>
                        </tfoot>
                      </table>
                    </div>"""

                    import streamlit.components.v1 as components
                    components.html(api_table_html, height=min(60 + len(logs_df) * 46, 800), scrolling=True)
                else:
                    st.info("No API calls logged yet.")
    else:
        st.info("No receipts saved yet.")
