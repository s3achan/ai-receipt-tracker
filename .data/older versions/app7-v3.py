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

INITIAL_CATEGORIES = [
    "Meat & Prepared Foods", "Produce", "Dairy & Eggs", "Beverages",
    "Frozen & Packaged Foods", "Baby", "Household", "Clothing", "Tax", "Other",
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
            pool_pre_ping=True,   # helps with dropped connections
            future=True,
        )

    return create_engine(url, echo=False, future=True)

engine = get_engine()

try:
    with engine.connect() as conn:
        conn.execute(text("select 1"))
    st.sidebar.success("✅ Connected to Supabase")
except Exception as e:
    st.sidebar.error(f"❌ DB connection failed: {e}")
    st.stop()

@st.cache_resource
def get_sessionmaker():
    return sessionmaker(bind=engine, expire_on_commit=False)

SessionLocal = get_sessionmaker()

#SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

class Receipt(Base):
    __tablename__ = "receipts"
    id:        Mapped[int]            = mapped_column(Integer, primary_key=True)
    store:     Mapped[str]            = mapped_column(String, default="Costco")
    date:      Mapped[date]           = mapped_column(SQLDate, default=date.today)
    subtotal:  Mapped[Optional[float]]= mapped_column(Float)
    tax:       Mapped[Optional[float]]= mapped_column(Float)
    total:     Mapped[Optional[float]]= mapped_column(Float)
    pdf_path:  Mapped[Optional[str]]  = mapped_column(String)
    file_hash: Mapped[Optional[str]]  = mapped_column(String, index=True)   # duplicate detection
    created_at:Mapped[Optional[datetime]] = mapped_column(DateTime, default=datetime.utcnow)
    items: Mapped[list["ReceiptItem"]] = relationship(
        "ReceiptItem", back_populates="receipt", cascade="all, delete-orphan"
    )

class ReceiptItem(Base):
    __tablename__ = "receipt_items"
    id:         Mapped[int]            = mapped_column(Integer, primary_key=True)
    receipt_id: Mapped[int]            = mapped_column(ForeignKey("receipts.id"), index=True)
    date:       Mapped[date]           = mapped_column(SQLDate, nullable=False)
    name:       Mapped[str]            = mapped_column(String, nullable=False)
    category:   Mapped[str]            = mapped_column(String, nullable=False)
    price:      Mapped[Optional[float]]= mapped_column(Float)
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
    """
    Safely add new columns to existing databases that predate the current schema.
    Handles both SQLite and PostgreSQL.
    """
    migrations = [
        ("receipts",  "file_hash",  "VARCHAR"),
        ("receipts",  "created_at", "TIMESTAMP"),
        ("api_logs",  "purpose",    "VARCHAR"),
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
                    pass  # column already exists, safe to ignore
                else:
                    logger.warning("Migration warning (%s.%s): %s", table, column, e)

_run_migrations()

# ===================== COST TRACKING =====================
def log_api_call(model: str, usage, purpose: str = ""):
    """Log token usage and estimated cost to DB."""
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
                session.query(ReceiptItem.name, ReceiptItem.category, Receipt.date)
                .join(Receipt).order_by(Receipt.date.desc()).all()
            )
            for name, category, _ in rows:
                key = str(name).strip().lower()
                if key and key not in mapping:
                    mapping[key] = category
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
    """Resize + compress to reduce OpenAI payload size and cost."""
    from PIL import Image as PILImage
    img = PILImage.open(io.BytesIO(image_bytes))
    img.thumbnail((max_dim, max_dim), PILImage.LANCZOS)
    buf = io.BytesIO()
    fmt = "JPEG" if img.mode == "RGB" else "PNG"
    img.save(buf, format=fmt, quality=quality if fmt == "JPEG" else None, optimize=True)
    return buf.getvalue()

def pdf_to_image_bytes(pdf_bytes: bytes, zoom: float = 2.0) -> tuple[bytes, list[bytes]]:
    """Returns (stitched_bytes_for_vision, [per_page_bytes_for_display])."""
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
    """Scan all PDF text layers for a receipt date."""
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

ITEM:   {{"type":"item","name":"...","category":"...","price":0.0,"taxable":false}}
DISCOUNT:{{"type":"discount","amount":0.0,"raw":"4.00-"}}

Rules:
- Costco Y marker → taxable:true. N marker → taxable:false.
- Discount line like "/2189436 4.00-" → DISCOUNT immediately after that item.
- Keep item price as BASE price; discounts applied in code.
- Categories: {category_list}
{count_note}

Return ONLY this JSON schema:
{{
  "store":"Costco","receipt_date":"2025-11-13",
  "subtotal":0.0,"tax":0.0,"total":0.0,
  "lines":[
    {{"type":"item","name":"","category":"Other","price":0.0,"taxable":false}},
    {{"type":"discount","amount":0.0,"raw":"4.00-"}}
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
    """Crop top+bottom strips of page 1, ask vision model for date only."""
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
def build_items_from_lines(parsed_data: dict) -> list[dict]:
    store   = parsed_data.get("store", "").strip().lower()
    items_out: list[dict] = []
    last_idx = None

    for ln in (parsed_data.get("lines") or []):
        t = str(ln.get("type", "")).strip().lower()
        if t == "item":
            bp = float(ln.get("price") or 0)
            items_out.append({
                "name":       str(ln.get("name", "")).strip(),
                "category":   str(ln.get("category", "Other")).strip() or "Other",
                "base_price": round(bp, 2),
                "discount":   0.0,
                "price":      round(bp, 2),
                "taxable":    bool(ln.get("taxable")),
                "item_tax":   0.0,
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
    return "costco" in s  # handles "Costco Wholesale", "COSTCO #123", etc.

def compute_tax(parsed_data: dict, items: list[dict]) -> tuple[dict, list[dict]]:
    """
    Never ignore tax.
    - If Costco: distribute tax across taxable items (taxable=True).
    - Else: distribute tax across all items with price > 0
            (or across taxable=True items if the model provided taxable flags).
    """
    store_raw   = parsed_data.get("store", "")
    is_costco   = _is_costco_store(store_raw)

    receipt_tax = float(parsed_data.get("tax") or 0)

    # If receipt-level tax is missing/0, don't invent it.
    if receipt_tax <= 0:
        parsed_data["tax"] = 0.0
        for it in items:
            it["item_tax"] = 0.0
        return parsed_data, items

    # Choose which items get tax allocation
    priced_items = [it for it in items if float(it.get("price") or 0) > 0]

    if not priced_items:
        parsed_data["tax"] = 0.0
        for it in items:
            it["item_tax"] = 0.0
        return parsed_data, items

    # If Costco: only taxable items. If none marked taxable, fall back to all priced items.
    if is_costco:
        taxable_items = [it for it in priced_items if bool(it.get("taxable"))]
        alloc_items   = taxable_items if taxable_items else priced_items
    else:
        # For non-Costco: if model provided taxable flags, use them; else tax everything priced.
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

    # Assign per-item tax
    for it in items:
        it["item_tax"] = 0.0

    for it in alloc_items:
        p = float(it.get("price") or 0)
        it["item_tax"] = round(p * tax_rate, 2)

    # Keep receipt tax as the parsed receipt tax (don’t recompute and drift)
    parsed_data["tax"] = round(receipt_tax, 2)
    return parsed_data, items
def add_tax_row(df: pd.DataFrame, tax: float) -> pd.DataFrame:
    if tax > 0:
        row = pd.DataFrame([{"name":"Sales Tax","category":"Tax","base_price":0.0,
                             "discount":0.0,"price":float(tax),"taxable":False,"item_tax":0.0}])
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
            item["category"] = mapping[key]
            hits += 1
    if hits:
        st.info(f"Category memory applied to {hits} item(s).")
    return items

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
    items_df = items_df[items_df["price"] >= 0]

    # Validate totals
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

def delete_all_data():
    session = SessionLocal()
    try:
        session.query(ReceiptItem).delete()
        session.query(Receipt).delete()
        session.commit()
        st.session_state["confirm_delete"] = False
        load_dynamic_categories()
        load_analytics.clear()
        get_monthly_cost.clear()
        st.sidebar.success("All data deleted.")
        st.rerun()
    except Exception as e:
        session.rollback()
        st.sidebar.error(f"Delete error: {e}")
    finally:
        session.close()

def update_receipt_metadata(receipt_id: int, new_store: str, new_date: date, new_total: float):
    session = SessionLocal()
    try:
        r = session.get(Receipt, receipt_id)
        if not r:
            st.sidebar.error(f"Receipt #{receipt_id} not found.")
            return
        r.store = new_store; r.date = new_date; r.total = float(new_total)
        for item in r.items: item.date = new_date
        session.commit()
        st.sidebar.success(f"Receipt #{receipt_id} updated.")
        st.rerun()
    except Exception as e:
        session.rollback()
        st.sidebar.error(f"Update error: {e}")
    finally:
        session.close()

def update_item_category(item_id: int, new_category: str):
    session = SessionLocal()
    try:
        item = session.get(ReceiptItem, item_id)
        if not item:
            st.sidebar.error(f"Item #{item_id} not found.")
            return
        cleaned = (new_category or "").strip()
        if not cleaned:
            st.sidebar.error("Category cannot be empty.")
            return
        item.category = cleaned
        session.commit()
        st.sidebar.success(f"Item #{item_id} → {cleaned}")
        load_dynamic_categories()
        st.rerun()
    except Exception as e:
        session.rollback()
        st.sidebar.error(f"Update error: {e}")
    finally:
        session.close()

# ===================== SIDEBAR =====================
st.sidebar.title("🧾 Receipt Classifier")

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
    receipts_df = pd.read_sql_table("receipts", engine, columns=["id","store","date","total"])
    if not receipts_df.empty:
        receipts_df["date"]  = pd.to_datetime(receipts_df["date"]).dt.date
        receipts_df["label"] = (
            "ID " + receipts_df["id"].astype(str) + " | " +
            receipts_df["store"].astype(str) + " | " +
            receipts_df["date"].astype(str) + " | $" +
            receipts_df["total"].round(2).astype(str)
        )

        st.sidebar.subheader("Edit Receipt")
        sel_label = st.sidebar.selectbox("Select", receipts_df["label"],
                                         index=None, placeholder="Select receipt…", key="sel_receipt")
        if sel_label:
            sel_id    = int(sel_label.split(" | ")[0].replace("ID ", ""))
            edit_df   = receipts_df[receipts_df["id"] == sel_id][["store","date","total"]].copy()
            edited_r  = st.sidebar.data_editor(edit_df, hide_index=True, key=f"edit_r_{sel_id}",
                           column_config={
                               "store": st.column_config.TextColumn("Store", required=True),
                               "date":  st.column_config.DateColumn("Date",  required=True),
                               "total": st.column_config.NumberColumn("Total", format="$%.2f"),
                           })
            if st.sidebar.button("Update Receipt", type="primary", width="stretch"):
                update_receipt_metadata(sel_id, edited_r.iloc[0]["store"],
                                        edited_r.iloc[0]["date"], float(edited_r.iloc[0]["total"]))

    st.sidebar.markdown("---")
    if inspect(engine).has_table("receipt_items"):
        items_sidebar = pd.read_sql_table("receipt_items", engine,
                                          columns=["id","name","category"])
        if not items_sidebar.empty:
            items_sidebar["label"] = (
                "Item " + items_sidebar["id"].astype(str) + " | " +
                items_sidebar["name"].str[:25] + " | " +
                items_sidebar["category"]
            )
            st.sidebar.subheader("Edit Item Category")
            sel_item = st.sidebar.selectbox("Select", items_sidebar["label"],
                                            index=None, placeholder="Select item…", key="sel_item")
            if sel_item:
                sel_item_id  = int(sel_item.split(" | ")[0].replace("Item ", ""))
                curr_cat     = items_sidebar[items_sidebar["id"] == sel_item_id]["category"].iloc[0]
                cat_select   = st.sidebar.selectbox("Category", CATEGORIES,
                                    index=CATEGORIES.index(curr_cat) if curr_cat in CATEGORIES else 0,
                                    key=f"cat_sel_{sel_item_id}")
                cat_custom   = st.sidebar.text_input("Or custom category", key=f"cat_txt_{sel_item_id}")
                final_cat    = cat_custom.strip() or cat_select
                if st.sidebar.button("Update Category", width="stretch"):
                    update_item_category(sel_item_id, final_cat)

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚠️ Danger Zone")
    if "confirm_delete" not in st.session_state:
        st.session_state["confirm_delete"] = False

    if not st.session_state["confirm_delete"]:
        if st.sidebar.button("Delete ALL Receipts", width="stretch"):
            st.session_state["confirm_delete"] = True
            st.rerun()
    else:
        st.sidebar.warning("Are you sure? This cannot be undone.")
        if st.sidebar.button("✅ Yes, delete everything", type="primary", width="stretch"):
            delete_all_data()
        if st.sidebar.button("❌ Cancel", width="stretch"):
            st.session_state["confirm_delete"] = False
            st.rerun()


# ===================== MAIN =====================
st.title("🧾 Receipt Classifier")
st.caption("Upload → Parse → Review → Save | Analytics")

# ---- Review & Edit ----
if (st.session_state.parsed_data is not None
        and st.session_state.df_to_save is not None
        and not st.session_state.df_to_save.empty):

    st.markdown("---")
    st.subheader("Review & Edit")

    df = st.session_state.df_to_save.copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])
    df = df[df["price"] >= 0]
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
            "name":       st.column_config.TextColumn("Item",       required=True),
            "category":   st.column_config.TextColumn("Category",   required=True),
            "base_price": st.column_config.NumberColumn("Base $",   format="$%.2f"),
            "discount":   st.column_config.NumberColumn("Discount", format="$%.2f"),
            "price":      st.column_config.NumberColumn("Net $",    required=True, format="$%.2f"),
            "taxable":    st.column_config.CheckboxColumn("Taxable"),
            "item_tax":   st.column_config.NumberColumn("Item Tax", format="$%.2f"),
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

        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Total Spent",   f"${items['price'].sum():,.2f}")
        a2.metric("Receipts",       len(receipts))
        a3.metric("Line Items",     len(items[items["category"].str.lower() != "tax"]))
        a4.metric("API Cost (mo.)", f"${monthly_cost:.4f}")

        st.markdown("---")

        # ---- Category breakdown ----
        cat = items[items["category"].str.lower() != "tax"].groupby("category")["price"].sum().sort_values(ascending=False)
        col1, col2 = st.columns(2)

        # Horizontal bar — easier to read category names
        fig_cat = px.bar(
            cat.reset_index(),
            x="price", y="category",
            orientation="h",
            title="Spend by Category",
            text="price",
            color="price",
            color_continuous_scale="Blues",
        )
        fig_cat.update_traces(texttemplate="$%{text:,.2f}", textposition="outside")
        fig_cat.update_layout(
            yaxis=dict(categoryorder="total ascending"),
            xaxis_title="Amount ($)",
            yaxis_title="",
            coloraxis_showscale=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=80, t=40, b=10),
            font=dict(size=13),
        )
        fig_cat.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
        col1.plotly_chart(fig_cat, width="stretch")

        # Donut chart
        fig_pie = px.pie(
            cat.reset_index(),
            names="category", values="price",
            title="Category Share",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(
            showlegend=False,
            margin=dict(l=10, r=10, t=40, b=10),
            font=dict(size=13),
        )
        col2.plotly_chart(fig_pie, width="stretch")

        st.markdown("---")

        # ---- Monthly spending ----
        monthly = items.groupby("month_label")["price"].sum().reset_index()
        monthly = monthly.sort_values("month_label")

        fig_monthly = px.bar(
            monthly,
            x="month_label", y="price",
            title="Monthly Spending",
            text="price",
            color="price",
            color_continuous_scale="Blues",
            category_orders={"month_label": sorted(monthly["month_label"].tolist())},
        )
        fig_monthly.update_traces(texttemplate="$%{text:,.2f}", textposition="outside")
        fig_monthly.update_layout(
            xaxis=dict(type="category", title="Month"),
            yaxis_title="Amount ($)",
            coloraxis_showscale=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=40, b=10),
            font=dict(size=13),
            bargap=0.3,
        )
        fig_monthly.update_xaxes(showgrid=False)
        fig_monthly.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
        st.plotly_chart(fig_monthly, width="stretch")

        # API usage table
        with st.expander("🔍 API Usage Log"):
            if inspect(engine).has_table("api_logs"):
                logs_df = pd.read_sql_table("api_logs", engine)
                if not logs_df.empty:
                    logs_df["created_at"] = pd.to_datetime(logs_df["created_at"])
                    logs_df = logs_df.sort_values("created_at", ascending=False).head(50)
                    st.dataframe(
                        logs_df[["created_at","model","purpose","input_tokens","output_tokens","cost_usd"]]
                        .style.format({"cost_usd": "${:.5f}"}),
                        width="stretch",
                    )
    else:
        st.info("No receipts saved yet.")