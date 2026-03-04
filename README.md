# 🧾 Receipt Classifier

An AI-powered receipt parsing and spending analytics app built with Streamlit, OpenAI, and PostgreSQL. Upload a receipt PDF or image, let the AI extract and categorize every line item, review and edit the results, then save to a database and explore your spending through interactive dashboards.

---

## 🌐 Live Demo

🚀 **Try the App Here:**  
👉 [Launch AI Class Receipt](https://ledgerlens-ai.streamlit.app/)
> Upload a receipt (PDF or image) and see AI automatically extract, classify, and store structured transaction data in real time.

## Application Overview 

![Spending Tracker Dashboard](assets/snapshot.png)

### 📤 Receipt Parsing
- Upload receipts as **PDF or image** (PNG, JPG, JPEG) up to 10 MB
- **Smart routing**: text-layer PDFs go to `gpt-4o-mini` (fast, cheap); image-only or garbled PDFs are rendered and sent to `gpt-4.1-mini` vision
- Extracts store name, date, all line items, discounts, subtotal, tax, and total
- Handles **Costco**, **Target**, **Walmart**, and generic receipt formats
- Correctly parses **refund receipts** with negative totals and line items
- Applies **Costco member discounts** inline (discount lines deducted from the preceding item's price)
- Distributes tax proportionally across taxable items per store rules (Y/N markers for Costco, T/N/N+ for Target/Walmart)

### 🧠 Category Memory
- Every saved receipt trains a per-item category lookup
- On future uploads, previously seen item names are auto-assigned their historically most-used category and sub-category
- User corrections accumulate weight — correct an item 3 times and that correction beats any one-off AI assignment

### 📊 Spending Analytics
Five analytics tabs powered by Plotly and styled HTML tables:

| Tab | What it shows |
|---|---|
| **Overview** | Spend by category (bar + pie), monthly spending trend |
| **Categories** | Drill into any category — top items chart + full item list |
| **Monthly** | Month selector with category breakdown and styled item table |
| **By Store** | Total spend per store, per-store category breakdown and monthly trend |
| **Drill Down** | Search + filter any item across all receipts |
| **API Log** | Every OpenAI call with model, purpose, token counts, and cost |

All item tables use **color-coded category badges**, green/red price coloring, and a total footer row.

### 🗃️ Data Management *(full version)*
- Edit receipt metadata (store, date, total) from the sidebar
- Recategorize individual items inline with a dropdown
- Delete individual receipts or all data
- Changes immediately reflected in all analytics

### 🔒 Demo Mode
Set `DEMO_MODE = True` at the top of `app_demo.py` to lock all write operations for public/recruiter demos:
- Parsing and analytics remain **fully live**
- Save, edit, delete, and recategorize buttons are **visible but disabled**
- A banner explains the restriction to viewers

### 💾 Storage
- **Database**: PostgreSQL (Supabase) in production, SQLite fallback for local development
- **File storage**: Optional AWS S3 upload of original receipt files with pre-signed 1-hour download links
- **Duplicate detection**: SHA-256 hash checked before save to warn on re-uploads

### 💰 Cost Monitoring
- Tracks every OpenAI API call (model, tokens, cost) in the database
- Monthly cost displayed live in the sidebar with a green/yellow/red indicator
- Configurable monthly limit — warns at 80%, errors at 100%

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| AI / Vision | OpenAI `gpt-4.1-mini` (vision), `gpt-4o-mini` (text) |
| Database ORM | SQLAlchemy 2.0 (mapped columns, relationships) |
| Database | PostgreSQL via psycopg2 / SQLite fallback |
| File Storage | AWS S3 via boto3 |
| PDF Parsing | pdfplumber (text layer), PyMuPDF / fitz (image render) |
| Charts | Plotly Express |
| Retry Logic | tenacity (exponential backoff on rate limits) |
| Environment | python-dotenv + Streamlit Secrets |

---

## Setup

### 1. Install dependencies

```bash
pip install streamlit openai sqlalchemy psycopg2-binary pdfplumber pymupdf \
            plotly pandas boto3 tenacity python-dotenv pillow
```

### 2. Configure secrets

For local development, create a `.env` file:

```env
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql://user:password@host:5432/dbname
AWS_ACCESS_KEY_ID=...           # optional
AWS_SECRET_ACCESS_KEY=...       # optional
AWS_DEFAULT_REGION=us-east-1    # optional
S3_BUCKET=your-bucket-name      # optional
APP_PASSWORD=yourpassword       # optional — gates the whole app
```

For Streamlit Cloud, add the same keys under **Settings → Secrets**.

### 3. Run

```bash
streamlit run app_demo.py
```

The app auto-creates all database tables on first run and applies schema migrations safely on subsequent runs.

---

## Demo vs Production

| | Demo (`DEMO_MODE = True`) | Production (`DEMO_MODE = False`) |
|---|---|---|
| Parse receipts | ✅ | ✅ |
| View analytics | ✅ | ✅ |
| Save receipts | ❌ disabled | ✅ |
| Edit / delete data | ❌ disabled | ✅ |
| Recategorize items | ❌ disabled | ✅ |

To switch to production mode, set `DEMO_MODE = False` at the top of `app_demo.py`.

---

## Project Structure

```
app_demo.py          # Main application (single-file Streamlit app)
receipts.db          # SQLite database (local dev only, auto-created)
.env                 # Local secrets (never commit)
README.md            # This file
```

### Database Schema

```
receipts
  id, store, date, subtotal, tax, total, pdf_path, file_hash, created_at

receipt_items
  id, receipt_id (FK), date, name, category, sub_category, price

api_logs
  id, created_at, model, input_tokens, output_tokens, cost_usd, purpose
```

---

## Supported Receipt Formats

| Store | Format | Notes |
|---|---|---|
| Costco | In-store PDF / image | Discount lines, Y/N tax markers |
| Target | In-store image | Department headers ignored, T/N/N+ tax markers |
| Walmart | In-store image | Same format as Target |
| Generic | PDF or image | Any receipt with prices and a date |
| Refund | Any | Negative totals and item prices handled automatically |

---

## Cost Estimates

Typical cost per receipt with the default models:

| Receipt type | Model used | Approx. cost |
|---|---|---|
| Text PDF (Costco) | `gpt-4o-mini` | ~$0.0003 |
| Image receipt | `gpt-4.1-mini` | ~$0.002–0.005 |

The default monthly cap is **$20**. Adjust `MONTHLY_COST_LIMIT_USD` in the config section.
