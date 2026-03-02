# 🧾 AI Class Receipt  
### Intelligent Receipt Parsing & Financial Categorization Engine

AI Class Receipt is an end-to-end AI-powered system that extracts, classifies, and structures financial transaction data from receipt images and PDFs into analytics-ready datasets.

This project combines OCR, LLM-based classification, structured data modeling, and interactive dashboards to transform unstructured receipts into actionable financial intelligence.

## Application Overeiw 

![Spending Tracker Dashboard](assets/snapshot.png)

## 🚀 Problem Statement

Financial documents such as receipts and invoices are unstructured and difficult to analyze at scale. Manual categorization is:

- Time-consuming  
- Error-prone  
- Not scalable  

AI Class Receipt automates the ingestion and classification process, turning raw receipts into structured financial data stored in a relational database.


## 🏗 System Architecture

User Upload (PDF/Image)  
↓  
Text Extraction (pdfplumber / Vision Model)  
↓  
LLM Classification (OpenAI API)  
↓  
Data Cleaning & Normalization (Pandas)  
↓  
PostgreSQL Storage (Supabase Pooler)  
↓  
Interactive Dashboard (Streamlit)


## 🛠 Tech Stack

### Frontend
- Streamlit

### Backend
- Python
- Pandas
- SQLAlchemy
- PostgreSQL (Supabase Transaction Pooler)

### AI & NLP
- OpenAI LLM (Text + Vision)
- Regex preprocessing
- Prompt-engineered structured extraction

### Data Visualization
- Plotly
- SQL aggregation queries



## 📂 Core Features

### 1️⃣ Receipt Parsing
- Upload PDF or image receipts
- Extract:
  - Merchant name
  - Transaction date
  - Line items
  - Total amount

### 2️⃣ AI-Based Categorization
Each line item is classified into:
- Category (e.g., Groceries, Electronics, Household)
- Sub-category (e.g., Produce, Utilities, Cleaning Supplies)

### 3️⃣ Data Cleaning
- Standardized date format (YYYY-MM-DD)
- Deduplication logic
- Numeric validation
- Category normalization

### 4️⃣ Database Design

**Tables:**
- `receipts`
- `receipt_items`

Features:
- Foreign key enforcement
- Persistent storage via Supabase
- Secure SSL connection

### 5️⃣ Analytics Dashboard
- Category-based spend breakdown
- Time-series expense tracking
- Interactive filtering
- Real-time aggregated insights


## 📊 Example Use Cases

- Personal finance automation
- Expense tracking systems
- SMB accounting tools
- Transaction-level analytics inputs
- AI data engineering portfolio demonstration

## 🧠 What This Project Demonstrates

- End-to-end AI system design
- LLM prompt engineering for structured outputs
- Relational database schema design
- Production-style data pipeline architecture
- Secure secret management
- Cloud deployment readiness
- 
## 🔐 Security Practices

- Environment variables stored in `.env`
- Secrets configured via Streamlit Cloud
- No hard-coded API keys
- SSL-encrypted database connections

## 📦 Deployment

- Streamlit Cloud (Frontend)
- Supabase PostgreSQL (Backend)

## 🔮 Future Enhancements

- Spending anomaly detection
- Merchant name normalization using embeddings
- Semantic receipt search via vector database
- Multi-user authentication
- Expense forecasting model