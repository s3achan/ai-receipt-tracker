# 🧾 AI Receipt Classifier & Expense Tracker

An end-to-end AI-powered receipt processing and expense tracking system
built with Streamlit, PostgreSQL, SQLAlchemy, OpenAI Vision models, and
AWS S3.

------------------------------------------------------------------------

## 🚀 Overview

This application transforms unstructured receipt images and PDFs into
structured, analytics-ready financial data using a normalized relational
database design.

### Key Capabilities:

-   Upload receipt images or PDFs
-   AI-powered item extraction
-   Hierarchical expense categorization
-   PostgreSQL relational storage
-   Foreign key--enforced data integrity
-   AWS S3 cloud storage for receipt images
-   Manual recurring expense insertion (Mortgage, Auto, etc.)
-   Dashboard-ready analytics

------------------------------------------------------------------------

## 🏗 System Architecture

User Upload\
→ AI Receipt Parsing (Vision + GPT)\
→ Structured Extraction\
→ PostgreSQL Database (Normalized Schema)\
→ Categorization & Aggregation\
→ Analytics Dashboard\
→ AWS S3 Storage

------------------------------------------------------------------------

## 🗂 Database Schema

### 🧾 receipts (Parent Table)

  Column       Description
  ------------ ----------------------
  id           Primary key
  date         Receipt date
  name         Receipt description
  created_at   Timestamp (optional)

### 🧾 receipt_items (Child Table)

  Column         Description
  -------------- ---------------------------
  id             Primary key
  receipt_id     Foreign key → receipts.id
  date           Item date
  name           Item description
  category       Main category
  sub_category   Nested category
  price          Item price

------------------------------------------------------------------------

## 🔒 Foreign Key Integrity

receipt_items.receipt_id → receipts.id

Prevents orphan expense records and ensures relational consistency.

------------------------------------------------------------------------

## 🧠 Hierarchical Expense Categorization

Example:

  Category    Sub Category
  ----------- ----------------
  Groceries   Meat & Seafood
  Groceries   Produce
  Groceries   Frozen Foods
  Housing     Mortgage
  Auto        Tesla Payment

Example SQL transformation:

UPDATE receipt_items SET sub_category = category, category = 'Groceries'
WHERE category IN ( 'Meat & Seafood', 'Snacks & Candy', 'Frozen Foods',
'Pantry & Dry Goods', 'Produce' );

------------------------------------------------------------------------

## ☁ AWS S3 Integration

-   Secure cloud image storage
-   IAM-based access control
-   Production-ready architecture

------------------------------------------------------------------------

## 💰 Manual Expense Example

INSERT INTO receipts (date, name) VALUES ('2026-03-01', 'Mortgage +
Tesla');

INSERT INTO receipt_items (receipt_id, date, name, category, price,
sub_category) VALUES (1, '2026-03-01', 'Mortgage', 'Housing', 3751.61,
NULL), (1, '2026-03-17', 'Tesla', 'Auto', 439.79, NULL);

------------------------------------------------------------------------

## 🛠 Tech Stack

Frontend: Streamlit\
Backend: Python + SQLAlchemy\
Database: PostgreSQL\
AI: OpenAI Vision + GPT\
Cloud: AWS S3\
Visualization: Plotly

------------------------------------------------------------------------

## 🎯 What This Demonstrates

-   Real-world relational modeling
-   Foreign key constraint handling
-   AI-powered document parsing
-   Financial data normalization
-   Cloud-integrated architecture
-   Analytics-ready data pipelines

------------------------------------------------------------------------

Built as a production-style AI + data engineering system.
