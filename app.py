import os
import json
import pandas as pd
import streamlit as st
import pdfplumber
import plotly.express as px
from openai import OpenAI
from datetime import date
from dotenv import load_dotenv
import re
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, ForeignKey, inspect
from sqlalchemy.orm import DeclarativeBase, sessionmaker, relationship, Mapped, mapped_column
from sqlalchemy import Integer, String, Float, Date as SQLDate

load_dotenv()

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
    "Other"
]

# Initialize a global list for categories. This will be updated from the DB.
CATEGORIES = list(INITIAL_CATEGORIES)

# SQLAlchemy Boilerplate
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
    pdf_path: Mapped[str | None] = mapped_column(String)
    items: Mapped[list["ReceiptItem"]] = relationship("ReceiptItem", back_populates="receipt",
                                                     cascade="all, delete-orphan")

class ReceiptItem(Base):
    __tablename__ = "receipt_items"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    receipt_id: Mapped[int] = mapped_column(ForeignKey("receipts.id"), index=True)
    date: Mapped[date] = mapped_column(SQLDate, nullable=False)  # <--- NEW FIELD ADDED
    name: Mapped[str] = mapped_column(String, nullable=False)
    category: Mapped[str] = mapped_column(String, nullable=False)
    price: Mapped[float | None] = mapped_column(Float)
    receipt: Mapped[Receipt] = relationship("Receipt", back_populates="items")

Base.metadata.create_all(engine)

# ===================== DYNAMIC CATEGORY LOADING & MAPPING =====================

def load_dynamic_categories():
    """Loads all unique categories from the database and updates the global CATEGORIES list."""
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
    """
    Loads a mapping of all unique item names to their MOST RECENT category.
    This provides the user-defined memory.
    """
    session = SessionLocal()
    mapping = {}
    try:
        if inspect(engine).has_table("receipt_items"):
            query = session.query(
                ReceiptItem.name, 
                ReceiptItem.category, 
                Receipt.date
            ).join(Receipt).order_by(Receipt.date.desc())
            
            for name, category, date in query.all():
                normalized_name = name.strip().lower()
                if normalized_name and normalized_name not in mapping:
                    mapping[normalized_name] = category
            
    except Exception as e:
        print(f"Error loading item category mapping: {e}")
    finally:
        session.close()
    return mapping

# Load categories at the start of the script execution (or Streamlit rerun)
load_dynamic_categories()

# Initialize session state variables
if "parsed_data" not in st.session_state:
    st.session_state.parsed_data = None
if "df_to_save" not in st.session_state:
    st.session_state.df_to_save = None

# ===================== CORE FUNCTIONS (Parsing & Saving) =====================

def extract_text(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        return "\n\n".join(page.extract_text() or "" for page in pdf.pages if page.extract_text())

def get_item_count(text: str) -> int | None:
    match = re.search(r"TOTAL NUMBER OF ITEMS SOLD\s*=\s*(\d+)", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

def parse_receipt(text: str, item_count: int | None) -> dict:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    count_constraint = ""
    if item_count is not None and item_count > 0:
        count_constraint = f"""
5.  **COUNT CHECK (CRITICAL):** The total number of items in the final "items" list MUST match the "TOTAL NUMBER OF ITEMS SOLD" count found on the receipt text (which is **{item_count}** in this case).
"""
    
    category_list = ", ".join(CATEGORIES) 

    prompt = f"""
From the receipt text, extract the items purchased, the subtotals, tax, total, and the **purchase date**.

Instructions for Item Extraction:
1.  **Item Definition:** Extract every line that represents a unique purchased product.
2.  **Discount Exclusion (CRITICAL):** **DO NOT** include any lines that represent discounts, instant savings, or negative prices in the 'items' list. Only list the base price of the item.
3.  **Price Rule:** Use the **full, undiscounted price** for the item.
4.  **Categorization:** Assign one exact category to each item, preferably from this comprehensive list of existing categories: {category_list}. **Prioritize categories in this list.** If an item is truly unique, use a category of your own.
{count_constraint}

Return ONLY valid JSON, including the receipt_date in YYYY-MM-DD format:
{{
  "items": [],
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
        messages=[{"role": "user", "content": prompt}]
    )
    return json.loads(resp.choices[0].message.content.strip())

def apply_category_memory(parsed_data: dict) -> list[dict]:
    """
    Applies the user-defined category mapping to the AI-parsed items.
    """
    items = parsed_data.get("items", [])
    if not items:
        return []

    category_map = get_item_category_mapping()
    
    if not category_map:
        return items

    st.info("Applying user-defined category memory...")

    for item in items:
        item_name = str(item.get("name", "")).strip().lower()
        if item_name in category_map:
            new_category = category_map[item_name]
            item["category"] = new_category
            st.write(f"Memory Hit: **{item['name']}** categorized as **{new_category}**.")
    
    return items


def save_to_database(items_df: pd.DataFrame, receipt_date: date):
    st.subheader("Saving to database")
    if st.session_state.parsed_data is None:
        st.error("Parse a receipt first!")
        return

    items_df['price'] = pd.to_numeric(items_df['price'], errors='coerce')
    items_df = items_df.dropna(subset=['price'])

    new_subtotal = items_df['price'].sum()
    tax_value = st.session_state.parsed_data.get("tax", 0.0)
    new_total = new_subtotal + tax_value

    session = SessionLocal()
    try:
        receipt = Receipt(
            store="Costco", date=receipt_date, subtotal=new_subtotal,
            tax=tax_value, total=new_total, pdf_path="uploaded.pdf"
        )
        session.add(receipt)
        session.flush()

        for _, item in items_df.iterrows():
            session.add(ReceiptItem(
                receipt_id=receipt.id,
                date=receipt_date,  # <--- PASSING DATE TO ITEM
                name=str(item.get("name", "")).strip(),
                category=item.get("category", "Other"),
                price=item.get("price")
            ))

        session.commit()
        st.success(f"Receipt #{receipt.id} saved! Final Total: ${new_total:,.2f} on {receipt_date.isoformat()}")
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
        st.sidebar.success("✅ All data deleted successfully! Please refresh and run the app again.")
        load_dynamic_categories()
        st.rerun()
    except Exception as e:
        session.rollback()
        st.sidebar.error(f"Error during data cleanup: {e}")
    finally:
        session.close()

# ===================== DATABASE EDIT FUNCTIONS =====================

def update_receipt_metadata(receipt_id: int, new_store: str, new_date: date, new_total: float):
    """Updates the store, date, and total for a specific receipt ID."""
    session: Session = SessionLocal()
    try:
        receipt_to_update = session.get(Receipt, receipt_id)

        if receipt_to_update:
            receipt_to_update.store = new_store
            receipt_to_update.date = new_date
            receipt_to_update.total = new_total
            
            # Update dates on all associated receipt items as well
            for item in receipt_to_update.items:
                 item.date = new_date

            session.commit()
            st.sidebar.success(f"Successfully updated Receipt ID **{receipt_id}**! Store: **{new_store}**, Date: **{new_date.isoformat()}**, Total: **${new_total:,.2f}**")
            st.rerun()
        else:
            st.sidebar.error(f"Could not find Receipt with ID: {receipt_id}")

    except Exception as e:
        session.rollback()
        st.sidebar.error(f"An error occurred during metadata update: {e}")
    finally:
        session.close()

def update_item_category(item_id: int, new_category: str):
    """Updates the category for a specific item ID and reloads dynamic categories."""
    session: Session = SessionLocal()
    try:
        item_to_update = session.get(ReceiptItem, item_id)
        
        # Strip whitespace and check if the category is valid/not empty
        cleaned_category = new_category.strip()
        if not cleaned_category:
             st.sidebar.error("New category cannot be empty.")
             return # Stop update if empty

        if item_to_update:
            item_to_update.category = cleaned_category
            session.commit()
            st.sidebar.success(f"Item ID **{item_id}** (**{item_to_update.name}**) recategorized to **{cleaned_category}**.")
            load_dynamic_categories()
            st.rerun()
        else:
            st.sidebar.error(f"Could not find Receipt Item with ID: {item_id}")

    except Exception as e:
        session.rollback()
        st.sidebar.error(f"An error occurred during category update: {e}")
    finally:
        session.close()


# ===================== UI: SIDEBAR & UPLOAD =====================
st.set_page_config(page_title="Receipt Classifier", layout="wide")

if not os.getenv("OPENAI_API_KEY"):
    st.error("Put your key in .env file!")
    st.stop()

# --- SIDEBAR CONTENT (Upload and Data Management) ---
st.sidebar.header("🧾 Upload Receipt")
uploaded_file = st.sidebar.file_uploader("Upload your receipt (PDF)", type="pdf")
st.sidebar.markdown("---")

st.sidebar.header("⚙️ Data Management & Editing")

if os.path.exists(DB_PATH) and inspect(engine).has_table("receipts"):
    
    # Load all receipts for Metadata Editing
    receipts_df = pd.read_sql_table("receipts", engine, columns=['id', 'store', 'date', 'total'])
    if not receipts_df.empty:
        receipts_df['date'] = pd.to_datetime(receipts_df['date']).dt.date
        receipts_df['label'] = 'ID ' + receipts_df['id'].astype(str) + ' | ' + receipts_df['store'] + ' | ' + receipts_df['date'].astype(str) + ' | $' + receipts_df['total'].round(2).astype(str)
    
        # 1. Edit Saved Receipt Metadata
        st.sidebar.subheader("1. Edit Receipt Header (Store, Date, Total)")
        
        selected_label = st.sidebar.selectbox(
            "Select Receipt to Edit",
            receipts_df['label'],
            index=None,
            placeholder="Select a receipt...",
            key="select_receipt_date"
        )

        if selected_label:
            selected_id = int(selected_label.split(' | ')[0].replace('ID ', ''))
            
            # Load the single receipt into a DataFrame for editing
            receipt_to_edit = receipts_df[receipts_df['id'] == selected_id][['store', 'date', 'total']].copy()
            
            st.sidebar.caption(f"Editing Receipt ID: **{selected_id}**")

            # Use data_editor in the sidebar
            edited_receipt_df = st.sidebar.data_editor(
                receipt_to_edit,
                hide_index=True,
                column_config={
                    "store": st.column_config.TextColumn("Store Name", required=True),
                    "date": st.column_config.DateColumn("Purchase Date", required=True), 
                    "total": st.column_config.NumberColumn("Total Price", required=True, format="$%.2f")
                },
                key=f"edit_receipt_{selected_id}"
            )

            if st.sidebar.button("Update Receipt Metadata", key="update_metadata_button", type="primary", use_container_width=True):
                
                # Extract values from the edited DataFrame
                new_store = edited_receipt_df.iloc[0]['store']
                new_date = edited_receipt_df.iloc[0]['date']
                new_total = edited_receipt_df.iloc[0]['total']
                
                # Call the new update function
                update_receipt_metadata(
                    receipt_id=selected_id, 
                    new_store=new_store, 
                    new_date=new_date, 
                    new_total=new_total
                )
        
        st.sidebar.markdown("---") # Separator between the two management tools

    # Load all items for Recategorization
    items_df = pd.read_sql_table("receipt_items", engine, columns=['id', 'receipt_id', 'name', 'category', 'price', 'date'])
    if not items_df.empty:
        # Merge with receipts to get date/total context
        receipt_cols = ['id', 'store', 'date', 'total']
        # This join is no longer needed to get the date, but kept for context in the label
        items_df = pd.merge(items_df, receipts_df[receipt_cols], left_on='receipt_id', right_on='id', suffixes=('_item', '_receipt'), how='left')
        
        # Create descriptive labels for the select box
        items_df['label'] = (
            'Item ID ' + items_df['id_item'].astype(str) +
            ' | ' + items_df['name'].str[:30] + '... ' +
            ' | Current Cat: ' + items_df['category']
        )
        
        # 2. Recategorize Saved Items
        st.sidebar.subheader("2. Edit Individual Item Categories")
        
        selected_item_label = st.sidebar.selectbox(
            "Select Item to Recategorize",
            items_df['label'],
            index=None,
            placeholder="Select an item...",
            key="select_item_cat"
        )

        if selected_item_label:
            selected_item_id = int(selected_item_label.split(' | ')[0].replace('Item ID ', ''))
            current_category = items_df[items_df['id_item'] == selected_item_id]['category'].iloc[0]
            
            # Dropdown for existing categories
            new_category_from_select = st.sidebar.selectbox(
                "Select Existing Category",
                CATEGORIES,
                index=CATEGORIES.index(current_category) if current_category in CATEGORIES else (CATEGORIES.index("Other") if "Other" in CATEGORIES else 0),
                key=f"edit_cat_select_{selected_item_id}"
            )
            
            st.sidebar.caption("OR create a new one:")

            # Text input for new/custom category
            new_category_from_text = st.sidebar.text_input(
                "Custom Category Name", 
                value="",
                key=f"edit_cat_text_{selected_item_id}"
            )

            # Determine which category to use for the update
            final_category = new_category_from_text.strip() or new_category_from_select

            if st.sidebar.button("Update Item Category", key="update_category_button", type="secondary", use_container_width=True):
                if not final_category.strip():
                     st.sidebar.error("Please select an existing category OR type a custom category name.")
                else:
                    update_item_category(selected_item_id, final_category)


    # 3. Delete All Data
    st.sidebar.markdown("---")
    st.sidebar.subheader("3. ⚠️ Clear Database")
    st.sidebar.caption("Use this button to clear all data.")
    if st.sidebar.button("⚠️ DELETE ALL EXISTING RECEIPTS (Start Fresh)", use_container_width=True):
        delete_all_data()

else:
    st.sidebar.info("No receipts saved yet. Upload a PDF to begin tracking.")


# --- MAIN CONTENT HEADER ---
st.title("Receipt Classifier")
st.caption("Upload → Parse → Edit → Save | Analytics | Data Editing")


# ===================== UI: PARSING AND EDITING SECTION =====================

if uploaded_file:
    # --- Parsing Logic ---
    if st.button("Parse & Categorize", type="primary"):
        st.session_state.df_to_save = None
        st.session_state.parsed_data = None
        with st.spinner("Reading PDF..."):
            text = extract_text(uploaded_file)
            item_count = get_item_count(text)
            
            if item_count is not None and item_count > 1:
                st.info(f"Detected **{item_count}** items sold from the receipt text. Instructing AI to match this count.")
            else:
                st.warning("Could not automatically detect an item count on this receipt. Relying on AI's ability to exclude discounts.")

        with st.spinner("Asking GPT-4o-mini..."):
            parsed_data = parse_receipt(text, item_count)
            
        parsed_data["items"] = apply_category_memory(parsed_data)
        st.session_state.parsed_data = parsed_data
        
        st.session_state.df_to_save = pd.DataFrame(st.session_state.parsed_data.get("items", []))


    if (st.session_state.parsed_data is not None
        and st.session_state.df_to_save is not None
        and not st.session_state.df_to_save.empty):

        # --- Data Cleaning and Editing UI ---
        st.markdown("---")
        st.subheader("Confirm & Edit Parsed Data")
        
        temp_df = st.session_state.df_to_save.copy()
        temp_df['price'] = pd.to_numeric(temp_df['price'], errors='coerce')
        st.session_state.df_to_save = temp_df[temp_df['price'] >= 0].dropna(subset=['price'])
        
        parsed_date_str = st.session_state.parsed_data.get("receipt_date")
        try:
            parsed_date = pd.to_datetime(parsed_date_str).date()
        except (TypeError, ValueError):
            st.warning("Could not extract date from receipt. Using today's date.")
            parsed_date = date.today()
            
        receipt_date = st.date_input("Confirm/Edit Receipt Date", parsed_date, key="receipt_date_input")

        st.subheader("Edit Items")
        edited_df = st.data_editor(st.session_state.df_to_save, 
                                   num_rows="dynamic", 
                                   column_config={"category": st.column_config.TextColumn("Category", required=True)}, 
                                   key="receipt_editor")
        st.session_state.df_to_save = edited_df

        edited_df['price'] = pd.to_numeric(edited_df['price'], errors='coerce')
        current_subtotal = edited_df['price'].sum()
        current_tax = st.session_state.parsed_data.get("tax", 0.0)
        current_total = current_subtotal + current_tax

        col1, col2, col3 = st.columns(3)
        col1.metric("Subtotal (Recalc)", f"${current_subtotal:,.2f}")
        col2.metric("Tax (Parsed)", f"${current_tax:,.2f}")
        col3.metric("Total (Recalc)", f"${current_total:,.2f}")

        st.download_button("Download CSV", edited_df.to_csv(index=False).encode(), f"costco_{date.today()}.csv")

        st.markdown("---")
        if st.button("SAVE TO DATABASE (Permanent)", type="primary", use_container_width=True):
            save_to_database(st.session_state.df_to_save, receipt_date)

else:
    st.info("No file uploaded yet. Use the sidebar to upload a receipt PDF and begin tracking!")


# ===================== ANALYTICS =====================
st.markdown("---")
st.subheader("Spending Analytics")
if os.path.exists(DB_PATH) and inspect(engine).has_table("receipts"):
    
    # We can now query the date directly from receipt_items
    items = pd.read_sql_table("receipt_items", engine)
    items["price"] = pd.to_numeric(items["price"], errors="coerce")
    receipts = pd.read_sql_table("receipts", engine)
    # Merging is no longer strictly necessary for date/price, but kept for a cleaner structure
    merged_df = items.copy()
    merged_df['date'] = pd.to_datetime(merged_df['date'])
    merged_df['month_label'] = merged_df['date'].dt.strftime('%Y-%m')

    total = merged_df["price"].sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Spent", f"${total:,.2f}")
    c2.metric("Receipts", len(receipts))
    c3.metric("Items", len(items))
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    cat = merged_df.groupby("category")["price"].sum().sort_values(ascending=False)
    col1.subheader("Category Totals")
    col1.dataframe(cat.reset_index().style.format({"price": "${:,.2f}"}), use_container_width=True)
    col2.plotly_chart(px.pie(cat, names=cat.index, values=cat.values, title="Category Breakdown"), use_container_width=True)

    st.markdown("---")
    st.subheader("Spending Trends")
    monthly_spending = merged_df.groupby("month_label")["price"].sum().reset_index()
    st.subheader("Total Monthly Spending")
    st.dataframe(monthly_spending.rename(columns={'month_label': 'Month', 'price': 'Total Spent'}).style.format({"Total Spent": "${:,.2f}"}), use_container_width=True)
    fig = px.bar(monthly_spending, x='month_label', y='price', title='Total Spending by Month', labels={'month_label': 'Month (YYYY-MM)', 'price': 'Total Spent'}, text='price')
    fig.update_traces(texttemplate='$%{text:,.2f}', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Monthly Breakdown by Category")
    monthly_category_spending = merged_df.groupby(["month_label", "category"])["price"].sum().reset_index()
    fig_stacked = px.bar(monthly_category_spending, x='month_label', y='price', color='category', title='Monthly Spending Stacked by Category', labels={'month_label': 'Month (YYYY-MM)', 'price': 'Total Spent', 'category': 'Category'}, hover_data=['category'])
    fig_stacked.update_layout(xaxis={'categoryorder':'category ascending'})
    st.plotly_chart(fig_stacked, use_container_width=True)

st.caption(f"Database: {os.path.abspath(DB_PATH)}")