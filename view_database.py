"""
view_database.py — Quick script to view database contents
"""

import sqlite3
import pandas as pd

db_path = "data/churn_predictions.db"

# Connect to database
conn = sqlite3.connect(db_path)

# Get all table names
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

print("=== Tables in Database ===")
for table in tables:
    print(f"- {table[0]}")

print("\n" + "="*80 + "\n")

# View each table
for table in tables:
    table_name = table[0]
    print(f"=== Table: {table_name} ===")
    
    # Get row count
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    print(f"Total rows: {count}")
    
    if count > 0:
        # Show first few rows
        df = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 5", conn)
        print(df.to_string())
    else:
        print("(Empty table)")
    
    print("\n" + "="*80 + "\n")

conn.close()
