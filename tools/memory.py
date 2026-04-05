import sqlite3

conn = sqlite3.connect("memory.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT,
    report TEXT
)
""")

conn.commit()

def save_to_memory(query: str, report: str):
    cursor.execute(
        "INSERT INTO history (query, report) VALUES (?, ?)",
        (query, report)
    )
    conn.commit()