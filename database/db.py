import sqlite3

def init_db():
    conn = sqlite3.connect("career.db")
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        email TEXT PRIMARY KEY,
        password TEXT
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        email TEXT,
        target_role TEXT,
        country TEXT,
        opportunity_score REAL,
        percentile REAL
    )
    """)

    conn.commit()
    conn.close()

def authenticate_user(email, password):
    conn = sqlite3.connect("career.db")
    c = conn.cursor()
    c.execute(
        "SELECT * FROM users WHERE email=? AND password=?",
        (email, password)
    )
    result = c.fetchone()
    conn.close()
    return result

def register_user(email, password):
    conn = sqlite3.connect("career.db")
    c = conn.cursor()
    c.execute(
        "INSERT OR IGNORE INTO users VALUES (?,?)",
        (email, password)
    )
    conn.commit()
    conn.close()
