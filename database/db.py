import sqlite3
import os
from datetime import datetime

# Absolute path for Streamlit Cloud
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "users.db")

def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def register_user(email, hashed_pw):
    conn = get_connection()
    c = conn.cursor()

    created_at = datetime.utcnow().isoformat()

    try:
        c.execute("""
            INSERT INTO users (email, password, created_at)
            VALUES (?, ?, ?)
        """, (email, hashed_pw, created_at))

        conn.commit()
        return True

    except sqlite3.IntegrityError:
        # Email already exists
        return False

    finally:
        conn.close()


def login_user(email, hashed_pw):
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
        SELECT * FROM users
        WHERE email = ? AND password = ?
    """, (email, hashed_pw))

    user = c.fetchone()
    conn.close()

    return user
def get_user_history(email):

    conn = sqlite3.connect("career.db")
    c = conn.cursor()

    c.execute("""
        SELECT target_role, country,
               opportunity_score, probability,
               percentile, created_at
        FROM sessions
        WHERE email=?
        ORDER BY created_at ASC
    """, (email,))

    rows = c.fetchall()
    conn.close()

    return rows
