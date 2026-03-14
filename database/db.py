import sqlite3
import os
from datetime import datetime

# -------------------------------------------------
# DATABASE PATH (Streamlit Cloud Safe)
# -------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "career_platform.db")


# -------------------------------------------------
# CONNECTION
# -------------------------------------------------

def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


# -------------------------------------------------
# INIT DATABASE
# -------------------------------------------------

def init_db():

    conn = get_connection()
    c = conn.cursor()

    # USERS TABLE
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    # ANALYSIS HISTORY TABLE
    c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            target_role TEXT,
            country TEXT,
            opportunity_score REAL,
            probability REAL,
            percentile REAL,
            created_at TEXT
        )
    """)

    conn.commit()
    conn.close()


# -------------------------------------------------
# REGISTER USER
# -------------------------------------------------

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
        return False

    finally:
        conn.close()


# -------------------------------------------------
# LOGIN USER
# -------------------------------------------------

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


# -------------------------------------------------
# SAVE ANALYSIS SESSION
# -------------------------------------------------

def save_session(email, role, country, opp, prob, percentile):

    conn = get_connection()
    c = conn.cursor()

    created_at = datetime.utcnow().isoformat()

    c.execute("""
        INSERT INTO sessions (
            email,
            target_role,
            country,
            opportunity_score,
            probability,
            percentile,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        email,
        role,
        country,
        opp,
        prob,
        percentile,
        created_at
    ))

    conn.commit()
    conn.close()


# -------------------------------------------------
# GET USER HISTORY
# -------------------------------------------------

def get_user_history(email):

    conn = get_connection()
    c = conn.cursor()

    c.execute("""
        SELECT
            target_role,
            country,
            opportunity_score,
            probability,
            percentile,
            created_at
        FROM sessions
        WHERE email = ?
        ORDER BY created_at ASC
    """, (email,))

    rows = c.fetchall()

    conn.close()

    return rows
