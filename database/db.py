import sqlite3
import hashlib
from datetime import datetime


# -------------------------------------------------
# INIT DATABASE
# -------------------------------------------------

def init_db():
    conn = sqlite3.connect("career.db")
    c = conn.cursor()

    # Users table
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        email TEXT PRIMARY KEY,
        password TEXT,
        created_at TEXT
    )
    """)

    # Analysis history table
    c.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT,
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
# PASSWORD HASHING (IMPORTANT)
# -------------------------------------------------

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# -------------------------------------------------
# REGISTER USER
# -------------------------------------------------

def register_user(email, password):
    conn = sqlite3.connect("career.db")
    c = conn.cursor()

    hashed_pw = hash_password(password)

    c.execute("""
        INSERT OR IGNORE INTO users (email, password, created_at)
        VALUES (?, ?, ?)
    """, (email, hashed_pw, datetime.utcnow()))

    conn.commit()
    conn.close()


# -------------------------------------------------
# AUTHENTICATE USER
# -------------------------------------------------

def authenticate_user(email, password):
    conn = sqlite3.connect("career.db")
    c = conn.cursor()

    hashed_pw = hash_password(password)

    c.execute("""
        SELECT * FROM users
        WHERE email=? AND password=?
    """, (email, hashed_pw))

    result = c.fetchone()

    conn.close()
    return result


# -------------------------------------------------
# SAVE ANALYSIS SESSION
# -------------------------------------------------

def save_session(email, role, country, opp, prob, percentile):

    conn = sqlite3.connect("career.db")
    c = conn.cursor()

    c.execute("""
        INSERT INTO sessions
        (email, target_role, country,
         opportunity_score, probability,
         percentile, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        email,
        role,
        country,
        opp,
        prob,
        percentile,
        datetime.utcnow()
    ))

    conn.commit()
    conn.close()


# -------------------------------------------------
# FETCH USER HISTORY
# -------------------------------------------------

def get_user_history(email):

    conn = sqlite3.connect("career.db")
    c = conn.cursor()

    c.execute("""
        SELECT target_role, country,
               opportunity_score, probability,
               percentile, created_at
        FROM sessions
        WHERE email=?
        ORDER BY created_at DESC
    """, (email,))

    rows = c.fetchall()
    conn.close()

    return rows
