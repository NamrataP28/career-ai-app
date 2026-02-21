import streamlit as st
import sqlite3

st.title("📈 Your Progress History")

conn = sqlite3.connect("career.db")
df = conn.execute("SELECT * FROM sessions").fetchall()
conn.close()

st.write(df)
