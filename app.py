import streamlit as st
import sqlite3
ffrom database.db import init_db, authenticate_user, register_user

st.set_page_config(page_title="Career Intelligence", layout="wide")

init_db()

st.markdown("""
# 🚀 Career Intelligence Platform

### Your Global Career Positioning Engine

We combine:

• Resume intelligence  
• Market demand modeling  
• Supply vs demand analysis  
• Salary forecasting  
• Visa feasibility  
• Interview probability  

To show you exactly where you stand in the global market.

---

""")

col1, col2 = st.columns([2,1])

with col2:
    st.subheader("Login")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if authenticate_user(email, password):
            st.session_state["user"] = email
            st.success("Logged in successfully.")
            st.switch_page("pages/1_Questionnaire.py")
        else:
            st.error("Invalid credentials")

    st.markdown("### New User?")
    if st.button("Register"):
        register_user(email, password)
        st.success("Registered successfully.")
