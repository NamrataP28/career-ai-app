import streamlit as st
import pandas as pd
from database.db import get_user_history
from database.models import AnalysisHistory

if "username" not in st.session_state:
    st.warning("Login first.")
    st.stop()

db = SessionLocal()

data = db.query(AnalysisHistory).filter(
    AnalysisHistory.username == st.session_state["username"]
).all()

if not data:
    st.info("No history yet.")
    st.stop()

df = pd.DataFrame([{
    "Role": d.role,
    "Country": d.country,
    "Opportunity": d.opportunity,
    "Probability": d.probability,
    "Percentile": d.percentile
} for d in data])

st.subheader("📊 Your Career Progress")

st.line_chart(df[["Opportunity", "Probability"]])
st.dataframe(df)
