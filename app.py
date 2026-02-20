import streamlit as st
import pandas as pd

st.set_page_config(page_title="AI Career Intelligence")

st.title("🚀 AI Career Intelligence Platform")

jobs = [
    {"Role": "Senior Data Analyst", "Country": "Netherlands", "Match %": 82, "Hiring Probability %": 17},
    {"Role": "Product Analyst", "Country": "UK", "Match %": 78, "Hiring Probability %": 14},
]

df = pd.DataFrame(jobs)

st.subheader("Top Matches")
st.dataframe(df)

st.subheader("Career Advice")

question = st.text_input("Ask a career question")

if question:
    st.info("Focus on improving skill alignment and targeting lower competition markets.")
