import streamlit as st
from database.db import init_db

# ------------------ INIT ------------------
init_db()
st.set_page_config(layout="wide")

# ------------------ THEME ------------------
st.markdown("""
<style>
html, body, [class*="css"]  {
    font-family: 'Roboto', sans-serif;
}

.main {
    background: linear-gradient(135deg, #6A11CB 0%, #2575FC 100%);
}

.block-container {
    padding-top: 2rem;
}

.card {
    background: white;
    padding: 1.5rem;
    border-radius: 16px;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.1);
}

.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0px 5px 20px rgba(0,0,0,0.08);
}

.gold {
    color: #F7B500;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ------------------ LANDING ------------------
st.title("🚀 Career Intelligence Platform")

st.markdown("""
<div class="card">
<h3>Understand Your Market Position Like 6-Figure Leaders</h3>
<p>
We analyze your resume against global job demand, supply competitiveness,
salary benchmarks, and skill gaps to show your real-world standing.
</p>
</div>
""", unsafe_allow_html=True)

st.info("Use the sidebar to start with Questionnaire → then move to Analysis.")
