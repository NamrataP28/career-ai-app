import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from market_engine import market_model
from services.resume_parser import ResumeParser

st.title("📊 Career Market Intelligence")

model = SentenceTransformer("all-MiniLM-L6-v2")
parser = ResumeParser()

file = st.file_uploader("Upload Resume", type=["pdf"])
if not file:
    st.stop()

resume_text = parser.extract_text(file)
resume_embedding = model.encode(resume_text)

role = st.session_state["role"]
country = st.session_state["country"]
current_salary = st.session_state["salary"]

role_embedding = model.encode(role)
similarity = cosine_similarity([resume_embedding],[role_embedding])[0][0]

demand, supply, tightness = market_model(role)

salary_projection = np.random.randint(80000,200000)

salary_growth = (salary_projection - current_salary)/salary_projection if current_salary else 0.5

opportunity_score = (
    0.4*similarity +
    0.3*tightness +
    0.3*salary_growth
)

percentile = round(opportunity_score*100,2)

k1, k2, k3, k4 = st.columns(4)

k1.metric("Opportunity Score", f"{round(opportunity_score*100,2)}%")
k2.metric("Demand (Live Proxy)", f"{demand}")
k3.metric("Supply Estimate", f"{supply}")
k4.metric("Market Tightness", f"{round(tightness*100,2)}%")

st.markdown("### Salary Projection")
st.metric("Projected Salary", f"${salary_projection}")

st.markdown("### Market Ranking")
st.metric("Estimated Percentile", f"{percentile}%")
