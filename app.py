import streamlit as st
import plotly.express as px
from sentence_transformers import SentenceTransformer
from services.resume_parser import ResumeParser
from services.demand_service import DemandService
from services.salary_service import SalaryService
from services.visa_model import VisaModel
from services.ranking_engine import RankingEngine
from services.gpt_service import GPTService

model = SentenceTransformer("all-MiniLM-L6-v2")

parser = ResumeParser()
demand_service = DemandService()
salary_service = SalaryService()
visa_model = VisaModel()
ranking_engine = RankingEngine()
gpt_service = GPTService()

st.title("AI Career Intelligence Platform")

file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if not file:
    st.stop()

resume_text = parser.extract_text(file)
resume_embedding = model.encode(resume_text)

roles = [
    {"role": "Data Analyst", "country": "UK", "salary": 60000},
    {"role": "Product Analyst", "country": "Canada", "salary": 75000},
    {"role": "Fraud Risk Analyst", "country": "India", "salary": 40000}
]

results = []

for item in roles:

    demand = demand_service.get_live_demand(item["role"])
    salary_norm = item["salary"] / 100000
    visa_score = visa_model.visa_score(item["country"])

    job_embedding = model.encode(item["role"])

    similarity, score = ranking_engine.compute_score(
        resume_embedding,
        job_embedding,
        demand / 100000 if demand else 0,
        salary_norm,
        visa_score
    )

    results.append({
        "Role": item["role"],
        "Country": item["country"],
        "Match %": round(similarity * 100, 2),
        "Score": round(score * 100, 2),
        "Live Demand": demand
    })

import pandas as pd
df = pd.DataFrame(results).sort_values("Score", ascending=False)

st.dataframe(df)

fig = px.bar(df, x="Role", y="Score", hover_data=["Live Demand"])
st.plotly_chart(fig)

if st.button("Generate Personalized Roadmap"):
    roadmap = gpt_service.generate_roadmap(resume_text, df.iloc[0]["Role"])
    st.write(roadmap)
