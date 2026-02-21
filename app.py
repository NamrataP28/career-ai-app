import streamlit as st
import plotly.express as px
import pandas as pd
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from services.resume_parser import ResumeParser
from services.visa_model import VisaModel
from services.gpt_service import GPTService

# -----------------------------------
# INITIALIZE
# -----------------------------------

st.set_page_config(layout="wide")
st.title("🚀 AI Career Intelligence Platform")

model = SentenceTransformer("all-MiniLM-L6-v2")
parser = ResumeParser()
visa_model = VisaModel()
gpt_service = GPTService()

# -----------------------------------
# FILE UPLOAD
# -----------------------------------

file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if not file:
    st.stop()

resume_text = parser.extract_text(file)
resume_embedding = model.encode(resume_text)

# -----------------------------------
# INDUSTRY AUTO DETECTION
# -----------------------------------

def detect_industry(resume_embedding):

    industries = {
        "Data & Analytics": "data analytics business intelligence sql python statistics",
        "Product": "product management roadmap stakeholder ux experimentation",
        "Finance": "finance accounting valuation investment risk audit",
        "Marketing": "marketing growth seo campaign branding performance",
        "Software Engineering": "software engineering backend frontend cloud devops",
        "Risk & Compliance": "fraud compliance regulatory aml kyc",
        "Consulting": "consulting strategy advisory transformation"
    }

    best_match = None
    best_score = -1

    for industry, keywords in industries.items():
        emb = model.encode(keywords)
        score = cosine_similarity([resume_embedding], [emb])[0][0]

        if score > best_score:
            best_score = score
            best_match = industry

    return best_match


industry = detect_industry(resume_embedding)

st.subheader("🧠 Detected Industry")
st.success(industry)

# -----------------------------------
# SKILL EXTRACTION
# -----------------------------------

def extract_skills(text):

    skills_db = [
        "python", "sql", "excel", "power bi", "tableau",
        "machine learning", "aws", "azure", "react",
        "financial modeling", "risk analysis",
        "product strategy", "seo", "marketing analytics"
    ]

    text_lower = text.lower()
    return [s for s in skills_db if s in text_lower]


resume_skills = extract_skills(resume_text)

# -----------------------------------
# LIVE ROLE FETCH WITH SALARY + PPP
# -----------------------------------

@st.cache_data(ttl=3600)
def fetch_roles(industry):

    countries = {
        "UK": "gb",
        "USA": "us",
        "Canada": "ca",
        "Australia": "au",
        "India": "in",
        "Singapore": "sg"
    }

    # PPP multipliers (approximate)
    ppp_index = {
        "UK": 1.0,
        "USA": 1.0,
        "Canada": 0.9,
        "Australia": 0.95,
        "India": 0.35,
        "Singapore": 1.1
    }

    roles = []

    for country_name, code in countries.items():

        url = f"https://api.adzuna.com/v1/api/jobs/{code}/search/1"

        params = {
            "app_id": st.secrets["ADZUNA_APP_ID"],
            "app_key": st.secrets["ADZUNA_APP_KEY"],
            "results_per_page": 20,
            "what": industry
        }

        try:
            r = requests.get(url, params=params)
            data = r.json()

            total_demand = data.get("count", 0)

            for job in data.get("results", []):

                salary_min = job.get("salary_min") or 0
                salary_max = job.get("salary_max") or 0
                avg_salary = (salary_min + salary_max) / 2 if salary_max else salary_min

                salary_ppp = avg_salary / ppp_index[country_name] if avg_salary else 0

                roles.append({
                    "Role": job.get("title"),
                    "Country": country_name,
                    "Demand": total_demand,
                    "Salary": avg_salary,
                    "Salary_PPP": salary_ppp,
                    "Description": job.get("description")
                })

        except:
            continue

    return pd.DataFrame(roles).drop_duplicates(subset=["Role", "Country"])


roles_df = fetch_roles(industry)

if roles_df.empty:
    st.warning("No live roles found.")
    st.stop()

# -----------------------------------
# RESUME-FIRST GLOBAL SCORING
# -----------------------------------

results = []

max_demand = roles_df["Demand"].max() or 1
max_salary_ppp = roles_df["Salary_PPP"].max() or 1

for _, row in roles_df.iterrows():

    job_embedding = model.encode(row["Role"])
    similarity = cosine_similarity([resume_embedding], [job_embedding])[0][0]

    visa_score = visa_model.visa_score(row["Country"])
    demand_norm = row["Demand"] / max_demand
    salary_norm = row["Salary_PPP"] / max_salary_ppp

    job_skills = extract_skills(row["Description"] or "")
    skill_overlap = len(set(resume_skills) & set(job_skills))
    skill_match = skill_overlap / len(job_skills) if job_skills else 0

    final_score = (
        0.45 * similarity +
        0.20 * demand_norm +
        0.15 * visa_score +
        0.10 * salary_norm +
        0.10 * skill_match
    )

    results.append({
        "Role": row["Role"],
        "Country": row["Country"],
        "Match %": round(similarity * 100, 2),
        "Skill Match %": round(skill_match * 100, 2),
        "Visa Score": round(visa_score * 100, 2),
        "Market Demand %": round(demand_norm * 100, 2),
        "Salary (PPP Adjusted)": round(row["Salary_PPP"], 0),
        "Overall Score": round(final_score * 100, 2)
    })

ranked_df = pd.DataFrame(results).sort_values("Overall Score", ascending=False)

# -----------------------------------
# COUNTRY FILTER
# -----------------------------------

st.subheader("🌍 Global Role Ranking")

country_filter = st.selectbox(
    "Filter by Country",
    ["Worldwide"] + sorted(ranked_df["Country"].unique())
)

display_df = ranked_df if country_filter == "Worldwide" else ranked_df[ranked_df["Country"] == country_filter]

st.dataframe(display_df.head(20), use_container_width=True)

# -----------------------------------
# INTERACTIVE GRAPH
# -----------------------------------

fig = px.bar(
    display_df.head(10),
    x="Overall Score",
    y="Role",
    orientation="h",
    color="Country",
    hover_data=[
        "Match %",
        "Skill Match %",
        "Visa Score",
        "Market Demand %",
        "Salary (PPP Adjusted)"
    ],
    height=450
)

fig.update_layout(yaxis=dict(categoryorder="total ascending"))
st.plotly_chart(fig, use_container_width=True)

# -----------------------------------
# CAREER CLUSTERING
# -----------------------------------

st.subheader("🔎 Adjacent Career Paths")

top_embedding = model.encode(ranked_df.iloc[0]["Role"])

ranked_df["Role Similarity"] = ranked_df["Role"].apply(
    lambda r: cosine_similarity(
        [top_embedding],
        [model.encode(r)]
    )[0][0]
)

cluster_df = ranked_df.sort_values("Role Similarity", ascending=False).iloc[1:6]

st.dataframe(
    cluster_df[["Role", "Country", "Overall Score"]],
    use_container_width=True
)

# -----------------------------------
# GPT ROADMAP
# -----------------------------------

st.subheader("🎯 AI Career Roadmap")

if st.button("Generate Personalized Roadmap for Top Role"):

    top_role = ranked_df.iloc[0]["Role"]

    roadmap = gpt_service.generate_roadmap(resume_text, top_role)

    st.write(roadmap)
