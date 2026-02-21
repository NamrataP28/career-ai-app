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
# LIVE ROLE FETCH WITH DEMAND
# -----------------------------------

@st.cache_data(ttl=3600)
def fetch_roles_with_demand(industry):

    countries = {
        "UK": "gb",
        "USA": "us",
        "Canada": "ca",
        "Australia": "au",
        "India": "in",
        "Singapore": "sg"
    }

    all_roles = []

    for country_name, country_code in countries.items():

        url = f"https://api.adzuna.com/v1/api/jobs/{country_code}/search/1"

        params = {
            "app_id": st.secrets["ADZUNA_APP_ID"],
            "app_key": st.secrets["ADZUNA_APP_KEY"],
            "results_per_page": 20,
            "what": industry
        }

        try:
            response = requests.get(url, params=params)
            data = response.json()

            total_count = data.get("count", 0)

            for job in data.get("results", []):
                all_roles.append({
                    "Role": job.get("title"),
                    "Country": country_name,
                    "Demand": total_count
                })

        except:
            continue

    df_roles = pd.DataFrame(all_roles).drop_duplicates(subset=["Role", "Country"])

    return df_roles


roles_df = fetch_roles_with_demand(industry)

if roles_df.empty:
    st.warning("No live roles found.")
    st.stop()

# -----------------------------------
# RESUME-FIRST GLOBAL SCORING
# -----------------------------------

results = []

max_demand = roles_df["Demand"].max() if roles_df["Demand"].max() > 0 else 1

for _, row in roles_df.iterrows():

    job_embedding = model.encode(row["Role"])

    similarity = cosine_similarity(
        [resume_embedding],
        [job_embedding]
    )[0][0]

    visa_score = visa_model.visa_score(row["Country"])

    demand_norm = row["Demand"] / max_demand

    # Balanced weighted score
    final_score = (
        0.6 * similarity +
        0.25 * demand_norm +
        0.15 * visa_score
    )

    results.append({
        "Role": row["Role"],
        "Country": row["Country"],
        "Match %": round(similarity * 100, 2),
        "Visa Score": round(visa_score * 100, 2),
        "Market Demand Index": round(demand_norm * 100, 2),
        "Overall Score": round(final_score * 100, 2)
    })

ranked_df = pd.DataFrame(results).sort_values(
    "Overall Score",
    ascending=False
)

# -----------------------------------
# COUNTRY FILTER
# -----------------------------------

st.subheader("🌍 Global Role Ranking")

country_filter = st.selectbox(
    "Filter by Country",
    ["Worldwide"] + sorted(ranked_df["Country"].unique())
)

if country_filter != "Worldwide":
    display_df = ranked_df[ranked_df["Country"] == country_filter]
else:
    display_df = ranked_df

st.dataframe(display_df.head(20), use_container_width=True)

# -----------------------------------
# INTERACTIVE VISUAL (CLEANER & SMARTER)
# -----------------------------------

fig = px.bar(
    display_df.head(10),
    x="Overall Score",
    y="Role",
    color="Country",
    orientation="h",
    hover_data=["Match %", "Visa Score", "Market Demand Index"],
    height=450
)

fig.update_layout(
    yaxis=dict(categoryorder="total ascending")
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------------
# GPT PERSONALIZED ROADMAP
# -----------------------------------

st.subheader("🎯 AI Career Roadmap")

if st.button("Generate Personalized Roadmap for Top Role"):

    top_role = ranked_df.iloc[0]["Role"]

    roadmap = gpt_service.generate_roadmap(
        resume_text,
        top_role
    )

    st.write(roadmap)
