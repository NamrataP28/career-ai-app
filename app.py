import streamlit as st
import plotly.express as px
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from services.resume_parser import ResumeParser
from services.visa_model import VisaModel
from services.gpt_service import GPTService

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(layout="wide")
st.title("🚀 AI Career Intelligence Platform — India Live")

# -----------------------------------
# INITIALIZE SERVICES
# -----------------------------------

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
# CORPORATE ROLE BANK
# -----------------------------------

corporate_keywords = [
    "Data Analyst",
    "Business Analyst",
    "Product Manager",
    "Marketing Manager",
    "Financial Analyst",
    "Risk Analyst",
    "Operations Manager",
    "Strategy Consultant",
    "Software Engineer",
    "Machine Learning Engineer",
    "Program Manager",
    "Finance Manager",
    "Growth Manager",
    "Corporate Strategy"
]

st.subheader("🔎 Searching Corporate Roles in India")

# -----------------------------------
# FETCH LIVE INDIA ROLES
# -----------------------------------

@st.cache_data(ttl=1800)
def fetch_india_roles(keywords):

    headers = {
        "X-RapidAPI-Key": st.secrets["RAPIDAPI_KEY"],
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
    }

    all_roles = []

    for keyword in keywords:

        url = "https://jsearch.p.rapidapi.com/search"

        params = {
            "query": f"{keyword} India",
            "page": "1",
            "num_pages": "1",
            "employment_types": "FULLTIME"
        }

        response = requests.get(url, headers=headers, params=params)

        if response.status_code != 200:
            continue

        data = response.json()
        jobs = data.get("data", [])

        for job in jobs:

            salary = job.get("job_salary") or 0

            all_roles.append({
                "Role": job.get("job_title"),
                "Company": job.get("employer_name"),
                "Location": job.get("job_city"),
                "Salary": salary
            })

    df = pd.DataFrame(all_roles)

    if df.empty:
        return df

    return df.drop_duplicates(subset=["Role", "Company"])


roles_df = fetch_india_roles(corporate_keywords)

if roles_df.empty:
    st.error("⚠ No live roles found. Check RapidAPI subscription.")
    st.stop()

# -----------------------------------
# RESUME-FIRST RANKING
# -----------------------------------

results = []

max_salary = roles_df["Salary"].max() if roles_df["Salary"].max() > 0 else 1

for _, row in roles_df.iterrows():

    job_embedding = model.encode(row["Role"])

    similarity = cosine_similarity(
        [resume_embedding],
        [job_embedding]
    )[0][0]

    salary_norm = row["Salary"] / max_salary if max_salary else 0

    final_score = (
        0.75 * similarity +
        0.25 * salary_norm
    )

    results.append({
        "Role": row["Role"],
        "Company": row["Company"],
        "Location": row["Location"],
        "Match %": round(similarity * 100, 2),
        "Salary": row["Salary"],
        "Opportunity Score %": round(final_score * 100, 2)
    })

ranked_df = pd.DataFrame(results).sort_values(
    "Opportunity Score %",
    ascending=False
)

# -----------------------------------
# DISPLAY RESULTS
# -----------------------------------

st.subheader("🇮🇳 India — Live Ranked Roles")

st.dataframe(
    ranked_df.head(20),
    use_container_width=True
)

# -----------------------------------
# VISUALIZATION
# -----------------------------------

fig = px.bar(
    ranked_df.head(10),
    x="Opportunity Score %",
    y="Role",
    orientation="h",
    hover_data=["Match %", "Salary", "Company"],
    height=450
)

fig.update_layout(yaxis=dict(categoryorder="total ascending"))

st.plotly_chart(fig, use_container_width=True)

# -----------------------------------
# GPT ROADMAP
# -----------------------------------

if st.button("Generate Career Roadmap for Top Role"):

    top_role = ranked_df.iloc[0]["Role"]

    roadmap = gpt_service.generate_roadmap(
        resume_text,
        top_role
    )

    st.markdown(roadmap)
