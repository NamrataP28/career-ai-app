import streamlit as st
import pandas as pd
import pdfplumber
import re
import requests
import plotly.express as px
import pycountry
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# -----------------------------------
# CONFIG
# -----------------------------------
st.set_page_config(page_title="AI Career Intelligence", layout="wide")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# -----------------------------------
# LOAD MODEL
# -----------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -----------------------------------
# ADZUNA LIVE JOB COUNT
# -----------------------------------
def get_live_demand(role, country="gb"):
    try:
        url = f"https://api.adzuna.com/v1/api/jobs/{country}/search/1"
        params = {
            "app_id": st.secrets["ADZUNA_APP_ID"],
            "app_key": st.secrets["ADZUNA_APP_KEY"],
            "results_per_page": 1,
            "what": role
        }
        response = requests.get(url, params=params)
        data = response.json()
        return data.get("count", 0)
    except:
        return 0

# -----------------------------------
# RESUME PROCESSING
# -----------------------------------
st.title("🚀 AI Career Intelligence Platform")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if not uploaded_file:
    st.stop()

resume_text = ""
with pdfplumber.open(uploaded_file) as pdf:
    for page in pdf.pages:
        resume_text += page.extract_text()

resume_embedding = model.encode(resume_text)

# Extract skills from resume (basic version)
resume_skills = re.findall(r"\b[A-Za-z\+]{2,}\b", resume_text.lower())
resume_skills = list(set(resume_skills))

# -----------------------------------
# SAMPLE ROLE LIST (Replace with DB later)
# -----------------------------------
roles = [
    "Business Intelligence Analyst",
    "Data Analyst",
    "Product Analyst",
    "Fraud Risk Analyst",
    "Senior Data Analyst"
]

results = []

for role in roles:

    job_embedding = model.encode(role)
    similarity = cosine_similarity(
        [resume_embedding],
        [job_embedding]
    )[0][0]

    live_demand = get_live_demand(role)

    salary_estimate = 70000  # placeholder

    salary_norm = salary_estimate / 100000

    global_score = (
        0.5 * similarity +
        0.3 * (live_demand / 100000 if live_demand else 0) +
        0.2 * salary_norm
    )

    results.append({
        "Role": role,
        "Match %": round(similarity * 100, 2),
        "Live Demand": live_demand,
        "Interview Probability": round(global_score * 100, 2)
    })

df = pd.DataFrame(results)
df = df.sort_values("Interview Probability", ascending=False)

st.subheader("🌍 Global Role Ranking (Resume First)")
st.dataframe(df, use_container_width=True)

# -----------------------------------
# INTERACTIVE GRAPH
# -----------------------------------
fig = px.bar(
    df,
    x="Role",
    y="Interview Probability",
    hover_data=["Match %", "Live Demand"],
    height=400
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------------
# SKILL GAP ANALYSIS
# -----------------------------------
st.subheader("🧠 Skill Gap Analysis")

top_role = df.iloc[0]["Role"]

role_keywords = top_role.lower().split()
missing_skills = [
    word for word in role_keywords
    if word not in resume_skills
]

st.write("Top Role:", top_role)
st.write("Potential Missing Skills:", missing_skills)

# -----------------------------------
# GPT ROADMAP
# -----------------------------------
st.subheader("🎯 Personalized Roadmap")

if st.button("Generate AI Roadmap"):

    prompt = f"""
    The candidate resume text is:
    {resume_text}

    Target role:
    {top_role}

    Provide:
    1. Skill gaps
    2. Certifications
    3. Project ideas
    4. 90-day roadmap
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    st.write(response.choices[0].message.content)
