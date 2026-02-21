import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from services.resume_parser import ResumeParser
from services.visa_model import VisaModel
from services.gpt_service import GPTService

st.set_page_config(layout="wide")
st.title("🚀 AI Career Intelligence Engine — Phase 2")

model = SentenceTransformer("all-MiniLM-L6-v2")
parser = ResumeParser()
visa_model = VisaModel()
gpt_service = GPTService()

# -----------------------------------
# STEP 1: USER INPUT
# -----------------------------------

st.header("Step 1: Career Preferences")

career_goal = st.selectbox(
    "Primary Goal",
    ["Salary Growth", "Global Mobility", "Leadership Growth", "Stability"]
)

preferred_roles = st.multiselect(
    "Select Target Roles",
    [
        "Data Analyst","Business Analyst","Product Manager",
        "Marketing Manager","Financial Analyst",
        "Strategy Consultant","Operations Manager",
        "Software Engineer","Machine Learning Engineer"
    ]
)

preferred_countries = st.multiselect(
    "Preferred Countries",
    ["India","USA","UK","Canada","Germany","Singapore","Australia"]
)

current_salary = st.number_input("Current Salary (USD)", min_value=0)

if not preferred_roles:
    st.stop()

# -----------------------------------
# STEP 2: RESUME
# -----------------------------------

file = st.file_uploader("Upload Resume", type=["pdf"])
if not file:
    st.stop()

resume_text = parser.extract_text(file)
resume_embedding = model.encode(resume_text)

# -----------------------------------
# SKILL CLUSTERS
# -----------------------------------

skill_clusters = {
    "Data Analyst":["sql","python","excel","statistics","power bi"],
    "Product Manager":["roadmap","stakeholder","strategy","experimentation"],
    "Marketing Manager":["seo","campaign","branding","growth"],
    "Financial Analyst":["valuation","forecast","modeling","finance"],
    "Software Engineer":["backend","frontend","api","cloud"],
}

def skill_coverage(role):
    role_skills = skill_clusters.get(role, [])
    if not role_skills:
        return 0.4
    matches = sum([1 for s in role_skills if s in resume_text.lower()])
    return matches / len(role_skills)

# -----------------------------------
# MARKET SIMULATION
# -----------------------------------

def generate_market():
    rows = []
    for role in preferred_roles:
        for country in preferred_countries:
            rows.append({
                "Role": role,
                "Country": country,
                "Demand": np.random.randint(200, 1000),
                "Salary": np.random.randint(60000, 250000)
            })
    return pd.DataFrame(rows)

market_df = generate_market()

max_demand = market_df["Demand"].max()
max_salary = market_df["Salary"].max()

# -----------------------------------
# SCORING ENGINE
# -----------------------------------

results = []

for _, row in market_df.iterrows():

    role_emb = model.encode(row["Role"])
    similarity = cosine_similarity([resume_embedding],[role_emb])[0][0]

    skill_score = skill_coverage(row["Role"])
    demand_norm = row["Demand"] / max_demand
    salary_norm = row["Salary"] / max_salary
    visa_score = visa_model.visa_score(row["Country"])

    # Salary delta
    if current_salary > 0:
        salary_delta = max(row["Salary"] - current_salary, 0) / max_salary
    else:
        salary_delta = 0.5

    # Transition risk (lower similarity = higher risk)
    transition_risk = 1 - similarity

    # Interview probability model
    interview_prob = (
        0.4*similarity +
        0.2*skill_score +
        0.2*demand_norm +
        0.1*visa_score +
        0.1*salary_delta
    )

    final_score = (
        0.30*similarity +
        0.20*skill_score +
        0.15*demand_norm +
        0.15*salary_delta +
        0.10*visa_score -
        0.10*transition_risk
    )

    results.append({
        "Role":row["Role"],
        "Country":row["Country"],
        "Resume Match %":round(similarity*100,2),
        "Skill Coverage %":round(skill_score*100,2),
        "Interview Probability %":round(interview_prob*100,2),
        "Transition Risk %":round(transition_risk*100,2),
        "Salary Delta %":round(salary_delta*100,2),
        "Opportunity Score %":round(final_score*100,2)
    })

ranked_df = pd.DataFrame(results).sort_values(
    "Opportunity Score %",
    ascending=False
)

# -----------------------------------
# MARKET PERCENTILE
# -----------------------------------

ranked_df["Market Percentile %"] = ranked_df["Opportunity Score %"].rank(pct=True)*100

# -----------------------------------
# DISPLAY
# -----------------------------------

st.header("📊 Market Intelligence Dashboard")

st.dataframe(ranked_df, use_container_width=True)

fig = px.bar(
    ranked_df,
    x="Opportunity Score %",
    y="Role",
    color="Country",
    orientation="h",
    hover_data=[
        "Resume Match %",
        "Skill Coverage %",
        "Interview Probability %",
        "Transition Risk %",
        "Market Percentile %"
    ],
    height=550
)

fig.update_layout(yaxis=dict(categoryorder="total ascending"))
st.plotly_chart(fig, use_container_width=True)

# -----------------------------------
# MARKET POSITION
# -----------------------------------

avg_percentile = ranked_df["Market Percentile %"].mean()

st.subheader("📈 Your Market Standing")

st.metric("Average Market Percentile", f"{round(avg_percentile,2)}%")

if avg_percentile > 70:
    st.success("You are top-tier competitive.")
elif avg_percentile > 50:
    st.info("You are moderately competitive.")
else:
    st.warning("Upskilling recommended.")

# -----------------------------------
# GPT GAP ANALYSIS
# -----------------------------------

if st.button("Generate Advanced Skill Gap Report"):

    top_role = ranked_df.iloc[0]["Role"]

    prompt = f"""
    Candidate resume:
    {resume_text}

    Target role:
    {top_role}

    Provide:
    1. Specific skill gaps
    2. Certifications
    3. Portfolio project ideas
    4. 90-day improvement plan
    5. Interview preparation strategy
    """

    roadmap = gpt_service.generate_roadmap(resume_text, top_role)
    st.markdown(roadmap)

st.markdown("---")
st.success("Your career growth is compounding. Keep building.")
