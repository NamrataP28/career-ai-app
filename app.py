import streamlit as st
import pandas as pd

st.set_page_config(page_title="AI Career Intelligence", layout="wide")

# ---------------------------
# HEADER
# ---------------------------

st.title("🚀 AI Career Intelligence Platform")
st.markdown("AI-powered career matching & probability engine")
st.markdown("---")

# ---------------------------
# SIDEBAR PROFILE
# ---------------------------

st.sidebar.header("👤 Profile Setup")

skills_input = st.sidebar.text_input(
    "Enter your skills (comma separated)",
    "SQL, Python, Power BI"
)

experience = st.sidebar.slider("Years of Experience", 0, 15, 4)

skills = [s.strip() for s in skills_input.split(",")]

st.sidebar.markdown("---")
st.sidebar.write("🎯 Update your profile to recalculate matches")

# ---------------------------
# JOB DATA
# ---------------------------

jobs = [
    {
        "Role": "Senior Data Analyst",
        "Country": "Netherlands",
        "Skills": ["SQL", "Python", "Stakeholder Management"],
        "Exp": 5,
        "Competition": 0.6,
        "Demand": 0.72,
        "Visa": 0.7
    },
    {
        "Role": "Product Analyst",
        "Country": "UK",
        "Skills": ["SQL", "A/B Testing"],
        "Exp": 4,
        "Competition": 0.75,
        "Demand": 0.68,
        "Visa": 0.8
    },
    {
        "Role": "Fraud Risk Analyst",
        "Country": "Singapore",
        "Skills": ["Fraud Analysis", "SQL"],
        "Exp": 3,
        "Competition": 0.58,
        "Demand": 0.74,
        "Visa": 0.9
    }
]

# ---------------------------
# MATCHING ENGINE
# ---------------------------

results = []

for job in jobs:
    overlap = len(set(skills) & set(job["Skills"]))
    skill_score = overlap / len(job["Skills"])
    exp_score = 1 if experience >= job["Exp"] else experience / job["Exp"]

    match_score = 0.5 * skill_score + 0.5 * exp_score

    probability = (
        match_score *
        (1 - job["Competition"]) *
        job["Demand"] *
        job["Visa"]
    )

    results.append({
        "Role": job["Role"],
        "Country": job["Country"],
        "Match %": round(match_score * 100, 2),
        "Hiring Probability %": round(probability * 100, 2)
    })

df = pd.DataFrame(results).sort_values(
    by="Hiring Probability %",
    ascending=False
)

# ---------------------------
# KPI SECTION
# ---------------------------

st.subheader("📊 Career Intelligence Overview")

col1, col2, col3 = st.columns(3)

best_role = df.iloc[0]["Role"]
best_prob = df.iloc[0]["Hiring Probability %"]
avg_match = round(df["Match %"].mean(), 2)

col1.metric("🏆 Best Role", best_role)
col2.metric("📈 Highest Probability", f"{best_prob}%")
col3.metric("📊 Avg Match Score", f"{avg_match}%")

st.markdown("---")

# ---------------------------
# TABS
# ---------------------------

tab1, tab2, tab3 = st.tabs(
    ["🎯 Job Matches", "📈 Skill Growth", "🤖 Career Advisor"]
)

# ---------------------------
# TAB 1 — MATCHES
# ---------------------------

with tab1:
    st.subheader("Top Ranked Opportunities")
    st.dataframe(df, use_container_width=True)

# ---------------------------
# TAB 2 — SKILL SIMULATION
# ---------------------------

with tab2:

    st.subheader("🔎 Skill Gap Analysis")

    selected_role = st.selectbox("Select Role", df["Role"])

    selected_job = next(j for j in jobs if j["Role"] == selected_role)

    matched = set(skills) & set(selected_job["Skills"])
    missing = set(selected_job["Skills"]) - set(skills)

    st.write("✅ Matched Skills:", list(matched))
    st.write("❌ Missing Skills:", list(missing))

    st.markdown("---")

    st.subheader("📈 Simulate Skill Improvement")

    new_skill = st.text_input("Add a new skill to simulate")

    if st.button("Simulate Improvement"):

        updated_skills = skills + [new_skill]

        overlap = len(set(updated_skills) & set(selected_job["Skills"]))
        skill_score = overlap / len(selected_job["Skills"])
        exp_score = 1 if experience >= selected_job["Exp"] else experience / selected_job["Exp"]

        new_match = 0.5 * skill_score + 0.5 * exp_score

        new_prob = (
            new_match *
            (1 - selected_job["Competition"]) *
            selected_job["Demand"] *
            selected_job["Visa"]
        )

        old_prob = df[df["Role"] == selected_role]["Hiring Probability %"].values[0]

        delta = round((new_prob * 100) - old_prob, 2)

        st.success(
            f"New Hiring Probability: {round(new_prob * 100,2)}%  ( +{delta}% improvement )"
        )

# ---------------------------
# TAB 3 — CHATBOT
# ---------------------------

with tab3:

    st.subheader("🧠 Strategic Career Advisor")

    question = st.text_input("Ask a strategic career question")

    if question:

        st.info(
            f"""
            Based on your profile:
            - Your strongest opportunity is **{best_role}**
            - Highest hiring probability: **{best_prob}%**
            - Improve skill overlap to increase probability
            - Target markets with lower competition index
            """
        )
