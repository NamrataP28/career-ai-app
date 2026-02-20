import streamlit as st
import pandas as pd
import numpy as np
import pdfplumber
import re
from openai import OpenAI

st.set_page_config(page_title="AI Career Intelligence", layout="wide")

st.title("🚀 AI Career Intelligence Platform")
st.markdown("Upload resume → Get global job probability insights")

# -----------------------------
# 1️⃣ RESUME UPLOAD
# -----------------------------

st.sidebar.header("📄 Resume Upload")

uploaded_file = st.sidebar.file_uploader("Upload Resume (PDF)", type=["pdf"])

extracted_skills = []
experience = 0
resume_text = ""

if uploaded_file:
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            resume_text += page.extract_text()

    # Basic skill extraction dictionary
    skill_db = [
        "SQL","Python","Power BI","Excel",
        "Stakeholder Management","A/B Testing",
        "Fraud Analysis","Machine Learning",
        "Product Strategy","Data Visualization"
    ]

    extracted_skills = [
        skill for skill in skill_db
        if skill.lower() in resume_text.lower()
    ]

    exp_match = re.search(r'(\d+)\+?\s*years', resume_text.lower())
    experience = int(exp_match.group(1)) if exp_match else 0

    st.sidebar.success("Resume processed successfully")

else:
    st.warning("Please upload resume to continue")

# -----------------------------
# 2️⃣ JOB DATASET (SCALABLE)
# -----------------------------

jobs = [
    {
        "Role": "Senior Data Analyst",
        "Country": "Netherlands",
        "Skills": ["SQL","Python","Stakeholder Management"],
        "Exp": 5,
        "Competition": 0.6,
        "Demand": 0.72,
        "Visa": 0.7
    },
    {
        "Role": "Product Analyst",
        "Country": "UK",
        "Skills": ["SQL","A/B Testing"],
        "Exp": 4,
        "Competition": 0.75,
        "Demand": 0.68,
        "Visa": 0.8
    },
    {
        "Role": "Fraud Risk Analyst",
        "Country": "Singapore",
        "Skills": ["Fraud Analysis","SQL"],
        "Exp": 3,
        "Competition": 0.58,
        "Demand": 0.74,
        "Visa": 0.9
    }
]

# -----------------------------
# 3️⃣ MATCHING ENGINE
# -----------------------------

results = []

for job in jobs:

    overlap = len(set(extracted_skills) & set(job["Skills"]))
    skill_score = overlap / len(job["Skills"]) if job["Skills"] else 0

    exp_score = (
        1 if experience >= job["Exp"]
        else experience / job["Exp"] if job["Exp"] > 0 else 1
    )

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

# -----------------------------
# 4️⃣ DASHBOARD OUTPUT
# -----------------------------

if uploaded_file:

    st.subheader("👤 Extracted Profile")
    st.write("Skills Detected:", extracted_skills)
    st.write("Estimated Experience:", experience, "years")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    best_role = df.iloc[0]["Role"]
    best_prob = df.iloc[0]["Hiring Probability %"]
    avg_match = round(df["Match %"].mean(), 2)

    col1.metric("🏆 Best Role", best_role)
    col2.metric("📈 Highest Probability", f"{best_prob}%")
    col3.metric("📊 Avg Match Score", f"{avg_match}%")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(
        ["🎯 Matches","📈 Skill Simulation","🤖 Career Advisor"]
    )

    # -----------------------------
    # TAB 1
    # -----------------------------

    with tab1:
        st.subheader("Top Global Opportunities")
        st.dataframe(df, use_container_width=True)

    # -----------------------------
    # TAB 2
    # -----------------------------

    with tab2:

        selected_role = st.selectbox("Select Role", df["Role"])
        selected_job = next(j for j in jobs if j["Role"] == selected_role)

        matched = set(extracted_skills) & set(selected_job["Skills"])
        missing = set(selected_job["Skills"]) - set(extracted_skills)

        st.write("✅ Matched Skills:", list(matched))
        st.write("❌ Missing Skills:", list(missing))

        new_skill = st.text_input("Add skill to simulate")

        if st.button("Simulate"):

            updated_skills = extracted_skills + [new_skill]

            overlap = len(set(updated_skills) & set(selected_job["Skills"]))
            skill_score = overlap / len(selected_job["Skills"])

            exp_score = (
                1 if experience >= selected_job["Exp"]
                else experience / selected_job["Exp"]
            )

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
                f"New Probability: {round(new_prob * 100,2)}%  ( +{delta}% improvement )"
            )

    # -----------------------------
    # TAB 3 (AI ADVISOR)
    # -----------------------------

    with tab3:

        question = st.text_input("Ask career strategy question")

        if question:

            st.info(
                f"""
                Based on your resume:

                • Strongest role: {best_role}
                • Highest probability: {best_prob}%
                • Improve missing skills for selected role
                • Target lower competition countries
                """
            )
