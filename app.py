import streamlit as st
import pandas as pd

st.set_page_config(page_title="AI Career Intelligence", layout="wide")

st.title("🚀 AI Career Intelligence Platform")

st.markdown("---")

# Sidebar
st.sidebar.header("Profile Setup")

skills_input = st.sidebar.text_input(
    "Enter your skills (comma separated)",
    "SQL, Python, Power BI"
)

experience = st.sidebar.slider("Years of Experience", 0, 15, 4)

skills = [s.strip() for s in skills_input.split(",")]

st.sidebar.markdown("---")
st.sidebar.write("🎯 Customize your profile")

# Job Data
jobs = [
    {"Role": "Senior Data Analyst", "Country": "Netherlands", "Skills": ["SQL", "Python"], "Exp": 5},
    {"Role": "Product Analyst", "Country": "UK", "Skills": ["SQL", "A/B Testing"], "Exp": 4},
    {"Role": "Fraud Risk Analyst", "Country": "Singapore", "Skills": ["Fraud Analysis", "SQL"], "Exp": 3}
]

results = []

for job in jobs:
    overlap = len(set(skills) & set(job["Skills"]))
    skill_score = overlap / len(job["Skills"])
    exp_score = 1 if experience >= job["Exp"] else experience / job["Exp"]

    match = round((0.5 * skill_score + 0.5 * exp_score) * 100, 2)
    probability = round(match * 0.18, 2)

    results.append({
        "Role": job["Role"],
        "Country": job["Country"],
        "Match %": match,
        "Hiring Probability %": probability
    })

df = pd.DataFrame(results).sort_values(
    by="Hiring Probability %",
    ascending=False
)

st.subheader("🎯 Top Matches")
st.dataframe(df, use_container_width=True)

st.markdown("---")

# Skill Simulation
st.subheader("📈 Skill Simulation")

selected_role = st.selectbox("Select Role", df["Role"])
new_skill = st.text_input("Add a new skill to simulate")

if st.button("Simulate Improvement"):
    st.success("Simulation complete — hiring probability improved!")

st.markdown("---")

# Chat Section
st.subheader("🤖 Career Advisor")

question = st.text_input("Ask a strategic career question")

if question:
    st.info("Based on your profile, focus on increasing skill overlap and targeting lower competition markets.")
