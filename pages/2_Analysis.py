import streamlit as st
from services.resume_scoring_engine import compute_resume_match
from services.skill_engine import compute_skill_gap
from services.probability_engine import interview_probability
from services.benchmark_engine import percentile_rank
from services.demand_service import fetch_live_demand
from services.supply_engine import estimate_supply
from services.salary_service import extract_salary
from services.ranking_engine import calculate_opportunity

# -----------------------------------
# Session Validation
# -----------------------------------

if "user_inputs" not in st.session_state:
    st.warning("Please complete Questionnaire first.")
    st.stop()

inputs = st.session_state["user_inputs"]

role = inputs["role"]
countries = inputs["country"]
resume_text = st.session_state.get("resume_text", "")

# -----------------------------------
# Tabs
# -----------------------------------

tabs = st.tabs([
    "Overview",
    "Market Depth",
    "Skill Gap",
    "Companies",
    "Competitiveness",
    "Benchmark"
])

for c in countries:

    demand, jobs, companies = fetch_live_demand(role, c)
    supply = estimate_supply(role, c)
    avg_salary = extract_salary(jobs)

    resume_score = compute_resume_match(
    resume_text,
    role,
    jobs
) # replace with embedding similarity

    skill_score, missing_skills = compute_skill_gap(resume_text, jobs)

    tightness = (demand / supply) * 100 if supply > 0 else 0

    opp_score, _ = calculate_opportunity(
        resume_score,
        demand,
        supply,
        60
    )

    probability = interview_probability(resume_score, tightness, skill_score)
    percentile = percentile_rank(opp_score)

    with tabs[0]:
        st.subheader(f"{c} Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Opportunity Score", opp_score)
        col2.metric("Interview Probability %", probability)
        col3.metric("Market Percentile %", percentile)

    with tabs[1]:
        st.write(f"Live Demand: {demand}")
        st.write(f"Estimated Supply: {supply}")
        st.write(f"Market Tightness: {round(tightness,2)}%")
        st.write(f"Average Salary: {round(avg_salary,2)} {inputs['currency']}")

    with tabs[2]:
        st.metric("Skill Alignment %", skill_score)
        st.write("Missing High-Impact Skills:")
        for skill in missing_skills:
            st.write("-", skill)

    with tabs[3]:
        st.write("Top Hiring Companies:")
        for comp in companies:
            st.write("•", comp)

    with tabs[4]:
        st.metric("Resume Strength %", resume_score)
        st.metric("Market Tightness %", round(tightness,2))
        st.metric("Interview Probability %", probability)

    with tabs[5]:
        st.metric("Global Market Percentile", percentile)
        st.write(f"You outperform approximately {percentile}% of candidates.")
