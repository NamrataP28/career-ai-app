import streamlit as st
import pandas as pd

from services.resume_scoring_engine import compute_resume_match
from services.skill_engine import compute_skill_gap
from services.probability_engine import interview_probability
from services.demand_service import fetch_live_demand
from services.supply_engine import estimate_supply
from services.salary_service import extract_salary
from services.ranking_engine import calculate_opportunity


# --------------------------------------------------
# INLINE COUNTRY ATTRACTIVENESS (avoid import error)
# --------------------------------------------------

def country_attractiveness(country):

    index = {
        "USA": 90,
        "Germany": 85,
        "UK": 88,
        "Singapore": 80,
        "Canada": 82,
        "Australia": 83,
        "India": 70
    }

    return index.get(country, 75)


# --------------------------------------------------
# SESSION VALIDATION
# --------------------------------------------------

if "user_inputs" not in st.session_state:
    st.warning("Please complete Questionnaire first.")
    st.stop()

inputs = st.session_state["user_inputs"]

role = inputs["role"]
countries = inputs["country"]
resume_text = st.session_state.get("resume_text", "")
currency = inputs["currency"]

if not countries:
    st.warning("Please select at least one country.")
    st.stop()


# --------------------------------------------------
# FETCH + CALCULATE
# --------------------------------------------------

country_results = []

for c in countries:

    try:
        demand, jobs, companies = fetch_live_demand(role, c)
        supply = estimate_supply(role, c)
        avg_salary = extract_salary(jobs)
    except:
        continue

    if jobs is None:
        jobs = []

    if companies is None:
        companies = []

    resume_score = compute_resume_match(resume_text, role, jobs)
    skill_score, missing_skills = compute_skill_gap(resume_text, jobs)

    country_index = country_attractiveness(c)

    opp_score, tightness = calculate_opportunity(
        resume_score,
        demand,
        supply,
        60,
        country_index
    )

    probability = interview_probability(
        resume_score,
        tightness,
        skill_score
    )

    country_results.append({
        "Country": c,
        "Opportunity": round(opp_score, 2),
        "Probability": round(probability, 2),
        "Resume Score": round(resume_score, 2),
        "Skill Score": round(skill_score, 2),
        "Tightness": round(tightness, 2),
        "Demand": demand,
        "Supply": supply,
        "Salary": round(avg_salary, 2),
        "Companies": companies,
        "Missing Skills": missing_skills
    })


df = pd.DataFrame(country_results)

if df.empty:
    st.warning("No market data available.")
    st.stop()


# --------------------------------------------------
# GLOBAL RANKING
# --------------------------------------------------

df["Composite Score"] = (
    df["Opportunity"] * 0.5 +
    df["Probability"] * 0.3 +
    df["Resume Score"] * 0.2
).round(2)

df = df.sort_values("Composite Score", ascending=False).reset_index(drop=True)

df["Market Percentile"] = (
    df["Composite Score"].rank(pct=True) * 100
).round(2)


# --------------------------------------------------
# COUNTRY SELECTOR (Clean UX)
# --------------------------------------------------

st.subheader("🌍 Ranked Countries")

selected_country = st.selectbox(
    "Select Country for Detailed Analysis",
    df["Country"]
)

row = df[df["Country"] == selected_country].iloc[0]


# --------------------------------------------------
# TABS
# --------------------------------------------------

tabs = st.tabs([
    "Overview",
    "Market Depth",
    "Skill Gap",
    "Companies",
    "Competitiveness",
    "Benchmark"
])


# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

with tabs[0]:

    col1, col2, col3 = st.columns(3)

    col1.metric("Opportunity Score",
                f"{row['Opportunity']:.2f}")

    col2.metric("Interview Probability %",
                f"{row['Probability']:.2f}")

    col3.metric("Market Percentile %",
                f"{row['Market Percentile']:.2f}")


# --------------------------------------------------
# MARKET DEPTH
# --------------------------------------------------

with tabs[1]:

    st.write(f"Live Demand: {row['Demand']}")
    st.write(f"Estimated Supply: {row['Supply']}")
    st.write(f"Market Tightness: {row['Tightness']:.2f}%")
    st.write(f"Average Salary: {row['Salary']:.2f} {currency}")


# --------------------------------------------------
# SKILL GAP
# --------------------------------------------------

with tabs[2]:

    st.metric("Skill Alignment %",
              f"{row['Skill Score']:.2f}")

    st.write("Missing High-Impact Skills:")

    if row["Missing Skills"]:
        for skill in row["Missing Skills"]:
            st.write("-", skill)
    else:
        st.success("No major skill gaps detected.")


# --------------------------------------------------
# COMPANIES
# --------------------------------------------------

with tabs[3]:

    if row["Companies"]:
        for comp in row["Companies"]:
            st.write("•", comp)
    else:
        st.write("No company data available.")


# --------------------------------------------------
# COMPETITIVENESS
# --------------------------------------------------

with tabs[4]:

    col1, col2, col3 = st.columns(3)

    col1.metric("Resume Strength %",
                f"{row['Resume Score']:.2f}")

    col2.metric("Market Tightness %",
                f"{row['Tightness']:.2f}")

    col3.metric("Interview Probability %",
                f"{row['Probability']:.2f}")


# --------------------------------------------------
# BENCHMARK
# --------------------------------------------------

with tabs[5]:

    st.metric("Global Market Percentile",
              f"{row['Market Percentile']:.2f}")

    st.write(
        f"You outperform approximately "
        f"{row['Market Percentile']:.2f}% of candidates "
        f"in {row['Country']}."
    )
