import streamlit as st
import pandas as pd

from services.resume_scoring_engine import compute_resume_match
from services.skill_engine import compute_skill_gap
from services.probability_engine import interview_probability
from services.demand_service import fetch_live_demand
from services.supply_engine import estimate_supply
from services.salary_service import extract_salary
from services.ranking_engine import calculate_opportunity
from services.country_index import country_attractiveness


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
# FETCH + CALCULATE PER COUNTRY
# --------------------------------------------------

country_results = []

for c in countries:

    demand, jobs, companies = fetch_live_demand(role, c)
    supply = estimate_supply(role, c)
    avg_salary = extract_salary(jobs)

    # Resume match score (embedding-based)
    resume_score = compute_resume_match(resume_text, role, jobs)

    # Skill gap
    skill_score, missing_skills = compute_skill_gap(resume_text, jobs)

    # Country strength
    country_index = country_attractiveness(c)

    # Opportunity + Tightness
    opp_score, tightness = calculate_opportunity(
        resume_score,
        demand,
        supply,
        60,  # replace later with real salary growth model
        country_index
    )

    # Interview probability
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


# --------------------------------------------------
# GLOBAL RANKING MODEL
# --------------------------------------------------

df = pd.DataFrame(country_results)

if df.empty:
    st.warning("No live data found.")
    st.stop()

# Composite global ranking
df["Composite Score"] = (
    df["Opportunity"] * 0.5 +
    df["Probability"] * 0.3 +
    df["Resume Score"] * 0.2
)

df["Composite Score"] = df["Composite Score"].round(2)

# Sort countries
df = df.sort_values("Composite Score", ascending=False).reset_index(drop=True)

# True percentile
df["Market Percentile"] = (
    df["Composite Score"].rank(pct=True) * 100
).round(2)


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
# RENDER RESULTS
# --------------------------------------------------

for _, row in df.iterrows():

    # ---------- OVERVIEW ----------
    with tabs[0]:
        st.subheader(f"{row['Country']} Overview")

        col1, col2, col3 = st.columns(3)

        col1.metric("Opportunity Score",
                    f"{row['Opportunity']:.2f}")

        col2.metric("Interview Probability %",
                    f"{row['Probability']:.2f}")

        col3.metric("Market Percentile %",
                    f"{row['Market Percentile']:.2f}")


    # ---------- MARKET DEPTH ----------
    with tabs[1]:
        st.subheader(f"{row['Country']} Market Depth")

        st.write(f"Live Demand: {row['Demand']}")
        st.write(f"Estimated Supply: {row['Supply']}")
        st.write(f"Market Tightness: {row['Tightness']:.2f}%")
        st.write(f"Average Salary: {row['Salary']:.2f} {currency}")


    # ---------- SKILL GAP ----------
    with tabs[2]:
        st.subheader(f"{row['Country']} Skill Analysis")

        st.metric("Skill Alignment %",
                  f"{row['Skill Score']:.2f}")

        st.write("Missing High-Impact Skills:")

        if row["Missing Skills"]:
            for skill in row["Missing Skills"]:
                st.write("-", skill)
        else:
            st.success("No critical skill gaps detected.")


    # ---------- COMPANIES ----------
    with tabs[3]:
        st.subheader(f"{row['Country']} Top Hiring Companies")

        if row["Companies"]:
            for comp in row["Companies"]:
                st.write("•", comp)
        else:
            st.write("No company data available.")


    # ---------- COMPETITIVENESS ----------
    with tabs[4]:
        st.subheader(f"{row['Country']} Competitiveness")

        col1, col2, col3 = st.columns(3)

        col1.metric("Resume Strength %",
                    f"{row['Resume Score']:.2f}")

        col2.metric("Market Tightness %",
                    f"{row['Tightness']:.2f}")

        col3.metric("Interview Probability %",
                    f"{row['Probability']:.2f}")


    # ---------- BENCHMARK ----------
    with tabs[5]:
        st.subheader(f"{row['Country']} Benchmark")

        st.metric("Global Market Percentile",
                  f"{row['Market Percentile']:.2f}")

        st.write(
            f"You outperform approximately "
            f"{row['Market Percentile']:.2f}% of candidates "
            f"in {row['Country']}."
        )

        st.markdown("---")
