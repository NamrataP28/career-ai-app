import streamlit as st
import pandas as pd
import numpy as np

st.title("Step 2: Market Intelligence Analysis")

if "user_inputs" not in st.session_state:
    st.warning("Complete Questionnaire first.")
    st.stop()

inputs = st.session_state["user_inputs"]

tabs = st.tabs([
    "Overview",
    "Global Ranking",
    "Salary Benchmark",
    "Skill Gap",
    "Competitiveness"
])

# -------- OVERVIEW --------
with tabs[0]:
    col1, col2, col3, col4 = st.columns(4)

    opportunity_score = round(np.random.uniform(55, 85),2)
    demand = np.random.randint(200, 1500)
    supply = np.random.randint(500, 4000)
    tightness = round((demand/supply)*100,2)

    col1.metric("Opportunity Score", f"{opportunity_score}")
    col2.metric("Live Demand", demand)
    col3.metric("Supply Estimate", supply)
    col4.metric("Market Tightness", f"{tightness}%")

# -------- GLOBAL RANKING --------
with tabs[1]:
    df = pd.DataFrame({
        "Country": inputs["country"],
        "Opportunity Score": np.random.uniform(50,90,len(inputs["country"]))
    }).sort_values("Opportunity Score", ascending=False)

    st.dataframe(df)

# -------- SALARY --------
with tabs[2]:
    projected_salary = round(inputs["salary"] * np.random.uniform(1.1,1.5),2)
    st.metric("Projected Salary", f"{projected_salary} {inputs['currency']}")

# -------- SKILL GAP --------
with tabs[3]:
    st.write("Recommended Skill Improvements:")
    st.write("- Advanced SQL")
    st.write("- System Design")
    st.write("- Cloud Fundamentals")

# -------- COMPETITIVENESS --------
with tabs[4]:
    percentile = round(np.random.uniform(40,90),2)
    st.metric("Your Estimated Market Percentile", f"{percentile}%")
