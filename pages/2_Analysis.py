from services.skill_engine import compute_skill_gap
from services.probabilty_engine import probabilty_engine
from services.benchmark_engine import percentile_rank

role = inputs["role"]
countries = inputs["country"]
resume_text = st.session_state.get("resume_text")

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

    resume_score = 75  # plug embedding similarity here

    skill_score, missing_skills = compute_skill_gap(resume_text, jobs)

    tightness = (demand / supply) * 100 if supply > 0 else 0

    opp_score, _ = calculate_opportunity(
        resume_score,
        demand,
        supply,
        60
    )

    probability = interview_probabilty(resume_score, tightness, skill_score)
    percentile = percentile_rank(opp_score)

    # -------- OVERVIEW --------
    with tabs[0]:
        st.subheader(f"{c} Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Opportunity Score", opp_score)
        col2.metric("Interview Probability %", probability)
        col3.metric("Market Percentile %", percentile)

    # -------- MARKET DEPTH --------
    with tabs[1]:
        st.write(f"Live Demand: {demand}")
        st.write(f"Estimated Supply: {supply}")
        st.write(f"Market Tightness: {round(tightness,2)}%")
        st.write(f"Average Salary: {round(avg_salary,2)} {inputs['currency']}")

    # -------- SKILL GAP --------
    with tabs[2]:
        st.metric("Skill Alignment %", skill_score)
        st.write("Missing High-Impact Skills:")
        for skill in missing_skills:
            st.write("-", skill)

    # -------- COMPANIES --------
    with tabs[3]:
        st.write("Top Hiring Companies:")
        for comp in companies:
            st.write("•", comp)

    # -------- COMPETITIVENESS --------
    with tabs[4]:
        st.metric("Resume Strength %", resume_score)
        st.metric("Market Tightness %", round(tightness,2))
        st.metric("Interview Probability %", probability)

    # -------- BENCHMARK --------
    with tabs[5]:
        st.metric("Global Market Percentile", percentile)
        st.write("You outperform approximately", percentile, "% of candidates.")
