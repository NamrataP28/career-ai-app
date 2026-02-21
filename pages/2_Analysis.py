from services.demand_service import fetch_live_demand
from services.supply_engine import estimate_supply
from services.salary_service import extract_salary
from services.ranking_engine import calculate_opportunity

role = inputs["role"]
countries = inputs["country"]

for c in countries:
    demand, jobs = fetch_live_demand(role, c)
    supply = estimate_supply(role, c)
    avg_salary = extract_salary(jobs)

    salary_growth = 50  # temporary normalization

    resume_score = 75  # plug embedding engine here

    opp_score, tightness = calculate_opportunity(
        resume_score,
        demand,
        supply,
        salary_growth
    )

    st.metric(f"{c} Opportunity Score", opp_score)
    st.write(f"Market Tightness: {tightness}%")
    st.write(f"Live Demand: {demand}")
    st.write(f"Estimated Supply: {supply}")
    st.write(f"Avg Salary: {round(avg_salary,2)} {inputs['currency']}")
