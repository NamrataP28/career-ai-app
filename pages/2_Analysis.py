import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import io

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

from services.resume_scoring_engine import compute_resume_match
from services.skill_engine import compute_skill_gap
from services.probability_engine import interview_probability
from services.demand_service import fetch_live_demand
from services.supply_engine import estimate_supply
from services.salary_service import extract_salary
from services.ranking_engine import calculate_opportunity
from services.gpt_service import GPTService


# --------------------------------------------------
# INIT
# --------------------------------------------------

gpt_service = GPTService()

if "user_inputs" not in st.session_state:
    st.warning("Please complete Questionnaire first.")
    st.stop()

inputs = st.session_state["user_inputs"]

role = inputs["role"]
alt_role = inputs.get("alt_role")
countries = inputs["country"]
resume_text = st.session_state.get("resume_text", "")
currency = inputs["currency"]
current_salary = inputs.get("salary", 0)

if not countries:
    st.warning("Please select at least one country.")
    st.stop()


# --------------------------------------------------
# COUNTRY ATTRACTIVENESS INLINE
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
# COMPUTE COUNTRY METRICS
# --------------------------------------------------

results = []

for c in countries:

    try:
        demand, jobs, companies = fetch_live_demand(role, c)
        supply = estimate_supply(role, c)
        avg_salary = extract_salary(jobs)
    except:
        continue

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

    results.append({
        "Country": c,
        "Opportunity": round(opp_score, 2),
        "Probability": round(probability, 2),
        "Resume Score": round(resume_score, 2),
        "Skill Score": round(skill_score, 2),
        "Tightness": round(tightness, 2),
        "Salary": round(avg_salary, 2),
        "Companies": companies,
        "Missing Skills": missing_skills
    })

df = pd.DataFrame(results)

if df.empty:
    st.warning("No live data available.")
    st.stop()

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
# COUNTRY SELECTOR
# --------------------------------------------------

st.subheader("🌍 Global Ranking")

selected_country = st.selectbox(
    "Select Country",
    df["Country"]
)

row = df[df["Country"] == selected_country].iloc[0]


# --------------------------------------------------
# TABS
# --------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Executive Overview",
    "Visual Intelligence",
    "Skill Strategy",
    "Growth Projection",
    "AI Advisor"
])


# ==================================================
# TAB 1 — EXECUTIVE OVERVIEW
# ==================================================

with tab1:

    col1, col2, col3 = st.columns(3)

    col1.metric("Opportunity Score", f"{row['Opportunity']:.2f}")
    col2.metric("Interview Probability %", f"{row['Probability']:.2f}")
    col3.metric("Market Percentile %", f"{row['Market Percentile']:.2f}")

    st.markdown("---")

    salary_gap = row["Salary"] - current_salary
    salary_gap_pct = (
        (salary_gap / current_salary * 100)
        if current_salary > 0 else 0
    )

    st.subheader("💰 Salary Benchmark")

    colA, colB = st.columns(2)

    colA.metric("Market Average Salary",
                f"{row['Salary']:.2f} {currency}")

    colB.metric("Salary Gap %",
                f"{salary_gap_pct:.2f}%")

    if alt_role:
        alt_score = compute_resume_match(resume_text, alt_role, [])
        st.markdown("---")
        st.subheader("🔎 Alternate Role Comparison")
        st.write(f"Primary Role Match: {row['Resume Score']:.2f}")
        st.write(f"Alternate Role Match: {alt_score:.2f}")


# ==================================================
# TAB 2 — VISUAL INTELLIGENCE
# ==================================================

with tab2:

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=row["Probability"],
        title={"text": "Interview Probability %"},
        gauge={"axis": {"range": [0, 100]}}
    ))

    st.plotly_chart(fig, use_container_width=True)

    categories = ["Resume", "Skill", "Market Tightness", "Opportunity"]

    radar = go.Figure()

    radar.add_trace(go.Scatterpolar(
        r=[
            row["Resume Score"],
            row["Skill Score"],
            row["Tightness"],
            row["Opportunity"]
        ],
        theta=categories,
        fill='toself'
    ))

    radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False
    )

    st.plotly_chart(radar, use_container_width=True)


# ==================================================
# TAB 3 — SKILL STRATEGY
# ==================================================

with tab3:

    st.metric("Skill Alignment %",
              f"{row['Skill Score']:.2f}")

    st.subheader("Missing High-Impact Skills")

    if row["Missing Skills"]:
        for skill in row["Missing Skills"]:
            st.write("•", skill)
    else:
        st.success("No major gaps detected.")

    st.markdown("---")

    st.subheader("Top Hiring Companies")

    for comp in row["Companies"]:
        comp_prob = row["Probability"] * 0.9
        st.write(f"• {comp} → Est. Interview Probability: {comp_prob:.2f}%")


# ==================================================
# TAB 4 — GROWTH PROJECTION
# ==================================================

with tab4:

    years = st.slider("Projection Years", 1, 5, 3)

    projected_salary = row["Salary"] * (1 + 0.08) ** years

    st.metric(
        f"Projected Salary in {years} years",
        f"{projected_salary:.2f} {currency}"
    )

    st.markdown("""
    Strategic Advice:
    - Improve 1 high-impact skill every 60 days  
    - Target firms in high tightness markets  
    - Negotiate based on percentile strength  
    """)


# ==================================================
# TAB 5 — AI ADVISOR
# ==================================================

with tab5:

    st.subheader("🤖 AI Career Advisor")

    user_question = st.text_area(
        "Ask anything about your positioning:",
        placeholder="How can I increase my probability in Germany?"
    )

    if st.button("Ask AI Advisor"):

        prompt = f"""
        Role: {role}
        Country: {selected_country}
        Resume Score: {row['Resume Score']}
        Opportunity Score: {row['Opportunity']}
        Interview Probability: {row['Probability']}
        Skill Gaps: {row['Missing Skills']}
        Salary: {row['Salary']}

        User Question:
        {user_question}

        Provide clear, actionable guidance.
        """

        response = gpt_service.generate_roadmap(resume_text, role)

        st.markdown(response)


# ==================================================
# PDF EXPORT
# ==================================================

st.markdown("---")
st.subheader("📄 Download Executive Report")

def generate_pdf():

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("Career Intelligence Report", styles["Heading1"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(f"Role: {role}", styles["Normal"]))
    elements.append(Paragraph(f"Country: {selected_country}", styles["Normal"]))
    elements.append(Paragraph(f"Opportunity Score: {row['Opportunity']}", styles["Normal"]))
    elements.append(Paragraph(f"Interview Probability: {row['Probability']}", styles["Normal"]))
    elements.append(Paragraph(f"Market Percentile: {row['Market Percentile']}", styles["Normal"]))

    doc.build(elements)
    buffer.seek(0)
    return buffer

pdf_file = generate_pdf()

st.download_button(
    label="Download Career Report",
    data=pdf_file,
    file_name="career_report.pdf",
    mime="application/pdf"
)
