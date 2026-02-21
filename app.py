import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from services.resume_parser import ResumeParser
from services.visa_model import VisaModel
from services.gpt_service import GPTService

st.set_page_config(layout="wide")
st.title("🚀 Career Intelligence Platform")

# -----------------------------------
# INITIALIZE
# -----------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")
parser = ResumeParser()
visa_model = VisaModel()
gpt_service = GPTService()

# -----------------------------------
# USER INPUT SECTION (Executive Style)
# -----------------------------------

col1, col2, col3 = st.columns(3)

career_goal = col1.selectbox(
    "Primary Goal",
    ["Salary Growth", "Global Mobility", "Leadership Growth", "Stability"]
)

target_role = col2.selectbox(
    "Target Role",
    ["Data Analyst","Business Analyst","Product Manager",
     "Marketing Manager","Financial Analyst",
     "Strategy Consultant","Software Engineer"]
)

target_country = col3.selectbox(
    "Target Country",
    ["India","USA","UK","Canada","Germany","Singapore","Australia"]
)

current_salary = st.number_input("Current Salary (USD)", min_value=0)

file = st.file_uploader("Upload Resume", type=["pdf"])
if not file:
    st.stop()

resume_text = parser.extract_text(file)
resume_embedding = model.encode(resume_text)

# -----------------------------------
# MARKET SIMULATION ENGINE
# -----------------------------------

market_demand = np.random.randint(300, 1200)
avg_salary = np.random.randint(80000, 220000)
competition_index = np.random.uniform(0.3, 0.9)

role_embedding = model.encode(target_role)
similarity = cosine_similarity([resume_embedding],[role_embedding])[0][0]

visa_score = visa_model.visa_score(target_country)

salary_growth = (avg_salary - current_salary)/avg_salary if current_salary else 0.5
transition_risk = 1 - similarity

interview_probability = (
    0.4*similarity +
    0.2*salary_growth +
    0.2*(1-competition_index) +
    0.2*visa_score
)

opportunity_score = (
    0.35*similarity +
    0.20*salary_growth +
    0.15*(1-competition_index) +
    0.15*visa_score +
    0.15*(market_demand/1200)
)

# -----------------------------------
# EXECUTIVE SCORECARD (6figr style)
# -----------------------------------

st.markdown("## 📊 Executive Career Scorecard")

k1, k2, k3, k4 = st.columns(4)

k1.metric("Opportunity Score", f"{round(opportunity_score*100,2)}%")
k2.metric("Interview Probability", f"{round(interview_probability*100,2)}%")
k3.metric("Salary Growth Potential", f"{round(salary_growth*100,2)}%")
k4.metric("Market Demand Index", f"{round((market_demand/1200)*100,2)}%")

# -----------------------------------
# SKILL RADAR CHART
# -----------------------------------

st.markdown("## 🧠 Skill Coverage Radar")

skill_categories = ["Technical","Strategy","Communication","Leadership","Domain"]
skill_values = [
    similarity*100,
    np.random.randint(50,90),
    np.random.randint(40,85),
    np.random.randint(30,80),
    np.random.randint(50,95)
]

fig = go.Figure()
fig.add_trace(go.Scatterpolar(
    r=skill_values,
    theta=skill_categories,
    fill='toself'
))

fig.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0,100])),
    showlegend=False,
    height=450
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------------
# WHAT-IF SIMULATOR
# -----------------------------------

st.markdown("## 🎯 What-If Skill Simulator")

skill_boost = st.slider("If you improve skill alignment by:", 0, 30, 10)

simulated_similarity = min(similarity + (skill_boost/100),1)

simulated_score = (
    0.35*simulated_similarity +
    0.20*salary_growth +
    0.15*(1-competition_index) +
    0.15*visa_score +
    0.15*(market_demand/1200)
)

st.success(f"New Opportunity Score: {round(simulated_score*100,2)}%")

# -----------------------------------
# MARKET BENCHMARK
# -----------------------------------

st.markdown("## 📈 Market Positioning")

candidate_pool = np.random.normal(0.6, 0.15, 1000)
percentile = (candidate_pool < opportunity_score).mean()*100

st.metric("Market Percentile Rank", f"{round(percentile,2)}%")

if percentile > 75:
    st.success("Top-tier competitive profile.")
elif percentile > 50:
    st.info("Above-average competitiveness.")
else:
    st.warning("Upskilling recommended.")

# -----------------------------------
# SALARY FORECAST
# -----------------------------------

st.markdown("## 💰 3-Year Salary Projection")

years = [1,2,3]
forecast = [avg_salary*(1.1**y) for y in years]

fig2 = px.line(
    x=years,
    y=forecast,
    labels={"x":"Years","y":"Projected Salary"},
    markers=True
)

st.plotly_chart(fig2, use_container_width=True)

# -----------------------------------
# AI ROADMAP
# -----------------------------------

st.markdown("## 🤖 AI Growth Strategy")

if st.button("Generate Strategic Roadmap"):

    roadmap = gpt_service.generate_roadmap(
        resume_text,
        target_role
    )

    st.markdown(roadmap)

# -----------------------------------
# DAILY MOTIVATION
# -----------------------------------

st.markdown("## 🧘 Daily Strength")

quotes = [
    "You have the right to perform your duty, but not to the fruits of your actions.",
    "Skill compounds. Effort compounds faster.",
    "Rejections are redirections.",
    "Consistency beats intensity."
]

st.success(np.random.choice(quotes))
