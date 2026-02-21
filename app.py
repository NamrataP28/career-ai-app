import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from services.resume_parser import ResumeParser
from services.visa_model import VisaModel
from services.gpt_service import GPTService

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(layout="wide")
st.title("🚀 AI Career Intelligence Engine — Global")

# -----------------------------------
# INITIALIZE SERVICES
# -----------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")
parser = ResumeParser()
visa_model = VisaModel()
gpt_service = GPTService()

# -----------------------------------
# FILE UPLOAD
# -----------------------------------

file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
if not file:
    st.stop()

resume_text = parser.extract_text(file)
resume_embedding = model.encode(resume_text)

# -----------------------------------
# GLOBAL CORPORATE ROLE BANK
# -----------------------------------

role_bank = [
    # Analytics
    "Data Analyst","Business Analyst","Product Analyst",
    "Marketing Analyst","Financial Analyst","Risk Analyst",
    "FP&A Analyst","Analytics Manager",

    # Product
    "Product Manager","Growth Product Manager",
    "Product Operations Manager","Head of Product",

    # Marketing
    "Marketing Manager","Growth Manager",
    "Performance Marketing Manager",
    "Digital Marketing Manager",

    # Finance
    "Finance Manager","Investment Analyst",
    "Corporate Finance Manager","Strategy Finance Manager",

    # Consulting
    "Management Consultant","Strategy Consultant",
    "Transformation Consultant",

    # Operations
    "Operations Manager","Program Manager",
    "Supply Chain Manager",

    # Tech
    "Software Engineer","Backend Engineer",
    "Machine Learning Engineer","Data Engineer",
    "Engineering Manager",

    # Leadership
    "Director of Strategy","Head of Analytics",
    "Chief Product Officer"
]

countries = [
    "India",
    "USA",
    "UK",
    "Canada",
    "Germany",
    "Singapore",
    "Australia"
]

# -----------------------------------
# SIMULATED MARKET ENGINE (Stable)
# -----------------------------------

def generate_market_data():

    rows = []

    for role in role_bank:
        for country in countries:

            demand = np.random.randint(50, 500)
            salary = np.random.randint(60000, 200000)

            rows.append({
                "Role": role,
                "Country": country,
                "Market Demand": demand,
                "Avg Salary": salary
            })

    return pd.DataFrame(rows)

market_df = generate_market_data()

# -----------------------------------
# RESUME-FIRST SCORING
# -----------------------------------

results = []

max_demand = market_df["Market Demand"].max()
max_salary = market_df["Avg Salary"].max()

for _, row in market_df.iterrows():

    role_embedding = model.encode(row["Role"])

    similarity = cosine_similarity(
        [resume_embedding],
        [role_embedding]
    )[0][0]

    visa_score = visa_model.visa_score(row["Country"])
    demand_norm = row["Market Demand"] / max_demand
    salary_norm = row["Avg Salary"] / max_salary

    final_score = (
        0.55 * similarity +
        0.20 * demand_norm +
        0.15 * visa_score +
        0.10 * salary_norm
    )

    results.append({
        "Role": row["Role"],
        "Country": row["Country"],
        "Match %": round(similarity * 100, 2),
        "Visa Score %": round(visa_score * 100, 2),
        "Demand Index %": round(demand_norm * 100, 2),
        "Salary Index %": round(salary_norm * 100, 2),
        "Global Opportunity Score %": round(final_score * 100, 2)
    })

ranked_df = pd.DataFrame(results).sort_values(
    "Global Opportunity Score %",
    ascending=False
)

# -----------------------------------
# COUNTRY FILTER
# -----------------------------------

st.subheader("🌍 Global Opportunity Ranking")

country_filter = st.selectbox(
    "Filter by Country",
    ["Worldwide"] + countries
)

if country_filter == "Worldwide":
    display_df = ranked_df
else:
    display_df = ranked_df[ranked_df["Country"] == country_filter]

st.dataframe(display_df.head(20), use_container_width=True)

# -----------------------------------
# INTERACTIVE VISUAL
# -----------------------------------

fig = px.bar(
    display_df.head(10),
    x="Global Opportunity Score %",
    y="Role",
    color="Country",
    orientation="h",
    hover_data=[
        "Match %",
        "Visa Score %",
        "Demand Index %",
        "Salary Index %"
    ],
    height=500
)

fig.update_layout(
    yaxis=dict(categoryorder="total ascending")
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------------
# AI CAREER ROADMAP
# -----------------------------------

if st.button("Generate Career Roadmap for Top Role"):

    top_role = ranked_df.iloc[0]["Role"]

    roadmap = gpt_service.generate_roadmap(
        resume_text,
        top_role
    )

    st.markdown(roadmap)

# -----------------------------------
# MOTIVATION BLOCK
# -----------------------------------

st.markdown("---")
st.subheader("🧘 Daily Strength")

quotes = [
    "You have the right to perform your duty, but not to the fruits of your actions. — Bhagavad Gita",
    "Focus on progress, not perfection.",
    "Rejections are redirections.",
    "Skill compounds. Effort compounds faster.",
    "Your current level is not your ceiling."
]

st.success(np.random.choice(quotes))
