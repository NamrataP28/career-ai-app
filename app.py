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
st.title("🚀 AI Career Intelligence Engine")

# -----------------------------------
# INITIALIZE
# -----------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")
parser = ResumeParser()
visa_model = VisaModel()
gpt_service = GPTService()

# -----------------------------------
# STEP 1: USER INTENT COLLECTION
# -----------------------------------

st.header("Step 1: Tell Us About Your Goals")

career_goal = st.selectbox(
    "What is your primary goal?",
    ["Salary Growth", "Global Mobility", "Leadership Growth", "Stability"]
)

career_direction = st.selectbox(
    "What are you looking for?",
    ["Stay in same domain", "Switch domain", "Explore opportunities"]
)

preferred_roles = st.multiselect(
    "Select roles you are interested in",
    [
        "Data Analyst","Business Analyst","Product Manager",
        "Marketing Manager","Financial Analyst",
        "Strategy Consultant","Operations Manager",
        "Software Engineer","Machine Learning Engineer"
    ]
)

preferred_countries = st.multiselect(
    "Preferred Countries",
    ["India","USA","UK","Canada","Germany","Singapore","Australia"]
)

current_salary = st.number_input("Your current annual salary (USD)", min_value=0)

if not preferred_roles:
    st.warning("Select at least one role of interest.")
    st.stop()

# -----------------------------------
# STEP 2: UPLOAD RESUME
# -----------------------------------

st.header("Step 2: Upload Resume")

file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if not file:
    st.stop()

resume_text = parser.extract_text(file)
resume_embedding = model.encode(resume_text)

# -----------------------------------
# MARKET SIMULATION ENGINE
# -----------------------------------

def generate_market_data():

    rows = []

    for role in preferred_roles:
        for country in preferred_countries:

            demand = np.random.randint(100, 800)
            salary = np.random.randint(60000, 220000)

            rows.append({
                "Role": role,
                "Country": country,
                "Market Demand": demand,
                "Avg Salary": salary
            })

    return pd.DataFrame(rows)

market_df = generate_market_data()

# -----------------------------------
# SCORING ENGINE
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

    interest_score = 1.0  # because user selected these roles

    visa_score = visa_model.visa_score(row["Country"])
    demand_norm = row["Market Demand"] / max_demand
    salary_norm = row["Avg Salary"] / max_salary

    # Salary growth potential
    if current_salary > 0:
        salary_growth = max(row["Avg Salary"] - current_salary, 0) / max_salary
    else:
        salary_growth = 0.5

    # Goal-based adjustment
    if career_goal == "Salary Growth":
        goal_boost = salary_growth
    elif career_goal == "Global Mobility":
        goal_boost = visa_score
    elif career_goal == "Leadership Growth":
        goal_boost = similarity
    else:
        goal_boost = demand_norm

    final_score = (
        0.35 * similarity +
        0.20 * interest_score +
        0.15 * salary_growth +
        0.15 * demand_norm +
        0.10 * visa_score +
        0.05 * goal_boost
    )

    results.append({
        "Role": row["Role"],
        "Country": row["Country"],
        "Match %": round(similarity * 100, 2),
        "Salary Growth %": round(salary_growth * 100, 2),
        "Demand %": round(demand_norm * 100, 2),
        "Visa %": round(visa_score * 100, 2),
        "Opportunity Score %": round(final_score * 100, 2),
        "Avg Salary": row["Avg Salary"]
    })

ranked_df = pd.DataFrame(results).sort_values(
    "Opportunity Score %",
    ascending=False
)

# -----------------------------------
# DISPLAY RESULTS
# -----------------------------------

st.header("Step 3: Your Market Position")

st.dataframe(ranked_df, use_container_width=True)

# -----------------------------------
# VISUALIZATION
# -----------------------------------

fig = px.bar(
    ranked_df,
    x="Opportunity Score %",
    y="Role",
    color="Country",
    orientation="h",
    hover_data=["Match %","Salary Growth %","Demand %","Visa %"],
    height=500
)

fig.update_layout(yaxis=dict(categoryorder="total ascending"))
st.plotly_chart(fig, use_container_width=True)

# -----------------------------------
# BENCHMARK POSITION
# -----------------------------------

st.subheader("📊 Market Benchmark")

avg_market_score = ranked_df["Opportunity Score %"].mean()

st.metric(
    "Your Average Global Opportunity Score",
    f"{round(avg_market_score,2)}%"
)

if avg_market_score > 70:
    st.success("You are strongly positioned in the market.")
elif avg_market_score > 50:
    st.info("You are moderately competitive.")
else:
    st.warning("Skill upgrade recommended for stronger positioning.")

# -----------------------------------
# GPT ROADMAP
# -----------------------------------

if st.button("Generate Personalized Roadmap"):

    top_role = ranked_df.iloc[0]["Role"]

    roadmap = gpt_service.generate_roadmap(
        resume_text,
        top_role
    )

    st.markdown(roadmap)

# -----------------------------------
# MOTIVATIONAL BLOCK
# -----------------------------------

st.markdown("---")
quotes = [
    "You have the right to perform your duty, but not to the fruits of your actions. — Bhagavad Gita",
    "Skill compounds. Keep building.",
    "Rejections are redirections.",
    "Growth happens outside comfort zones."
]

st.success(np.random.choice(quotes))
