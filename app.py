import streamlit as st
import pandas as pd
import numpy as np
import pdfplumber
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pycountry

st.set_page_config(page_title="AI Career Intelligence Engine", layout="wide")

st.title("🚀 AI Career Intelligence Engine")
st.markdown("AI-powered global career growth intelligence platform")

# -----------------------------
# LOAD MODEL
# -----------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -----------------------------
# LOAD DATASET
# -----------------------------

@st.cache_data
def load_jobs():
    return pd.read_csv("jobs_dataset.csv")

jobs_df = load_jobs()

# -----------------------------
# GLOBAL COUNTRY LIST
# -----------------------------

all_countries = sorted([country.name for country in pycountry.countries])

st.sidebar.header("📄 Upload Resume")
uploaded_file = st.sidebar.file_uploader("Upload Resume (PDF)", type=["pdf"])

st.sidebar.header("🌍 Your Current Country")
home_country = st.sidebar.selectbox(
    "Select Country",
    all_countries,
    index=all_countries.index("India") if "India" in all_countries else 0
)

resume_text = ""
extracted_skills = []
experience = 0

# -----------------------------
# RESUME PROCESSING
# -----------------------------

if uploaded_file:
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            resume_text += page.extract_text()

    skill_db = list(set(",".join(jobs_df["Skills"]).split(",")))

    extracted_skills = [
        skill.strip()
        for skill in skill_db
        if skill.lower() in resume_text.lower()
    ]

    exp_match = re.search(r'(\d+)\+?\s*years', resume_text.lower())
    experience = int(exp_match.group(1)) if exp_match else 0

    st.sidebar.success("Resume processed successfully")

else:
    st.warning("Upload resume to begin")

# -----------------------------
# MATCHING ENGINE
# -----------------------------

if uploaded_file:

    resume_embedding = model.encode(resume_text)

    results = []

    for _, row in jobs_df.iterrows():

        job_text = row["Role"] + " " + row["Skills"]
        job_embedding = model.encode(job_text)

        similarity = cosine_similarity(
            [resume_embedding],
            [job_embedding]
        )[0][0]

        required_skills = row["Skills"].split(",")

        skill_overlap = len(set(extracted_skills) & set(required_skills))
        skill_score = skill_overlap / len(required_skills)

        exp_score = (
            1 if experience >= row["Experience"]
            else experience / row["Experience"]
        )

        match_score = 0.5 * similarity + 0.3 * skill_score + 0.2 * exp_score

        # Market + Visa Modeling
        probability = (
            match_score *
            (1 - row["Competition"]) *
            row["Demand"] *
            row["Visa"]
        )

        # Market Difficulty Index
        market_difficulty = (
            row["Competition"] * 0.6 +
            (1 - row["Demand"]) * 0.4
        ) * 100

        # Home country boost
        home_boost = 1.2 if row["Country"] == home_country else 1

        adjusted_probability = probability * home_boost

        results.append({
            "Role": row["Role"],
            "Country": row["Country"],
            "Match %": round(match_score * 100, 2),
            "Estimated Interview Probability": round(adjusted_probability * 100, 2),
            "Avg Salary ($)": row["AvgSalary"],
            "Market Difficulty %": round(market_difficulty, 2)
        })

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="Estimated Interview Probability",
        ascending=False
    ).head(3)

    # -----------------------------
    # DASHBOARD
    # -----------------------------

    st.subheader("👤 Profile Intelligence")
    st.write("Skills Detected:", extracted_skills)
    st.write("Estimated Experience:", experience, "years")

    st.markdown("---")

    st.subheader("🏆 Top 3 Strategic Career Options")

    col1, col2, col3 = st.columns(3)

    for i in range(len(results_df)):
        col = [col1, col2, col3][i]
        col.metric(
            results_df.iloc[i]["Role"],
            f"{results_df.iloc[i]['Estimated Interview Probability']:.2f}%",
            results_df.iloc[i]["Country"]
        )

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🎯 Matches",
         "💰 Salary & Market",
         "🧠 Improvement Roadmap",
         "📊 Visual Intelligence"]
    )

    # -----------------------------
    # TAB 1 — MATCHES
    # -----------------------------

    with tab1:
        st.dataframe(
            results_df[
                ["Role","Country","Match %","Estimated Interview Probability"]
            ],
            use_container_width=True
        )

    # -----------------------------
    # TAB 2 — SALARY + MARKET
    # -----------------------------

    with tab2:
        st.dataframe(
            results_df[
                ["Role","Country","Avg Salary ($)","Market Difficulty %"]
            ],
            use_container_width=True
        )

    # -----------------------------
    # TAB 3 — ROADMAP
    # -----------------------------

    with tab3:

        top_job = results_df.iloc[0]["Role"]
        selected_row = jobs_df[jobs_df["Role"] == top_job].iloc[0]

        required_skills = selected_row["Skills"].split(",")
        missing_skills = list(set(required_skills) - set(extracted_skills))

        st.subheader("🚀 Next Best Action Plan")

        st.write("🎯 Target Role:", top_job)
        st.write("🌍 Recommended Country:", results_df.iloc[0]["Country"])

        if missing_skills:
            st.write("📌 Skills to Acquire:", missing_skills)
        else:
            st.write("✅ Skill requirement met")

        if experience < selected_row["Experience"]:
            gap = selected_row["Experience"] - experience
            st.write(f"📈 Gain approximately {gap} more years relevant experience")

    # -----------------------------
    # TAB 4 — VISUAL INTELLIGENCE
    # -----------------------------

    with tab4:

        fig1, ax1 = plt.subplots()
        ax1.bar(results_df["Role"], results_df["Estimated Interview Probability"])
        ax1.set_ylabel("Interview Probability (%)")
        ax1.set_xticklabels(results_df["Role"], rotation=45)
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        ax2.bar(results_df["Role"], results_df["Market Difficulty %"])
        ax2.set_ylabel("Market Difficulty (%)")
        ax2.set_xticklabels(results_df["Role"], rotation=45)
        st.pyplot(fig2)
