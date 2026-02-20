import streamlit as st
import pandas as pd
import numpy as np
import pdfplumber
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Career Intelligence", layout="wide")

st.title("🚀 AI Career Intelligence Engine")
st.markdown("Global AI-powered career growth system")

# -----------------------------
# LOAD MODEL
# -----------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -----------------------------
# LOAD DATA
# -----------------------------

@st.cache_data
def load_jobs():
    return pd.read_csv("jobs_dataset.csv")

jobs_df = load_jobs()

# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.header("📄 Resume Upload")
uploaded_file = st.sidebar.file_uploader("Upload Resume (PDF)", type=["pdf"])

st.sidebar.header("🌍 Home Country")
home_country = st.sidebar.selectbox(
    "Select Your Country",
    sorted(jobs_df["Country"].unique())
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

    st.sidebar.success("Resume processed")

else:
    st.warning("Upload resume to start")

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

        skill_overlap = len(
            set(extracted_skills) &
            set(row["Skills"].split(","))
        )

        skill_score = skill_overlap / len(row["Skills"].split(","))

        exp_score = (
            1 if experience >= row["Experience"]
            else experience / row["Experience"]
        )

        match_score = 0.5 * similarity + 0.3 * skill_score + 0.2 * exp_score

        probability = (
            match_score *
            (1 - row["Competition"]) *
            row["Demand"] *
            row["Visa"]
        )

        market_difficulty = (
            row["Competition"] * 0.6 +
            (1 - row["Demand"]) * 0.4
        ) * 100

        results.append({
            "Role": row["Role"],
            "Country": row["Country"],
            "Match %": round(match_score * 100, 2),
            "Estimated Interview Probability": round(probability * 100, 2),
            "Avg Salary ($)": row["AvgSalary"],
            "Market Difficulty %": round(market_difficulty, 2)
        })

    results_df = pd.DataFrame(results)

    # Home country boost
    results_df["HomeBoost"] = results_df["Country"].apply(
        lambda x: 1.2 if x == home_country else 1
    )

    results_df["AdjustedScore"] = (
        results_df["Estimated Interview Probability"] *
        results_df["HomeBoost"]
    )

    results_df = results_df.sort_values(
        by="AdjustedScore",
        ascending=False
    ).head(3)

    # -----------------------------
    # DASHBOARD
    # -----------------------------

    st.subheader("👤 Profile Intelligence")
    st.write("Skills:", extracted_skills)
    st.write("Experience:", experience, "years")

    st.markdown("---")

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
        ["🎯 Matches", "📈 Salary & Market", "🧠 Improvement Roadmap", "📊 Charts"]
    )

    # -----------------------------
    # TAB 1
    # -----------------------------

    with tab1:
        st.dataframe(
            results_df[
                ["Role","Country","Match %","Estimated Interview Probability"]
            ],
            use_container_width=True
        )

    # -----------------------------
    # TAB 2
    # -----------------------------

    with tab2:
        st.dataframe(
            results_df[
                ["Role","Country","Avg Salary ($)","Market Difficulty %"]
            ],
            use_container_width=True
        )

    # -----------------------------
    # TAB 3
    # -----------------------------

    with tab3:

        top_job = results_df.iloc[0]["Role"]
        selected_row = jobs_df[jobs_df["Role"] == top_job].iloc[0]

        missing = set(selected_row["Skills"].split(",")) - set(extracted_skills)

        st.subheader("Next Best Action Plan")
        st.write("Target Role:", top_job)
        st.write("Add Skills:", list(missing))

        if experience < selected_row["Experience"]:
            st.write(
                f"Gain {selected_row['Experience'] - experience} more years relevant exposure"
            )

        st.write("Focus market:", results_df.iloc[0]["Country"])

    # -----------------------------
    # TAB 4
    # -----------------------------

    with tab4:

        fig, ax = plt.subplots()
        ax.bar(results_df["Role"], results_df["Estimated Interview Probability"])
        ax.set_ylabel("Interview Probability (%)")
        ax.set_xticklabels(results_df["Role"], rotation=45)
        st.pyplot(fig)
