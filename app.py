import streamlit as st
import pandas as pd
import numpy as np
import pdfplumber
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Career Intelligence", layout="wide")

st.title("🚀 AI Career Intelligence Platform")
st.markdown("AI-powered global career probability engine")

# -----------------------------
# LOAD MODEL (Cached)
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
# RESUME UPLOAD
# -----------------------------

st.sidebar.header("📄 Upload Resume")
uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

resume_text = ""
extracted_skills = []
experience = 0

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
    st.warning("Upload resume to begin")

# -----------------------------
# AI SIMILARITY SCORING
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

        results.append({
            "Role": row["Role"],
            "Country": row["Country"],
            "Match %": round(match_score * 100, 2),
            "Hiring Probability %": round(probability * 100, 2)
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by="Hiring Probability %",
        ascending=False
    ).head(3)

    # -----------------------------
    # DASHBOARD OUTPUT
    # -----------------------------

    st.subheader("👤 Extracted Profile")
    st.write("Skills:", extracted_skills)
    st.write("Experience:", experience, "years")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    for i in range(3):
        col = [col1, col2, col3][i]
        col.metric(
            results_df.iloc[i]["Role"],
            f"{results_df.iloc[i]['Hiring Probability %']}%",
            results_df.iloc[i]["Country"]
        )

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(
        ["🎯 Top Matches", "📈 Resume Improvement", "📊 Probability Chart"]
    )

    # -----------------------------
    # TAB 1
    # -----------------------------

    with tab1:
        st.dataframe(results_df, use_container_width=True)

    # -----------------------------
    # TAB 2
    # -----------------------------

    with tab2:

        top_job = results_df.iloc[0]["Role"]
        selected_row = jobs_df[jobs_df["Role"] == top_job].iloc[0]

        missing = set(selected_row["Skills"].split(",")) - set(extracted_skills)

        st.subheader("Resume Improvement Suggestions")
        st.write("Top role:", top_job)
        st.write("Missing Skills to Improve:", list(missing))

        if experience < selected_row["Experience"]:
            st.write("Increase experience exposure to reach required level")

    # -----------------------------
    # TAB 3
    # -----------------------------

    with tab3:
        fig, ax = plt.subplots()
        ax.bar(results_df["Role"], results_df["Hiring Probability %"])
        ax.set_ylabel("Hiring Probability %")
        ax.set_xticklabels(results_df["Role"], rotation=45)
        st.pyplot(fig)
