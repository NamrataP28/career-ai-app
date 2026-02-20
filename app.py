import streamlit as st
import pandas as pd
import pdfplumber
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import pycountry

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(page_title="AI Career Intelligence Engine", layout="wide")

# ----------------------------------
# SESSION STATE INITIALIZATION
# ----------------------------------
if "page" not in st.session_state:
    st.session_state.page = 1

if "user" not in st.session_state:
    st.session_state.user = None

if "resume_text" not in st.session_state:
    st.session_state.resume_text = None

if "home_country" not in st.session_state:
    st.session_state.home_country = None

# ----------------------------------
# LOAD MODEL & DATA
# ----------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def load_jobs():
    return pd.read_csv("jobs_dataset.csv")

model = load_model()
jobs_df = load_jobs()

# ==========================================================
# PAGE 1 — LOGIN
# ==========================================================
if st.session_state.page == 1:

    st.title("🔐 AI Career Intelligence Engine")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username and password:
            st.session_state.user = username
            st.session_state.page = 2
            st.rerun()
        else:
            st.error("Please enter credentials")

# ==========================================================
# PAGE 2 — RESUME UPLOAD
# ==========================================================
elif st.session_state.page == 2:

    st.title("📄 Upload Resume")

    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

    all_countries = sorted([c.name for c in pycountry.countries])
    home_country = st.selectbox("Select Your Current Country", all_countries)

    if uploaded_file:

        resume_text = ""
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                resume_text += page.extract_text()

        st.session_state.resume_text = resume_text
        st.session_state.home_country = home_country

        st.success("Resume uploaded successfully")

        if st.button("Proceed to Analysis"):
            st.session_state.page = 3
            st.rerun()

    if st.button("⬅ Back to Login"):
        st.session_state.page = 1
        st.rerun()

# ==========================================================
# PAGE 3 — ANALYSIS
# ==========================================================
elif st.session_state.page == 3:

    st.title("🚀 Career Analysis Dashboard")

    resume_text = st.session_state.resume_text
    home_country = st.session_state.home_country

    if resume_text is None:
        st.error("Resume not found. Please re-upload.")
        st.session_state.page = 2
        st.rerun()

    # ----------------------------------
    # SKILL EXTRACTION
    # ----------------------------------
    skill_db = list(set(",".join(jobs_df["Skills"]).split(",")))
    extracted_skills = [
        skill.strip()
        for skill in skill_db
        if skill.lower() in resume_text.lower()
    ]

    exp_match = re.search(r'(\d+)\+?\s*years', resume_text.lower())
    experience = int(exp_match.group(1)) if exp_match else 0

    # ----------------------------------
    # MATCHING ENGINE
    # ----------------------------------
    resume_embedding = model.encode(resume_text)
    results = []

    for _, row in jobs_df.iterrows():

        job_text = row["Role"] + " " + row["Skills"]
        job_embedding = model.encode(job_text)

        similarity = cosine_similarity(
            [resume_embedding], [job_embedding]
        )[0][0]

        required_skills = row["Skills"].split(",")
        skill_overlap = len(set(extracted_skills) & set(required_skills))
        skill_score = skill_overlap / max(len(required_skills), 1)

        exp_score = 1 if experience >= row["Experience"] else experience / max(row["Experience"], 1)

        match_score = 0.5 * similarity + 0.3 * skill_score + 0.2 * exp_score

        competition_safe = max(row["Competition"], 0.01)

        # ROI formula unchanged
        market_opportunity = (row["Demand"] * row["AvgSalary"]) / competition_safe

        home_boost = 1.15 if row["Country"] == home_country else 1

        interview_probability = min(
            (0.6 * match_score +
             0.15 * (1 - row["Competition"]) +
             0.15 * row["Demand"] +
             0.1 * row["Visa"]) * home_boost * 100,
            100
        )

        results.append({
            "Role": row["Role"],
            "Country": row["Country"],
            "Match %": round(match_score * 100, 2),
            "Interview Probability": round(interview_probability, 2),
            "Market Opportunity Index": round(market_opportunity, 2),
            "Demand": round(row["Demand"] * 100, 2),
            "Competition": round(row["Competition"] * 100, 2),
            "Avg Salary ($)": row["AvgSalary"]
        })

    df = pd.DataFrame(results)

    df["Percentile Rank"] = (
        df["Interview Probability"].rank(pct=True) * 100
    ).round(2)

    results_df = df.sort_values(
        by="Interview Probability",
        ascending=False
    ).head(3)

    # ----------------------------------
    # EXECUTIVE SUMMARY
    # ----------------------------------
    st.subheader("📊 Executive Summary")

    top = results_df.iloc[0]

    col1, col2, col3 = st.columns(3)

    col1.metric("Interview Probability", f"{top['Interview Probability']}%")
    col2.metric(
        "Market Opportunity Index",
        f"{int(top['Market Opportunity Index']):,}"
    )
    col3.metric("Percentile Rank", f"{top['Percentile Rank']:.0f}th")

    # ----------------------------------
    # INTERACTIVE CHART
    # ----------------------------------
    st.subheader("📈 Competitive Comparison")

    fig = go.Figure()

    for metric in ["Demand", "Competition", "Match %"]:
        fig.add_trace(go.Bar(
            x=results_df["Role"],
            y=results_df[metric],
            name=metric,
            text=[f"{val}%" for val in results_df[metric]],
            textposition="outside"
        ))

    fig.update_layout(
        barmode="group",
        yaxis=dict(title="Score (%)", range=[0, 100]),
        legend=dict(orientation="h", y=1.1),
        height=450
    )

    st.plotly_chart(fig, use_container_width=True)

    # ----------------------------------
    # ROI CHART
    # ----------------------------------
    st.subheader("💰 Economic Opportunity")

    fig2 = go.Figure()

    fig2.add_trace(go.Bar(
        x=results_df["Role"],
        y=results_df["Market Opportunity Index"],
        text=[f"{int(val):,}" for val in results_df["Market Opportunity Index"]],
        textposition="outside"
    ))

    fig2.update_layout(
        yaxis_title="Market Opportunity Index (Demand × Salary ÷ Competition)",
        height=450
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.info(
        "Market Opportunity Index = (Demand × Average Salary) ÷ Competition.\n\n"
        "Higher values indicate stronger economic upside adjusted for competitive intensity."
    )

    if st.button("⬅ Upload Another Resume"):
        st.session_state.page = 2
        st.rerun()
