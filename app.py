import streamlit as st
import pandas as pd
import pdfplumber
import re
import random
import plotly.express as px
import pycountry
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="AI Career Intelligence", layout="wide")

# -----------------------------
# SESSION STATE
# -----------------------------
if "page" not in st.session_state:
    st.session_state.page = 1
if "resume_text" not in st.session_state:
    st.session_state.resume_text = None

# -----------------------------
# LOAD
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def load_jobs():
    return pd.read_csv("jobs_dataset.csv")

model = load_model()
jobs_df = load_jobs()

# ======================================================
# PAGE 1 — LOGIN
# ======================================================
if st.session_state.page == 1:

    st.title("🚀 AI Career Intelligence Platform")
    st.subheader("Strategic Career Positioning Engine")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Enter Platform"):
        if username and password:
            st.session_state.page = 2
            st.rerun()

# ======================================================
# PAGE 2 — UPLOAD
# ======================================================
elif st.session_state.page == 2:

    st.title("Upload Resume")

    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

    if uploaded_file:
        resume_text = ""
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                resume_text += page.extract_text()

        st.session_state.resume_text = resume_text
        st.success("Resume processed successfully")

        if st.button("Run AI Market Analysis"):
            st.session_state.page = 3
            st.rerun()

# ======================================================
# PAGE 3 — DASHBOARD
# ======================================================
elif st.session_state.page == 3:

    st.title("AI Career Intelligence Dashboard")

    resume_text = st.session_state.resume_text

    # -----------------------------
    # MATCH ENGINE
    # -----------------------------
    resume_embedding = model.encode(resume_text)
    results = []

    max_salary = jobs_df["AvgSalary"].max()

    for _, row in jobs_df.iterrows():

        job_text = row["Role"] + " " + row["Skills"]
        job_embedding = model.encode(job_text)

        similarity = cosine_similarity(
            [resume_embedding], [job_embedding]
        )[0][0]

        match_score = similarity

        interview_probability = (
            0.5 * similarity +
            0.3 * row["Demand"] +
            0.2 * (1 - row["Competition"])
        ) * 100

        market_opportunity = (
            row["Demand"] * row["AvgSalary"]
        ) / max(row["Competition"], 0.01)

        results.append({
            "Role": row["Role"],
            "Country": row["Country"],
            "Match %": round(match_score * 100, 2),
            "Demand": round(row["Demand"] * 100, 2),
            "Competition": round(row["Competition"] * 100, 2),
            "Interview Probability": round(interview_probability, 2),
            "Market Opportunity Index": round(market_opportunity, 2)
        })

    df = pd.DataFrame(results)

    # -----------------------------
    # TABS
    # -----------------------------
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🌍 Global Market",
         "📊 Competitive Intelligence",
         "🧠 AI Agent",
         "🕉 Daily Boost"]
    )

    # ==================================================
    # TAB 1 — GLOBAL MARKET
    # ==================================================
    with tab1:

        all_countries = sorted(df["Country"].unique())
        selected_country = st.selectbox(
            "Select Country",
            ["Worldwide"] + all_countries
        )

        if selected_country != "Worldwide":
            filtered_df = df[df["Country"] == selected_country]
        else:
            filtered_df = df

        filtered_df = filtered_df.sort_values(
            "Interview Probability",
            ascending=False
        )

        st.dataframe(filtered_df.head(10), use_container_width=True)

    # ==================================================
    # TAB 2 — COMPETITIVE INTELLIGENCE
    # ==================================================
    with tab2:

        left, right = st.columns([1, 2])

        with left:
            country_filter = st.selectbox(
                "Filter by Country",
                ["Worldwide"] + all_countries,
                key="tab2_country"
            )

        if country_filter != "Worldwide":
            comp_df = df[df["Country"] == country_filter]
        else:
            comp_df = df

        top5 = comp_df.sort_values(
            "Interview Probability",
            ascending=False
        ).head(5)

        fig = px.bar(
            top5,
            x="Role",
            y=["Demand", "Competition", "Match %"],
            barmode="group",
            height=350
        )

        st.plotly_chart(fig, use_container_width=True)

    # ==================================================
    # TAB 3 — FLOATING AI AGENT
    # ==================================================
    with tab3:

        st.markdown("### 🧠 Smart Career Assistant")

        question = st.text_input("Ask anything about skills, certifications...")

        if question:
            st.info("AI Response:")
            st.write(
                "To improve your positioning, focus on high-demand skills, "
                "build measurable projects, and align certifications with target roles."
            )

    # ==================================================
    # TAB 4 — DAILY BOOST
    # ==================================================
    with tab4:

        gita_quotes = [
            "You have the right to perform your duty, but not to the fruits of your actions.",
            "Set thy heart upon thy work, but never on its reward.",
            "The soul is neither born, nor does it die."
        ]

        funny_quotes = [
            "Job searching is just speed dating with companies.",
            "If at first you don’t succeed, redefine success.",
            "LinkedIn stalking is research."
        ]

        st.success(random.choice(gita_quotes + funny_quotes))
