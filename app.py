import streamlit as st
import pandas as pd
import pdfplumber
import re
import random
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pycountry

# ----------------------------------
# CONFIG
# ----------------------------------
st.set_page_config(page_title="AI Career Intelligence", layout="wide")

st.markdown("""
<style>
.metric-box {
    background-color: #f7f9fc;
    padding: 20px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------
# SESSION STATE
# ----------------------------------
if "page" not in st.session_state:
    st.session_state.page = 1
if "resume_text" not in st.session_state:
    st.session_state.resume_text = None
if "home_country" not in st.session_state:
    st.session_state.home_country = None

# ----------------------------------
# LOAD
# ----------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def load_jobs():
    return pd.read_csv("jobs_dataset.csv")

model = load_model()
jobs_df = load_jobs()

# =====================================================
# PAGE 1 — LOGIN
# =====================================================
if st.session_state.page == 1:

    st.title("🚀 AI Career Intelligence")
    st.subheader("Strategic Career Positioning Engine")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Enter Platform"):
        if username and password:
            st.session_state.page = 2
            st.rerun()
        else:
            st.error("Please enter credentials")

# =====================================================
# PAGE 2 — UPLOAD
# =====================================================
elif st.session_state.page == 2:

    st.title("Upload Resume")

    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    countries = sorted([c.name for c in pycountry.countries])
    home_country = st.selectbox("Select Your Base Country", countries)

    if uploaded_file:

        resume_text = ""
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                resume_text += page.extract_text()

        st.session_state.resume_text = resume_text
        st.session_state.home_country = home_country

        st.success("Resume successfully processed.")

        if st.button("Run Market Analysis"):
            st.session_state.page = 3
            st.rerun()

# =====================================================
# PAGE 3 — DASHBOARD
# =====================================================
elif st.session_state.page == 3:

    st.title("AI Career Intelligence Dashboard")

    resume_text = st.session_state.resume_text

    # ----------------------------------
    # Skill Extraction
    # ----------------------------------
    skill_db = list(set(",".join(jobs_df["Skills"]).split(",")))
    extracted_skills = [
        skill.strip()
        for skill in skill_db
        if skill.lower() in resume_text.lower()
    ]

    exp_match = re.search(r'(\d+)\+?\s*years', resume_text.lower())
    experience = int(exp_match.group(1)) if exp_match else 0

    resume_embedding = model.encode(resume_text)

    results = []

    max_salary = jobs_df["AvgSalary"].max()

    for _, row in jobs_df.iterrows():

        job_text = row["Role"] + " " + row["Skills"]
        job_embedding = model.encode(job_text)

        similarity = cosine_similarity(
            [resume_embedding], [job_embedding]
        )[0][0]

        required_skills = row["Skills"].split(",")
        skill_overlap = len(set(extracted_skills) & set(required_skills))
        skill_score = skill_overlap / max(len(required_skills), 1)

        exp_score = min(experience / max(row["Experience"], 1), 1)

        match_score = (
            0.5 * similarity +
            0.3 * skill_score +
            0.2 * exp_score
        )

        salary_norm = row["AvgSalary"] / max_salary

        interview_probability = (
            0.5 * match_score +
            0.2 * row["Demand"] +
            0.2 * (1 - row["Competition"]) +
            0.1 * row["Visa"]
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

    # ============================================
    # TABS
    # ============================================
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🌍 Global Market",
         "📊 Competitive Intelligence",
         "🤖 AI Advisor",
         "💡 Daily Boost"]
    )

    # ============================================
    # TAB 1 — GLOBAL MARKET
    # ============================================
    with tab1:

        market_options = ["Worldwide"] + sorted(df["Country"].unique())
        selected_market = st.selectbox("Select Market", market_options)

        if selected_market != "Worldwide":
            filtered_df = df[df["Country"] == selected_market]
        else:
            filtered_df = df

        top5 = filtered_df.sort_values(
            "Interview Probability",
            ascending=False
        ).head(5)

        st.dataframe(top5, use_container_width=True)

    # ============================================
    # TAB 2 — COMPETITIVE INTELLIGENCE
    # ============================================
    with tab2:

        st.subheader("Market Positioning Analysis")

        top3 = df.sort_values(
            "Interview Probability",
            ascending=False
        ).head(3)

        fig, ax = plt.subplots(figsize=(7,3))

        x = range(len(top3))

        width = 0.2

        bars1 = ax.bar(x, top3["Demand"], width=width)
        bars2 = ax.bar([i + width for i in x], top3["Competition"], width=width)
        bars3 = ax.bar([i + width*2 for i in x], top3["Match %"], width=width)

        ax.set_xticks([i + width for i in x])
        ax.set_xticklabels(top3["Role"], rotation=15)
        ax.set_ylim(0,100)

        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    height + 1,
                    f"{height:.2f}%",
                    ha='center',
                    fontsize=8
                )

        st.pyplot(fig)

        st.markdown("### Executive Insight")

        for _, row in top3.iterrows():
            st.markdown(f"**{row['Role']} ({row['Country']})**")
            gap = row["Demand"] - row["Competition"]

            if gap > 10:
                st.success("Favorable structural demand.")
            elif gap > 0:
                st.info("Balanced opportunity.")
            else:
                st.warning("Competitive pressure high.")

            st.markdown("---")

    # ============================================
    # TAB 3 — AI ADVISOR
    # ============================================
    with tab3:

        question = st.text_input("Ask about skill improvement or certifications")

        if question:

            top_role = df.sort_values(
                "Interview Probability",
                ascending=False
            ).iloc[0]

            job_row = jobs_df[jobs_df["Role"] == top_role["Role"]].iloc[0]
            required_skills = job_row["Skills"].split(",")

            missing_skills = [
                skill.strip()
                for skill in required_skills
                if skill.strip().lower() not in resume_text.lower()
            ]

            st.markdown("### Strategic Recommendation")
            st.write(f"To improve your candidacy for **{top_role['Role']}**:")
            st.write(", ".join(missing_skills[:5]))

            st.write("Suggested Actions:")
            st.write("- Build measurable projects")
            st.write("- Complete role-relevant certification")
            st.write("- Quantify resume impact metrics")

    # ============================================
    # TAB 4 — DAILY BOOST
    # ============================================
    with tab4:

        quotes = [
            "Rejection is redirection.",
            "Your skills compound daily.",
            "Momentum beats motivation.",
            "Focus on controllables."
        ]

        st.subheader("Daily Strength Reminder")
        st.success(random.choice(quotes))

        st.info("Apply intentionally, not emotionally.")
