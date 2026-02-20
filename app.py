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
st.set_page_config(page_title="AI Career Intelligence Engine", layout="wide")

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
# LOAD MODEL
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

    st.title("🔐 AI Career Intelligence Engine")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username and password:
            st.session_state.page = 2
            st.rerun()
        else:
            st.error("Enter valid credentials")

# =====================================================
# PAGE 2 — UPLOAD
# =====================================================
elif st.session_state.page == 2:

    st.title("📄 Upload Resume")

    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

    countries = sorted([c.name for c in pycountry.countries])
    home_country = st.selectbox("Select Your Country", countries)

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

# =====================================================
# PAGE 3 — FULL ANALYSIS
# =====================================================
elif st.session_state.page == 3:

    st.title("🚀 AI Career Intelligence Dashboard")

    resume_text = st.session_state.resume_text
    home_country = st.session_state.home_country

    # ----------------------------------
    # Extract skills
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

    # ----------------------------------
    # MATCHING ENGINE
    # ----------------------------------
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

        market_opportunity = (row["Demand"] * row["AvgSalary"]) / competition_safe

        interview_probability = min(
            (0.6 * match_score +
             0.15 * (1 - row["Competition"]) +
             0.15 * row["Demand"] +
             0.1 * row["Visa"]) * 100,
            100
        )

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
        ["🌍 Global Market View",
         "📊 Competitive Intelligence",
         "🤖 AI Career Advisor",
         "💡 Daily Boost"]
    )

    # ============================================
    # TAB 1 — GLOBAL MARKET VIEW
    # ============================================
    with tab1:

        st.subheader("Top 5 Role-Country Opportunities")

        top5 = df.sort_values(
            "Interview Probability",
            ascending=False
        ).head(5)

        st.dataframe(top5, use_container_width=True)

    # ============================================
    # TAB 2 — COMPETITIVE INTELLIGENCE
    # ============================================
    with tab2:

        st.subheader("Demand vs Competition vs Your Strength")

        top3 = df.sort_values(
            "Interview Probability",
            ascending=False
        ).head(3)

        fig, ax = plt.subplots(figsize=(8, 4))
        x = range(len(top3))

        bars1 = ax.bar(x, top3["Demand"], width=0.25, label="Demand")
        bars2 = ax.bar([i + 0.25 for i in x], top3["Competition"], width=0.25, label="Competition")
        bars3 = ax.bar([i + 0.50 for i in x], top3["Match %"], width=0.25, label="Your Strength")

        ax.set_xticks([i + 0.25 for i in x])
        ax.set_xticklabels(top3["Role"], rotation=20)
        ax.set_ylim(0, 100)
        ax.legend()

        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    height + 1,
                    f'{height:.2f}%',
                    ha='center'
                )

        st.pyplot(fig)

        st.markdown("### Economic Opportunity Insight")
        for _, row in top3.iterrows():
            st.write(f"**{row['Role']} ({row['Country']})**")
            st.write(
                f"Demand: {row['Demand']:.2f}% | "
                f"Competition: {row['Competition']:.2f}% | "
                f"Interview Probability: {row['Interview Probability']:.2f}%"
            )
            st.write("---")

    # ============================================
    # TAB 3 — AI CAREER ADVISOR
    # ============================================
    with tab3:

        st.subheader("Ask the AI Career Advisor")

        question = st.text_input("Ask about skills, certifications, improvements...")

        if question:

            missing_skills = []

            top_role = df.sort_values(
                "Interview Probability",
                ascending=False
            ).iloc[0]

            job_row = jobs_df[jobs_df["Role"] == top_role["Role"]].iloc[0]
            required_skills = job_row["Skills"].split(",")

            for skill in required_skills:
                if skill.strip().lower() not in resume_text.lower():
                    missing_skills.append(skill.strip())

            st.write("### AI Guidance")
            st.write(
                f"To improve for **{top_role['Role']}**, consider strengthening:"
            )
            st.write(", ".join(missing_skills[:5]))

            st.write("Suggested Strategy:")
            st.write("- Complete role-aligned certification")
            st.write("- Build 2 real-world portfolio projects")
            st.write("- Improve measurable impact metrics in resume")

    # ============================================
    # TAB 4 — DAILY BOOST
    # ============================================
    with tab4:

        quotes = [
            "Rejection is redirection.",
            "Every no brings you closer to the right yes.",
            "Your skills compound daily.",
            "Focus on controllables.",
            "Small improvements daily = massive career gains."
        ]

        st.subheader("💬 Today's Motivation")
        st.success(random.choice(quotes))

        st.subheader("Mini Challenge")
        st.info("Apply to 3 quality roles today — not 30 random ones.")
