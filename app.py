import streamlit as st
import pandas as pd
import pdfplumber
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pycountry
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
import io

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(page_title="AI Career Intelligence Engine", layout="wide")
st.title("🚀 AI Career Intelligence Engine")

# ----------------------------------
# LOGIN SYSTEM
# ----------------------------------
if "user" not in st.session_state:
    st.session_state.user = None

st.sidebar.header("🔐 Login")
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")

if st.sidebar.button("Login"):
    if username and password:
        st.session_state.user = username
        st.sidebar.success(f"Welcome {username}")

if not st.session_state.user:
    st.stop()

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

# ----------------------------------
# SIDEBAR
# ----------------------------------
all_countries = sorted([c.name for c in pycountry.countries])

st.sidebar.header("📄 Upload Resume")
uploaded_file = st.sidebar.file_uploader("Upload Resume (PDF)", type=["pdf"])

st.sidebar.header("🌍 Your Current Country")
home_country = st.sidebar.selectbox(
    "Select Country",
    all_countries,
    index=all_countries.index("India") if "India" in all_countries else 0
)

if not uploaded_file:
    st.info("Upload resume to begin")
    st.stop()

# ----------------------------------
# RESUME PROCESSING
# ----------------------------------
resume_text = ""
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

# ----------------------------------
# MATCHING ENGINE
# ----------------------------------
resume_embedding = model.encode(resume_text)
results = []

max_salary = jobs_df["AvgSalary"].max()

for _, row in jobs_df.iterrows():

    job_text = row["Role"] + " " + row["Skills"]
    job_embedding = model.encode(job_text)

    similarity = cosine_similarity([resume_embedding], [job_embedding])[0][0]

    required_skills = row["Skills"].split(",")
    skill_overlap = len(set(extracted_skills) & set(required_skills))
    skill_score = skill_overlap / max(len(required_skills), 1)

    exp_score = 1 if experience >= row["Experience"] else experience / max(row["Experience"], 1)

    match_score = 0.5 * similarity + 0.3 * skill_score + 0.2 * exp_score

    competition_safe = max(row["Competition"], 0.01)

    market_roi = (row["Demand"] * row["AvgSalary"]) / competition_safe
    effort_reward = competition_safe / max((row["Demand"] * match_score), 0.01)

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
        "Market ROI Score": round(market_roi, 2),
        "Effort-to-Reward Ratio": round(effort_reward, 2),
        "Demand": round(row["Demand"] * 100, 2),
        "Competition": round(row["Competition"] * 100, 2),
        "Visa": round(row["Visa"] * 100, 2),
        "Avg Salary ($)": row["AvgSalary"],
        "Projects": row.get("Projects", "")
    })

full_results_df = pd.DataFrame(results)

# Percentile Ranking based on Interview Probability
full_results_df["Percentile Rank"] = (
    full_results_df["Interview Probability"].rank(pct=True) * 100
).round(2)

# Top 3 Roles
results_df = full_results_df.sort_values(
    by="Interview Probability",
    ascending=False
).head(3)
# ----------------------------------
# EXECUTIVE STRATEGIC SCORECARD
# ----------------------------------
st.markdown("---")
st.subheader("📊 Executive Summary")

top = results_df.iloc[0]

col1, col2, col3, col4 = st.columns(4)

col1.metric("Interview Probability",
            f"{top['Interview Probability']:.1f}%")

col2.metric("Market ROI",
            f"{top['Market ROI Score']:.0f}")

col3.metric("Effort / Reward",
            f"{top['Effort-to-Reward Ratio']:.2f}")

col4.metric("Percentile Rank",
            f"{top['Percentile Rank']:.0f}th")

# ----------------------------------
# TABS
# ----------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Dashboard",
     "📈 Competitive Analysis",
     "🌍 Country Heatmap",
     "🎯 Growth Simulator",
     "🧠 Performance"]
)

# ----------------------------------
# TAB 1 — DASHBOARD
# ----------------------------------
with tab1:
    st.dataframe(results_df[[
        "Role","Country",
        "Match %",
        "Interview Probability",
        "Market ROI Score",
        "Effort-to-Reward Ratio",
        "Percentile Rank",
        "Avg Salary ($)"
    ]], use_container_width=True)
# ----------------------------------
# TAB 2 — Competitive Analysis
# ----------------------------------
with tab2:

    st.subheader("📈 Competitive Market Comparison")

    comparison_df = results_df.copy()

    # Avoid division by zero
    comparison_df["Effort Score"] = (
        comparison_df["Competition"] /
        comparison_df["Demand"].replace(0, 0.01)
    ).round(2)

    display_df = comparison_df[[
        "Role",
        "Match %",
        "Demand",
        "Competition",
        "Avg Salary ($)",
        "Interview Probability",
        "Market ROI Score",
        "Effort Score"
    ]]

    st.dataframe(display_df, use_container_width=True)

    # ----------------------------------
    # Demand vs Competition vs Strength
    # ----------------------------------
    st.markdown("### 📊 Demand vs Competition vs Your Strength")

    fig, ax = plt.subplots(figsize=(8,4))

    x = range(len(display_df))

    ax.bar(x, display_df["Demand"], width=0.25, label="Demand")
    ax.bar([i + 0.25 for i in x], display_df["Competition"], width=0.25, label="Competition")
    ax.bar([i + 0.50 for i in x], display_df["Match %"], width=0.25, label="Your Strength")

    ax.set_xticks([i + 0.25 for i in x])
    ax.set_xticklabels(display_df["Role"], rotation=25)
    ax.set_ylabel("Score (%)")
    ax.legend()

    st.pyplot(fig)

    # ----------------------------------
    # Economic Opportunity
    # ----------------------------------
    st.markdown("### 💰 Economic Opportunity (ROI by Role)")

    fig2, ax2 = plt.subplots(figsize=(8,4))
    ax2.bar(display_df["Role"], display_df["Market ROI Score"])
    ax2.set_ylabel("Market ROI Score")
    ax2.set_title("Economic Upside Comparison")

    st.pyplot(fig2)

    # ----------------------------------
    # Interpretation
    # ----------------------------------
    st.markdown("### 🧠 Interpretation")

    for _, row in display_df.iterrows():

        st.markdown(f"#### {row['Role']}")

        if row["Demand"] > row["Competition"]:
            st.success("Demand exceeds competition → Structural market opportunity.")
        else:
            st.warning("Competition heavier than demand → Entry difficulty higher.")

        if row["Match %"] > 70:
            st.success("Strong skill alignment.")
        elif row["Match %"] > 50:
            st.info("Moderate alignment — skill improvement helps.")
        else:
            st.error("Weak alignment — significant skill investment required.")

        st.markdown("---")

    # ----------------------------------
    # Strategic Insight Summary
    # ----------------------------------
    st.markdown("### 📊 Strategic Insight Summary")

    best_roi = display_df.sort_values("Market ROI Score", ascending=False).iloc[0]["Role"]
    strongest_skill = display_df.sort_values("Match %", ascending=False).iloc[0]["Role"]
    lowest_effort = display_df.sort_values("Effort Score").iloc[0]["Role"]

    st.info(f"💰 Highest Economic Upside: {best_roi}")
    st.info(f"🧠 Strongest Skill Alignment: {strongest_skill}")
    st.info(f"⚡ Lowest Effort Entry: {lowest_effort}")
# ----------------------------------
# TAB 3 — COUNTRY HEATMAP
# ----------------------------------
with tab3:

    country_summary = (
        full_results_df
        .groupby("Country")["Interview Probability"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(6,3))
    ax.barh(country_summary.index, country_summary.values)
    ax.set_title("Top 10 Countries by Interview Probability")
    st.pyplot(fig)

# ----------------------------------
# TAB 4 — GROWTH SIMULATOR
# ----------------------------------
# ----------------------------------
# TAB 4 — GROWTH SIMULATOR
# ----------------------------------
with tab4:

    st.subheader("🎯 Career Growth Simulator")

    new_skill = st.text_input("Add a skill to simulate improvement")

    if new_skill:

        job_row = jobs_df[jobs_df["Role"] == top["Role"]].iloc[0]

        # Recompute job embedding properly
        job_text = job_row["Role"] + " " + job_row["Skills"]
        job_embedding = model.encode(job_text)

        simulated_skills = extracted_skills + [new_skill]

        required_skills = job_row["Skills"].split(",")

        skill_overlap = len(set(simulated_skills) & set(required_skills))
        skill_score = skill_overlap / max(len(required_skills), 1)

        exp_score = 1 if experience >= job_row["Experience"] else experience / max(job_row["Experience"], 1)

        similarity_sim = cosine_similarity(
            [resume_embedding],
            [job_embedding]
        )[0][0]

        simulated_match = 0.5 * similarity_sim + 0.3 * skill_score + 0.2 * exp_score

        simulated_probability = min(
            (0.6 * simulated_match +
             0.15 * (1 - job_row["Competition"]) +
             0.15 * job_row["Demand"] +
             0.1 * job_row["Visa"]) * 100,
            100
        )

        st.success(f"New Interview Probability: {simulated_probability:.2f}%")
# ----------------------------------
# TAB 5 — PERFORMANCE
# ----------------------------------
with tab5:
    st.write("• 90 min daily skill building")
    st.write("• 3 targeted applications per day")
    st.write("• Weekly mock interview")
    st.write("• 20 min walk daily")
    st.write("• Stop job search after 8PM")

# ----------------------------------
# PDF REPORT
# ----------------------------------
def generate_pdf():
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("AI Career Intelligence Report", styles["Heading1"]))
    elements.append(Spacer(1, 12))

    for _, row in results_df.iterrows():
        elements.append(
            Paragraph(
                f"{row['Role']} - {row['Country']} - {row['Interview Probability']:.2f}%",
                styles["Normal"]
            )
        )
        elements.append(Spacer(1, 6))

    doc.build(elements)
    buffer.seek(0)
    return buffer
