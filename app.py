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

for _, row in jobs_df.iterrows():

    job_text = row["Role"] + " " + row["Skills"]
    job_embedding = model.encode(job_text)

    similarity = cosine_similarity([resume_embedding], [job_embedding])[0][0]

    required_skills = row["Skills"].split(",")
    skill_overlap = len(set(extracted_skills) & set(required_skills))
    skill_score = skill_overlap / len(required_skills)

    exp_score = 1 if experience >= row["Experience"] else experience / row["Experience"]
    match_score = 0.5 * similarity + 0.3 * skill_score + 0.2 * exp_score

    salary_score = row["AvgSalary"] / jobs_df["AvgSalary"].max()

    competitiveness_index = (
        0.4 * match_score +
        0.2 * row["Demand"] +
        0.15 * (1 - row["Competition"]) +
        0.15 * row["Visa"] +
        0.1 * salary_score
    )

    probability = (
        0.6 * match_score +
        0.15 * (1 - row["Competition"]) +
        0.15 * row["Demand"] +
        0.1 * row["Visa"]
    )

    home_boost = 1.15 if row["Country"] == home_country else 1
    adjusted_probability = min(probability * home_boost * 100, 100)

    market_difficulty = (
        row["Competition"] * 0.6 +
        (1 - row["Demand"]) * 0.4
    ) * 100

    results.append({
        "Role": row["Role"],
        "Country": row["Country"],
        "Match %": round(match_score * 100, 2),
        "Estimated Interview Probability": round(adjusted_probability, 2),
        "Competitiveness Index": round(competitiveness_index * 100, 2),
        "Market Difficulty %": round(market_difficulty, 2),
        "Avg Salary ($)": row["AvgSalary"],
        "Projects": row.get("Projects", "")
    })

full_results_df = pd.DataFrame(results)

# Percentile Ranking
full_results_df["Percentile Rank"] = (
    full_results_df["Match %"].rank(pct=True) * 100
).round(2)

results_df = full_results_df.sort_values(
    by="Competitiveness Index",
    ascending=False
).head(3)

# ----------------------------------
# EXECUTIVE SCORECARD
# ----------------------------------
st.markdown("---")
st.subheader("📊 Executive Scorecard")

top = results_df.iloc[0]

col1, col2, col3, col4 = st.columns(4)

col1.metric("Competitiveness",
            f"{top['Competitiveness Index']:.1f}%")

col2.metric("Interview Probability",
            f"{top['Estimated Interview Probability']:.1f}%")

col3.metric("Market Difficulty",
            f"{top['Market Difficulty %']:.1f}%")

col4.metric("Percentile Rank",
            f"{top['Percentile Rank']:.0f}th")

st.markdown("---")

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
        "Role","Country","Match %",
        "Estimated Interview Probability",
        "Competitiveness Index",
        "Percentile Rank",
        "Avg Salary ($)"
    ]], use_container_width=True)

# ----------------------------------
# TAB 2 — COMPETITIVE BREAKDOWN
# ----------------------------------
with tab2:

    job_row = jobs_df[jobs_df["Role"] == top["Role"]].iloc[0]

    factors = {
        "Skill Strength": top["Match %"],
        "Market Demand": job_row["Demand"] * 100,
        "Low Competition": (1 - job_row["Competition"]) * 100,
        "Visa Accessibility": job_row["Visa"] * 100,
        "Salary Power": (job_row["AvgSalary"] / jobs_df["AvgSalary"].max()) * 100,
    }

    factor_df = pd.DataFrame({
        "Factor": factors.keys(),
        "Score": factors.values()
    })

    fig, ax = plt.subplots(figsize=(6,3))
    ax.barh(factor_df["Factor"], factor_df["Score"])
    ax.set_xlim(0,100)
    st.pyplot(fig)

# ----------------------------------
# TAB 3 — COUNTRY HEATMAP
# ----------------------------------
with tab3:

    country_summary = (
        full_results_df
        .groupby("Country")["Competitiveness Index"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(6,3))
    ax.barh(country_summary.index, country_summary.values)
    ax.set_xlim(0,100)
    ax.set_title("Top 10 Countries for Your Profile")
    st.pyplot(fig)

# ----------------------------------
# TAB 4 — GROWTH SIMULATOR
# ----------------------------------
with tab4:

    st.subheader("🎯 Career Growth Simulator")

    new_skill = st.text_input("Add a skill to simulate improvement")

    if new_skill:

        simulated_skills = extracted_skills + [new_skill]

        required_skills = job_row["Skills"].split(",")
        skill_overlap = len(set(simulated_skills) & set(required_skills))
        skill_score = skill_overlap / len(required_skills)

        new_match_score = 0.5 * similarity + 0.3 * skill_score + 0.2 * exp_score

        simulated_index = (
            0.4 * new_match_score +
            0.2 * job_row["Demand"] +
            0.15 * (1 - job_row["Competition"]) +
            0.15 * job_row["Visa"] +
            0.1 * (job_row["AvgSalary"] / jobs_df["AvgSalary"].max())
        ) * 100

        st.success(f"New Competitiveness Index: {simulated_index:.2f}%")

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
                f"{row['Role']} - {row['Country']} - {row['Competitiveness Index']:.2f}%",
                styles["Normal"]
            )
        )
        elements.append(Spacer(1, 6))

    doc.build(elements)
    buffer.seek(0)
    return buffer

pdf_file = generate_pdf()

st.download_button(
    label="📄 Download Career Report",
    data=pdf_file,
    file_name="career_report.pdf",
    mime="application/pdf"
)
