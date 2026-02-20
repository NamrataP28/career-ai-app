import streamlit as st
import pandas as pd
import pdfplumber
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pycountry
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
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
# LOAD MODEL
# ----------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ----------------------------------
# LOAD DATA
# ----------------------------------
@st.cache_data
def load_jobs():
    return pd.read_csv("jobs_dataset.csv")

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

st.sidebar.success("Resume processed")

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
        "Avg Salary ($)": row["AvgSalary"],
        "Market Difficulty %": round(market_difficulty, 2),
        "Projects": row.get("Projects", "")
    })

results_df = pd.DataFrame(results).sort_values(
    by="Estimated Interview Probability",
    ascending=False
).head(3)

# ----------------------------------
# TOP METRICS
# ----------------------------------
st.markdown("---")
st.subheader("🏆 Top 3 Strategic Career Options")

cols = st.columns(3)
for i in range(len(results_df)):
    cols[i].metric(
        results_df.iloc[i]["Role"],
        f"{results_df.iloc[i]['Estimated Interview Probability']}%",
        results_df.iloc[i]["Country"]
    )

st.markdown("---")

# ----------------------------------
# TABS
# ----------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Dashboard", "📈 Visual", "🤖 Advisor", "🧠 Performance"]
)

# ----------------------------------
# TAB 1 — TABLE
# ----------------------------------
with tab1:
    st.dataframe(results_df[[
        "Role","Country","Match %",
        "Estimated Interview Probability","Avg Salary ($)"
    ]], use_container_width=True)

# ----------------------------------
# TAB 2 — COMPACT VISUALS
# ----------------------------------
with tab2:
    colA, colB = st.columns(2)

    with colA:
        fig1, ax1 = plt.subplots(figsize=(3,2))
        ax1.barh(results_df["Role"], results_df["Estimated Interview Probability"])
        ax1.set_title("Interview Probability")
        st.pyplot(fig1)

    with colB:
        fig2, ax2 = plt.subplots(figsize=(3,2))
        ax2.barh(results_df["Role"], results_df["Market Difficulty %"])
        ax2.set_title("Market Difficulty")
        st.pyplot(fig2)

# ----------------------------------
# TAB 3 — AI ADVISOR
# ----------------------------------
with tab3:

    top_role = results_df.iloc[0]["Role"]
    selected_row = results_df.iloc[0]

    st.subheader("💼 Real-World Project Simulation")

    if selected_row["Projects"]:
        for project in selected_row["Projects"].split(";"):
            st.write(f"• {project}")

    st.markdown("---")
    st.subheader("📈 Skill Gap Analysis")

    job_row = jobs_df[jobs_df["Role"] == top_role].iloc[0]
    required_skills = job_row["Skills"].split(",")
    missing_skills = list(set(required_skills) - set(extracted_skills))

    if missing_skills:
        for skill in missing_skills:
            st.write(f"• Improve {skill}")
    else:
        st.success("All required skills met.")

    st.markdown("---")
    st.subheader("💬 Career Strategy Chat")

    question = st.text_input("Ask a career question")

    if question:
        if "salary" in question.lower():
            st.write(f"Expected salary: ${selected_row['Avg Salary ($)']}")
        elif "country" in question.lower():
            st.write(f"Best market: {selected_row['Country']}")
        elif "stress" in question.lower():
            st.write("Daily structured routine + weekly review reduces job anxiety.")
        else:
            st.write("Focus on portfolio, networking, and targeted applications.")

# ----------------------------------
# TAB 4 — PERFORMANCE
# ----------------------------------
with tab4:
    st.subheader("🧠 Career Performance System")

    st.markdown("### 🔥 High Performance Habits")
    st.write("• 90 min daily skill building")
    st.write("• 3 targeted applications per day")
    st.write("• Weekly mock interview")

    st.markdown("### 🧘 Stress Control System")
    st.write("• 20 min walk daily")
    st.write("• Stop job search after 8PM")
    st.write("• Track weekly skill wins")

    st.info("Measure effort, not outcome.")

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
                f"{row['Role']} - {row['Country']} - {row['Estimated Interview Probability']}%",
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
