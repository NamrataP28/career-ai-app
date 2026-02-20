import streamlit as st
import pandas as pd
import pdfplumber
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pycountry
from openai import OpenAI
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
import io

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(page_title="AI Career Intelligence Engine", layout="wide")

st.title("🚀 AI Career Intelligence Engine")
st.markdown("AI-powered global career growth intelligence platform")

# ----------------------------------
# BASIC LOGIN (SESSION)
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
# SIDEBAR SETTINGS
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

resume_text = ""
extracted_skills = []
experience = 0

# ----------------------------------
# RESUME PROCESSING
# ----------------------------------
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
    st.info("Upload resume to begin")
    st.stop()

# ----------------------------------
# MATCHING ENGINE
# ----------------------------------
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

    probability = (
        0.6 * match_score +
        0.15 * (1 - row["Competition"]) +
        0.15 * row["Demand"] +
        0.1 * row["Visa"]
    )

    home_boost = 1.15 if row["Country"] == home_country else 1
    adjusted_probability = probability * home_boost

    market_difficulty = (
        row["Competition"] * 0.6 +
        (1 - row["Demand"]) * 0.4
    ) * 100

    results.append({
        "Role": row["Role"],
        "Country": row["Country"],
        "Match %": round(match_score * 100, 2),
        "Estimated Interview Probability": round(min(adjusted_probability * 100, 100), 2),
        "Avg Salary ($)": row["AvgSalary"],
        "Market Difficulty %": round(market_difficulty, 2)
    })

results_df = pd.DataFrame(results).sort_values(
    by="Estimated Interview Probability",
    ascending=False
).head(3)

# ----------------------------------
# DASHBOARD
# ----------------------------------
st.markdown("---")
st.subheader("🏆 Top 3 Strategic Career Options")

cols = st.columns(3)
for i in range(len(results_df)):
    cols[i].metric(
        results_df.iloc[i]["Role"],
        f"{results_df.iloc[i]['Estimated Interview Probability']:.2f}%",
        results_df.iloc[i]["Country"]
    )

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📈 Visual Intelligence", "🤖 AI Advisor"])

# -----------------------------
# TAB 1
# -----------------------------
with tab1:
    st.dataframe(
        results_df[
            ["Role","Country","Match %","Estimated Interview Probability","Avg Salary ($)"]
        ],
        use_container_width=True
    )

# -----------------------------
# TAB 2
# -----------------------------
with tab2:
    colA, colB = st.columns(2)

    with colA:
        fig1, ax1 = plt.subplots(figsize=(4,3))
        ax1.bar(results_df["Role"], results_df["Estimated Interview Probability"])
        ax1.set_title("Interview Probability (%)")
        ax1.tick_params(axis='x', rotation=45)
        st.pyplot(fig1)

    with colB:
        fig2, ax2 = plt.subplots(figsize=(4,3))
        ax2.bar(results_df["Role"], results_df["Market Difficulty %"])
        ax2.set_title("Market Difficulty (%)")
        ax2.tick_params(axis='x', rotation=45)
        st.pyplot(fig2)

# -----------------------------
# TAB 3 — GPT ADVISOR
# -----------------------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

with tab3:
    st.subheader("🤖 Strategic Career Advisor")

    user_question = st.text_input("Ask a strategic career question")

    if user_question:

        prompt = f"""
        Candidate Skills: {extracted_skills}
        Experience: {experience}
        Top 3 Roles: {results_df['Role'].tolist()}
        Countries: {results_df['Country'].tolist()}

        Provide practical, realistic career advice.
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        st.write(response.choices[0].message.content)

# ----------------------------------
# PDF REPORT GENERATOR
# ----------------------------------
def generate_pdf():
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("AI Career Intelligence Report", styles["Heading1"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Skills: {extracted_skills}", styles["Normal"]))
    elements.append(Paragraph(f"Experience: {experience} years", styles["Normal"]))
    elements.append(Spacer(1, 12))

    data = [["Role", "Country", "Interview Probability"]]

    for _, row in results_df.iterrows():
        data.append([
            row["Role"],
            row["Country"],
            f"{row['Estimated Interview Probability']:.2f}%"
        ])

    table = Table(data)
    elements.append(table)

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
