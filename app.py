import streamlit as st
import plotly.express as px
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from services.resume_parser import ResumeParser
from services.visa_model import VisaModel
from services.gpt_service import GPTService

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(layout="wide")
st.title("🚀 AI Career Intelligence Platform")

# -----------------------------------
# INITIALIZE SERVICES
# -----------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")
parser = ResumeParser()
visa_model = VisaModel()
gpt_service = GPTService()

# -----------------------------------
# FILE UPLOAD
# -----------------------------------

file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
if not file:
    st.stop()

resume_text = parser.extract_text(file)
resume_embedding = model.encode(resume_text)

# -----------------------------------
# GLOBAL ROLE BANK (Corporate Only)
# -----------------------------------

def get_global_role_bank():
    return [
        "Data Analyst","Business Analyst","Product Analyst",
        "Marketing Analyst","Financial Analyst","Risk Analyst",
        "Finance Manager","Investment Analyst",
        "Product Manager","Growth Product Manager",
        "Strategy Consultant","Management Consultant",
        "Marketing Manager","Digital Marketing Manager",
        "Operations Manager","Program Manager",
        "Software Engineer","Backend Engineer",
        "Machine Learning Engineer","Data Engineer",
        "Head of Analytics","Director of Strategy",
        "Supply Chain Manager","Commercial Manager",
        "Corporate Strategy Manager","FP&A Analyst"
    ]

# -----------------------------------
# DETECT TOP MATCHING ROLES
# -----------------------------------

def detect_top_roles(resume_embedding, top_n=5):
    role_bank = get_global_role_bank()
    scored = []

    for role in role_bank:
        role_emb = model.encode(role)
        score = cosine_similarity([resume_embedding],[role_emb])[0][0]
        scored.append((role, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [r[0] for r in scored[:top_n]]

top_roles = detect_top_roles(resume_embedding)

st.subheader("🔎 Top Resume-Matched Role Categories")
st.write(top_roles)

# -----------------------------------
# COUNTRY MAP
# -----------------------------------

countries = {
    "USA": "us",
    "UK": "gb",
    "Canada": "ca",
    "Germany": "de",
    "India": "in",
    "Singapore": "sg",
    "Australia": "au"
}

ppp_index = {
    "USA":1.0,
    "UK":1.0,
    "Canada":0.9,
    "Germany":0.95,
    "India":0.35,
    "Singapore":1.1,
    "Australia":0.95
}

# -----------------------------------
# FETCH LIVE ROLES (RapidAPI JSearch)
# -----------------------------------

@st.cache_data(ttl=1800)
def fetch_live_roles(top_roles):

    all_roles = []

    headers = {
        "X-RapidAPI-Key": st.secrets["RAPIDAPI_KEY"],
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
    }

    for country_name, country_code in countries.items():

        for role in top_roles:

            url = "https://jsearch.p.rapidapi.com/search"

            params = {
                "query": role,
                "page": "1",
                "num_pages": "1",
                "country": country_code
            }

            try:
                response = requests.get(url, headers=headers, params=params)

                if response.status_code != 200:
                    continue

                data = response.json()
                jobs = data.get("data", [])

                for job in jobs:

                    salary = job.get("job_salary")
                    salary_ppp = 0

                    if salary and isinstance(salary, (int, float)):
                        salary_ppp = salary / ppp_index.get(country_name, 1)

                    all_roles.append({
                        "Role": job.get("job_title"),
                        "Country": country_name,
                        "Demand": len(jobs),
                        "Salary_PPP": salary_ppp
                    })

            except:
                continue

    df = pd.DataFrame(all_roles)

    if df.empty:
        return df

    return df.drop_duplicates(subset=["Role","Country"])

# -----------------------------------
# FETCH DATA
# -----------------------------------

roles_df = fetch_live_roles(top_roles)

if roles_df.empty:
    st.warning("No live corporate roles found. Check RapidAPI key.")
    st.stop()

# -----------------------------------
# GLOBAL SCORING
# -----------------------------------

results = []

max_demand = roles_df["Demand"].max() or 1
max_salary = roles_df["Salary_PPP"].max() or 1

for _, row in roles_df.iterrows():

    job_embedding = model.encode(row["Role"])
    similarity = cosine_similarity([resume_embedding],[job_embedding])[0][0]

    visa_score = visa_model.visa_score(row["Country"])
    demand_norm = row["Demand"]/max_demand
    salary_norm = row["Salary_PPP"]/max_salary if max_salary else 0

    final_score = (
        0.5*similarity +
        0.25*demand_norm +
        0.15*visa_score +
        0.10*salary_norm
    )

    results.append({
        "Role":row["Role"],
        "Country":row["Country"],
        "Match %":round(similarity*100,2),
        "Visa Score %":round(visa_score*100,2),
        "Market Demand %":round(demand_norm*100,2),
        "PPP Salary Index %":round(salary_norm*100,2),
        "Global Score %":round(final_score*100,2)
    })

ranked_df = pd.DataFrame(results).sort_values("Global Score %",ascending=False)

# -----------------------------------
# COUNTRY FILTER
# -----------------------------------

st.subheader("🌍 Global Opportunity Ranking")

country_filter = st.selectbox(
    "Filter by Country",
    ["Worldwide"] + list(ranked_df["Country"].unique())
)

if country_filter == "Worldwide":
    display_df = ranked_df
else:
    display_df = ranked_df[ranked_df["Country"] == country_filter]

st.dataframe(display_df.head(20),use_container_width=True)

# -----------------------------------
# VISUALIZATION
# -----------------------------------

fig = px.bar(
    display_df.head(10),
    x="Global Score %",
    y="Role",
    color="Country",
    orientation="h",
    hover_data=["Match %","Visa Score %","Market Demand %"],
    height=500
)

st.plotly_chart(fig,use_container_width=True)

# -----------------------------------
# GPT ROADMAP
# -----------------------------------

if st.button("Generate Career Roadmap for Top Role"):
    top_role = ranked_df.iloc[0]["Role"]
    roadmap = gpt_service.generate_roadmap(resume_text, top_role)
    st.markdown(roadmap)
