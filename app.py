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
# GLOBAL ROLE BANK
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
        "Head of Analytics","Director of Strategy"
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
# FETCH LIVE ROLES (DEBUG ENABLED)
# -----------------------------------

@st.cache_data(ttl=1800)
def fetch_live_roles(top_roles):

    all_roles = []

    for country_name, code in countries.items():

        for role in top_roles:

            url = f"https://api.adzuna.com/v1/api/jobs/{code}/search/1"

            params = {
                "app_id": st.secrets["ADZUNA_APP_ID"],
                "app_key": st.secrets["ADZUNA_APP_KEY"],
                "results_per_page": 10,
                "what": role
            }

            try:
                response = requests.get(url, params=params)

                st.write(f"Calling API → {country_name} | {role}")
                st.write("Status Code:", response.status_code)

                if response.status_code != 200:
                    st.error(response.text)
                    continue

                data = response.json()

                total_demand = data.get("count", 0)

                for job in data.get("results", []):

                    salary_min = job.get("salary_min") or 0
                    salary_max = job.get("salary_max") or 0
                    avg_salary = (salary_min + salary_max)/2 if salary_max else salary_min
                    salary_ppp = avg_salary / ppp_index.get(country_name,1) if avg_salary else 0

                    all_roles.append({
                        "Role": job.get("title"),
                        "Country": country_name,
                        "Demand": total_demand,
                        "Salary_PPP": salary_ppp
                    })

            except Exception as e:
                st.error(f"Exception in {country_name} - {role}: {e}")
                continue

    df = pd.DataFrame(all_roles)

    st.write("Total roles fetched:", df.shape)

    if df.empty:
        return df

    return df.drop_duplicates(subset=["Role","Country"])

# -----------------------------------
# CALL FETCH
# -----------------------------------

roles_df = fetch_live_roles(top_roles)

if roles_df.empty:
    st.warning("No live corporate roles found.")
    st.stop()

# -----------------------------------
# SCORING
# -----------------------------------

results = []

max_demand = roles_df["Demand"].max() or 1
max_salary = roles_df["Salary_PPP"].max() or 1

for _, row in roles_df.iterrows():

    job_embedding = model.encode(row["Role"])
    similarity = cosine_similarity([resume_embedding],[job_embedding])[0][0]

    visa_score = visa_model.visa_score(row["Country"])
    demand_norm = row["Demand"]/max_demand
    salary_norm = row["Salary_PPP"]/max_salary

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

st.subheader("🌍 Global Opportunity Ranking")
st.dataframe(ranked_df.head(20),use_container_width=True)

# -----------------------------------
# VISUAL
# -----------------------------------

fig = px.bar(
    ranked_df.head(10),
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

if st.button("Generate Roadmap"):
    top_role = ranked_df.iloc[0]["Role"]
    roadmap = gpt_service.generate_roadmap(resume_text, top_role)
    st.markdown(roadmap)
