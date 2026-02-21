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
# CORPORATE SKILL ENGINE (EXPANDED)
# -----------------------------------

def extract_skills(text):

    corporate_skills = {

        # Data & Analytics
        "data analytics","business intelligence","sql","python","r",
        "power bi","tableau","data visualization","statistics",
        "machine learning","predictive modeling","forecasting",
        "etl","data warehousing","big data","hadoop","spark",

        # Product & Strategy
        "product management","roadmap","stakeholder management",
        "agile","scrum","ux","user research","market research",
        "a/b testing","growth strategy","pricing strategy",

        # Finance & Risk
        "financial modeling","valuation","budgeting","credit risk",
        "fraud detection","investment analysis","portfolio management",
        "financial reporting","p&l","compliance","aml","kyc",

        # Marketing & Growth
        "seo","sem","google analytics","campaign management",
        "brand strategy","performance marketing",
        "digital marketing","content strategy","crm",
        "customer acquisition","conversion optimization",

        # Technology & Engineering
        "software engineering","backend","frontend",
        "react","node","java","c++","cloud","aws","azure","gcp",
        "devops","docker","kubernetes","microservices",
        "api integration","system architecture",

        # Consulting & Operations
        "consulting","business transformation","process optimization",
        "operations management","supply chain",
        "change management","strategy development",
        "lean","six sigma",

        # Leadership & Corporate
        "team leadership","cross functional collaboration",
        "executive communication","board reporting",
        "stakeholder alignment","project management",
        "program management","decision making",
        "corporate strategy"
    }

    text_lower = text.lower()
    return list(set([s for s in corporate_skills if s in text_lower]))

resume_skills = extract_skills(resume_text)

# -----------------------------------
# MARKET SCOPE
# -----------------------------------

region_mode = st.radio(
    "Market Scope",
    ["Core Markets", "Full Global Scan"]
)

def get_country_map(mode):

    core = {
        "USA": "us",
        "UK": "gb",
        "Canada": "ca",
        "Germany": "de",
        "India": "in",
        "Singapore": "sg",
        "Australia": "au",
        "UAE": "ae"
    }

    full = {
        "USA": "us","Canada": "ca","Mexico": "mx",
        "UK": "gb","Ireland": "ie","Germany": "de",
        "France": "fr","Netherlands": "nl","Belgium": "be",
        "Spain": "es","Italy": "it","Poland": "pl",
        "Sweden": "se","Norway": "no","Denmark": "dk",
        "Austria": "at","Switzerland": "ch",
        "UAE": "ae","Saudi Arabia": "sa",
        "India": "in","Singapore": "sg",
        "Australia": "au","New Zealand": "nz",
        "South Africa": "za","Brazil": "br"
    }

    return core if mode == "Core Markets" else full

countries = get_country_map(region_mode)

# -----------------------------------
# PPP INDEX
# -----------------------------------

ppp_index = {
    "USA":1.0,"Canada":0.9,"Mexico":0.45,
    "UK":1.0,"Ireland":1.0,"Germany":0.95,
    "France":0.95,"Netherlands":1.0,"Belgium":0.95,
    "Spain":0.8,"Italy":0.85,"Poland":0.6,
    "Sweden":1.0,"Norway":1.2,"Denmark":1.1,
    "Austria":0.95,"Switzerland":1.3,
    "UAE":0.9,"Saudi Arabia":0.7,
    "India":0.35,"Singapore":1.1,
    "Australia":0.95,"New Zealand":0.85,
    "South Africa":0.4,"Brazil":0.5
}

# -----------------------------------
# FETCH LIVE ROLES
# -----------------------------------

@st.cache_data(ttl=1800)
def fetch_live_roles(resume_text):

    search_keywords = " ".join(resume_skills) if resume_skills else resume_text[:300]
    all_roles = []

    for country_name, code in countries.items():

        for page in range(1, 3):

            url = f"https://api.adzuna.com/v1/api/jobs/{code}/search/{page}"

            params = {
                "app_id": st.secrets["ADZUNA_APP_ID"],
                "app_key": st.secrets["ADZUNA_APP_KEY"],
                "results_per_page": 20,
                "what": search_keywords
            }

            try:
                r = requests.get(url, params=params)
                data = r.json()
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
                        "Salary_PPP": salary_ppp,
                        "Description": job.get("description")
                    })

            except:
                continue

    df = pd.DataFrame(all_roles)
    if df.empty:
        return df

    return df.drop_duplicates(subset=["Role","Country"])

roles_df = fetch_live_roles(resume_text)

if roles_df.empty:
    st.warning("No live corporate roles found.")
    st.stop()

# -----------------------------------
# SCORING ENGINE
# -----------------------------------

results = []
max_demand = roles_df["Demand"].max() or 1
max_salary = roles_df["Salary_PPP"].max() or 1

exclude_keywords = [
    "call centre","call center","telecaller",
    "driver","technician","cashier",
    "warehouse","delivery","retail associate"
]

for _, row in roles_df.iterrows():

    if any(word in row["Role"].lower() for word in exclude_keywords):
        continue

    job_embedding = model.encode(row["Role"])
    similarity = cosine_similarity([resume_embedding],[job_embedding])[0][0]

    visa_score = visa_model.visa_score(row["Country"])
    demand_norm = row["Demand"]/max_demand
    salary_norm = row["Salary_PPP"]/max_salary

    job_skills = extract_skills(row["Description"] or "")
    skill_overlap = len(set(resume_skills)&set(job_skills))
    skill_match = skill_overlap/max(len(job_skills),1)

    if any(keyword in row["Role"].lower() for keyword in 
           ["manager","lead","head","director","analyst","consultant","strategist"]):
        skill_match *= 1.1

    final_score = (
        0.40*similarity +
        0.20*demand_norm +
        0.15*visa_score +
        0.15*salary_norm +
        0.10*skill_match
    )

    results.append({
        "Role":row["Role"],
        "Country":row["Country"],
        "Skill Match %":round(similarity*100,2),
        "Technical Overlap %":round(skill_match*100,2),
        "Visa Feasibility %":round(visa_score*100,2),
        "Market Demand %":round(demand_norm*100,2),
        "PPP Salary Index %":round(salary_norm*100,2),
        "Global Opportunity Score %":round(final_score*100,2)
    })

ranked_df = pd.DataFrame(results).sort_values(
    "Global Opportunity Score %",
    ascending=False
)

# -----------------------------------
# COUNTRY FILTER
# -----------------------------------

st.subheader("🌍 Global Corporate Ranking")

country_filter = st.selectbox(
    "Select Country",
    ["Worldwide"] + sorted(ranked_df["Country"].unique())
)

display_df = ranked_df if country_filter=="Worldwide" else ranked_df[ranked_df["Country"]==country_filter]

st.dataframe(display_df.head(25),use_container_width=True)

# -----------------------------------
# INTERACTIVE VISUAL
# -----------------------------------

fig = px.scatter(
    display_df.head(25),
    x="Skill Match %",
    y="Market Demand %",
    size="PPP Salary Index %",
    color="Country",
    hover_data=[
        "Global Opportunity Score %",
        "Visa Feasibility %",
        "Technical Overlap %"
    ],
    height=600
)

st.plotly_chart(fig,use_container_width=True)

# -----------------------------------
# CAREER CLUSTERING
# -----------------------------------

st.subheader("🔎 Adjacent Career Paths")

top_embedding = model.encode(ranked_df.iloc[0]["Role"])

ranked_df["Role Similarity"] = ranked_df["Role"].apply(
    lambda r: cosine_similarity(
        [top_embedding],
        [model.encode(r)]
    )[0][0]
)

cluster_df = ranked_df.sort_values(
    "Role Similarity",
    ascending=False
).iloc[1:6]

st.dataframe(
    cluster_df[["Role","Country","Global Opportunity Score %"]],
    use_container_width=True
)

# -----------------------------------
# GPT ROADMAP
# -----------------------------------

st.subheader("🎯 AI Career Roadmap")

if st.button("Generate Personalized Roadmap for Top Role"):

    top_role = ranked_df.iloc[0]["Role"]
    roadmap = gpt_service.generate_roadmap(resume_text, top_role)
    st.markdown(roadmap)
