import streamlit as st
from services.role_taxonomy import get_role_universe
from services.resume_parser import ResumeParser

st.title("Step 1: Career Questionnaire")

parser = ResumeParser()

roles = get_role_universe()

# ---------------------------
# Resume Upload Section
# ---------------------------

st.subheader("Upload Your Resume")

resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

resume_text = ""

if resume_file:
    resume_text = parser.extract_text(resume_file)
    st.success("Resume Uploaded Successfully")

# ---------------------------
# Career Inputs
# ---------------------------

col1, col2 = st.columns(2)

with col1:
    target_role = st.selectbox("Primary Target Role", roles)
    alt_role = st.selectbox("Alternate Role Interest", roles)
    experience = st.slider("Years of Experience", 0, 20, 3)

with col2:
    country = st.multiselect(
        "Target Countries",
        ["USA","UK","Germany","India","Singapore","Canada","Australia"]
    )
    currency = st.selectbox("Preferred Currency", ["USD","EUR","INR","GBP","SGD"])
    career_goal = st.selectbox(
        "Career Objective",
        ["Salary Benchmark","Switch Role","Growth in Current Role",
         "Return After Career Break","Resume Health Check"]
    )

current_salary = st.number_input("Current Salary", min_value=0)

# ---------------------------
# Save to Session
# ---------------------------

if st.button("Proceed to Analysis"):

    if not resume_text:
        st.error("Please upload your resume first.")
        st.stop()

    if not country:
        st.error("Please select at least one country.")
        st.stop()

    st.session_state["resume_text"] = resume_text

    st.session_state["user_inputs"] = {
        "role": target_role,
        "alt_role": alt_role,
        "exp": experience,
        "country": country,
        "currency": currency,
        "goal": career_goal,
        "salary": current_salary
    }

    st.success("Saved! Go to Analysis tab.")
