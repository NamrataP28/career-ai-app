import streamlit as st

st.title("Step 1: Career Questionnaire")

role = st.selectbox("Target Role",
    ["Data Analyst","Product Manager","Marketing Manager",
     "Financial Analyst","Strategy Consultant","Software Engineer"]
)

country = st.selectbox("Target Country",
    ["USA","UK","Canada","Germany","India","Singapore","Australia"]
)

current_salary = st.number_input("Current Salary (USD)", min_value=0)

if st.button("Proceed to Analysis"):
    st.session_state["role"] = role
    st.session_state["country"] = country
    st.session_state["salary"] = current_salary
    st.switch_page("pages/2_Analysis.py")
