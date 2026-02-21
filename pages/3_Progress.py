import streamlit as st

st.title("Step 3: Career Roadmap & Progress")

if "user_inputs" not in st.session_state:
    st.warning("Complete Questionnaire first.")
    st.stop()

goal = st.session_state["user_inputs"]["goal"]

st.subheader("Your Recommended Next Steps")

if goal == "Switch Role":
    st.write("1️⃣ Build Portfolio Projects")
    st.write("2️⃣ Earn Relevant Certification")
    st.write("3️⃣ Optimize Resume for Target Role")
    st.write("4️⃣ Start Networking with Hiring Managers")

elif goal == "Return After Career Break":
    st.write("1️⃣ Skill Refresh (30 Days)")
    st.write("2️⃣ Freelance / Contract Projects")
    st.write("3️⃣ Update LinkedIn Profile")

elif goal == "Salary Benchmark":
    st.write("1️⃣ Negotiate using market data")
    st.write("2️⃣ Apply to 5 higher-paying roles weekly")

else:
    st.write("Continue skill enhancement and apply strategically.")

st.success("You are on a structured career growth path.")
