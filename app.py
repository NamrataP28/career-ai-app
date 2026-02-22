import streamlit as st
from database.db import init_db, authenticate_user, register_user

# ------------------ INIT ------------------
init_db()
st.set_page_config(layout="wide")

# ------------------ SESSION STATE ------------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if "email" not in st.session_state:
    st.session_state["email"] = None


# ------------------ THEME ------------------
st.markdown("""
<style>
html, body, [class*="css"]  {
    font-family: 'Roboto', sans-serif;
}

.main {
    background: linear-gradient(135deg, #6A11CB 0%, #2575FC 100%);
}

.block-container {
    padding-top: 2rem;
}

.card {
    background: white;
    padding: 1.5rem;
    border-radius: 16px;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.1);
    margin-bottom: 1rem;
}

.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0px 5px 20px rgba(0,0,0,0.08);
}

.gold {
    color: #F7B500;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# AUTHENTICATION SECTION
# =====================================================

if not st.session_state["authenticated"]:

    st.title("🚀 Career Intelligence Platform")

    st.markdown("""
    <div class="card">
    <h3>Understand Your Market Position Like 6-Figure Leaders</h3>
    <p>
    We analyze your resume against global job demand, supply competitiveness,
    salary benchmarks, and skill gaps to show your real-world standing.
    </p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Login", "Register"])

    # -------- LOGIN --------
    with tab1:
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            user = authenticate_user(email, password)
            if user:
                st.session_state["authenticated"] = True
                st.session_state["email"] = email
                st.success("Login successful.")
                st.rerun()
            else:
                st.error("Invalid credentials.")

    # -------- REGISTER --------
    with tab2:
        new_email = st.text_input("New Email")
        new_password = st.text_input("New Password", type="password")

        if st.button("Register"):
            register_user(new_email, new_password)
            st.success("Account created. You can now login.")

else:

    # =====================================================
    # LOGGED-IN LANDING
    # =====================================================

    st.sidebar.success(f"Logged in as {st.session_state['email']}")

    if st.sidebar.button("Logout"):
        st.session_state["authenticated"] = False
        st.session_state["email"] = None
        st.rerun()

    st.title("🚀 Career Intelligence Dashboard")

    st.markdown("""
    <div class="card">
    <h3>Welcome Back</h3>
    <p>
    Use the sidebar to:
    <ul>
    <li>📋 Complete Questionnaire</li>
    <li>📊 Run Market Analysis</li>
    <li>📈 View Progress Dashboard</li>
    </ul>
    </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    col1.markdown("""
    <div class="metric-card">
    <h4>Resume Intelligence</h4>
    <p>Understand skill strength and gaps</p>
    </div>
    """, unsafe_allow_html=True)

    col2.markdown("""
    <div class="metric-card">
    <h4>Market Modeling</h4>
    <p>Demand vs Supply competitiveness</p>
    </div>
    """, unsafe_allow_html=True)

    col3.markdown("""
    <div class="metric-card">
    <h4>Salary Benchmark</h4>
    <p>Compare your compensation globally</p>
    </div>
    """, unsafe_allow_html=True)

    st.info("Navigate using the sidebar to begin your analysis.")
