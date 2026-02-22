import streamlit as st
import pandas as pd
from database.db import get_user_history


# --------------------------------------------------
# SESSION CHECK
# --------------------------------------------------

if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
    st.warning("Please login first.")
    st.stop()

email = st.session_state.get("email")

# --------------------------------------------------
# FETCH HISTORY
# --------------------------------------------------

history = get_user_history(email)

if not history:
    st.info("No analysis history yet. Run your first analysis.")
    st.stop()

# Convert to DataFrame
df = pd.DataFrame(history, columns=[
    "Role",
    "Country",
    "Opportunity",
    "Probability",
    "Percentile",
    "Created At"
])

# Sort oldest → newest for chart
df = df.sort_values("Created At")

# --------------------------------------------------
# DASHBOARD UI
# --------------------------------------------------

st.title("📊 Your Career Progress Dashboard")

col1, col2, col3 = st.columns(3)

col1.metric("Latest Opportunity Score",
            f"{df.iloc[-1]['Opportunity']:.2f}")

col2.metric("Latest Interview Probability %",
            f"{df.iloc[-1]['Probability']:.2f}")

col3.metric("Latest Market Percentile %",
            f"{df.iloc[-1]['Percentile']:.2f}")

st.markdown("---")

# --------------------------------------------------
# PROGRESS CHART
# --------------------------------------------------

st.subheader("📈 Career Growth Trend")

chart_df = df[["Created At", "Opportunity", "Probability"]]
chart_df = chart_df.set_index("Created At")

st.line_chart(chart_df)

st.markdown("---")

# --------------------------------------------------
# FULL HISTORY TABLE
# --------------------------------------------------

st.subheader("📜 Full Analysis History")

st.dataframe(df, use_container_width=True)

# --------------------------------------------------
# PERFORMANCE INSIGHT
# --------------------------------------------------

improvement = (
    df.iloc[-1]["Opportunity"] -
    df.iloc[0]["Opportunity"]
)

if improvement > 0:
    st.success(f"You improved your Opportunity Score by {improvement:.2f} points.")
elif improvement < 0:
    st.warning("Your opportunity score declined. Consider skill reinforcement.")
else:
    st.info("Your score is stable.")
