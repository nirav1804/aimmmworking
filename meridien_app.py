import streamlit as st
import pandas as pd
from meridien import Meridien
import plotly.express as px # Needed for plot_response_curves if it returns a Plotly figure

st.set_page_config(page_title="MMM with Meridien", layout="wide")

st.title("📊 Marketing Mix Modeling - Meridien")

# Upload CSV
uploaded_file = st.file_uploader("Upload your media and revenue data (.csv)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    # --- Data Preprocessing for Meridien ---
    # Ensure 'date_start' and 'Revenue' columns exist
    required_columns = ['date_start', 'Revenue']
    if not all(col in df.columns for col in required_columns):
        st.error(f"❌ Missing required columns. Please ensure your CSV has '{required_columns[0]}' and '{required_columns[1]}' columns.")
        st.stop() # Stop execution if columns are missing

    # Convert 'date_start' to datetime objects
    # Assuming date format is DD/MM/YY based on your CSV snippet
    try:
        df['date_start'] = pd.to_datetime(df['date_start'], format='%d/%m/%y')
    except ValueError:
        st.error("❌ Date format in 'date_start' column is not 'DD/MM/YY'. Please check your CSV or adjust the date format in the code.")
        st.stop()

    st.subheader("⚙️ Running Meridien Model...")
    try:
        # Initialize Meridien model
        # 'date_start' is used as the date column based on your CSV
        mmm = Meridien(data=df, target_column='Revenue', date_column='date_start')

        # Fit the model
        mmm.fit()

        # Optimize the media plan
        results = mmm.optimize()

        st.subheader("📈 Optimized Media Plan")
        st.dataframe(results.optimized_spend)

        st.subheader("🧠 Model Insights")
        st.write(f"**R² Score:** {results.r2_score:.4f}") # Display R2 score with formatting
        st.write("**ROI Table:**")
        st.dataframe(results.roi_by_channel)

        st.subheader("📉 Visuals")
        # Meridien's plot_response_curves should return a Plotly figure
        fig = results.plot_response_curves()
        if fig:
            st.plotly_chart(fig, use_container_width=True) # Use use_container_width for responsiveness
        else:
            st.warning("Meridien did not generate a response curve plot. This might happen with insufficient data or model issues.")

    except Exception as e:
        st.error(f"❌ Error running model: {e}")
        st.exception(e) # Display full traceback for debugging
