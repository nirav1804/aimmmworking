import streamlit as st
import pandas as pd
from meridien_src.meridien import Meridien # Correct import path
import plotly.express as px

st.set_page_config(page_title="MMM with Meridien", layout="wide")

st.title("📊 Marketing Mix Modeling - Meridien")

# Upload CSV
uploaded_file = st.file_uploader("Upload your media and revenue data (.csv)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    # --- Data Preprocessing for Meridien ---
    required_columns = ['date_start', 'Revenue']
    if not all(col in df.columns for col in required_columns):
        st.error(f"❌ Missing required columns. Please ensure your CSV has '{required_columns[0]}' and '{required_columns[1]}' columns.")
        st.stop()

    try:
        # Ensure 'date_start' is parsed correctly. Using errors='coerce' to turn unparseable dates into NaT
        df['date_start'] = pd.to_datetime(df['date_start'], format='%d/%m/%y', errors='coerce')
        if df['date_start'].isnull().any():
            st.warning("Some dates in 'date_start' could not be parsed with 'DD/MM/YY' format. Rows with invalid dates might be affected.")
    except Exception as e:
        st.error(f"❌ Error parsing 'date_start' column: {e}. Please ensure dates are in 'DD/MM/YY' format.")
        st.stop()

    st.subheader("⚙️ Running Meridien Model...")
    try:
        # Initialize Meridien model
        mmm = Meridien(data=df, target_column='Revenue', date_column='date_start')

        # Fit the model
        mmm.fit()

        st.subheader("🧠 Model Insights")
        st.write(f"**R² Score:** {mmm.r2:.4f}") # Display R2 score from fitted model
        st.write("**Channel Coefficients, Estimated ROI & Normalized Contribution:**")
        st.dataframe(mmm.model_results_df)

        st.subheader("📈 Budget Allocation & Optimization")

        plan_type = st.radio(
            "Select Plan Type:",
            ("Enter Fixed Budget", "Enter Revenue Target")
        )

        # Default value for input_value
        default_input_value = 100000.0 # Default for fixed budget
        if plan_type == "Enter Revenue Target":
            # Suggest a target slightly above current average revenue, if available
            if 'Revenue' in df.columns and not df['Revenue'].empty:
                default_input_value = df['Revenue'].mean() * 1.2
                if pd.isna(default_input_value) or default_input_value == 0:
                    default_input_value = 500000.0 # Fallback if mean is NaN or 0
            else:
                default_input_value = 500000.0 # Fallback if Revenue column is missing or empty

        input_value = st.number_input(
            f"Enter {plan_type.split(' ')[1]} Value:",
            min_value=0.0,
            value=float(default_input_value), # Ensure it's a float
            step=1000.0,
            format="%.2f"
        )

        if st.button("Generate Recommendations"):
            results = mmm.optimize(input_value, plan_type)

            st.subheader("📊 Recommended Budget Allocation")
            st.dataframe(results.optimized_spend)

            st.subheader("🌟 Scenario Plans")
            # Create tabs dynamically based on scenario names
            tabs = st.tabs(list(results.scenarios.keys()))
            for i, label in enumerate(results.scenarios.keys()): # Iterate over keys, not items
                with tabs[i]:
                    st.write(f"**{label}**")
                    st.dataframe(results.scenarios[label]) # Access scenario_df from results.scenarios
                    # Add a total spend for each scenario
                    st.write(f"**Total Recommended Spend:** ${results.scenarios[label]['recommended_spend'].sum():,.2f}")


            st.subheader("📉 Visuals")
            fig = results.plot_response_curves()
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Could not generate response curve plot. Ensure there is enough data and valid coefficients.")

    except Exception as e:
        st.error(f"❌ Error running model: {e}")
        st.exception(e) # Display full traceback for debugging
