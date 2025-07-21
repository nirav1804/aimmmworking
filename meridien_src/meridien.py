import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st # Used for st.info/success/warning messages within the class

class MeridienResults:
    """
    Holds the results of the Meridien model's optimization.
    """
    def __init__(self, optimized_spend_df, roi_df, r2, scenarios_dict, model, media_cols, df_original_for_plot):
        self.optimized_spend = optimized_spend_df
        self.roi_by_channel = roi_df
        self.r2_score = r2
        self.scenarios = scenarios_dict
        self._model = model # Store the fitted model for plotting
        self._media_cols = media_cols
        self._df_original_for_plot = df_original_for_plot # Store aggregated df for plotting ranges

    def plot_response_curves(self):
        """
        Generates Plotly response curves based on the fitted linear regression model.
        For linear regression, the response is directly proportional to the coefficient.
        """
        if not self._model or not self._media_cols:
            st.warning("Model not fitted or media columns not identified for plotting response curves.")
            return None

        fig = go.Figure()
        
        # Determine a reasonable max spend for the x-axis based on original aggregated data
        max_overall_spend = 0
        for col in self._media_cols:
            if col in self._df_original_for_plot.columns: # Check against the aggregated dataframe
                max_overall_spend = max(max_overall_spend, self._df_original_for_plot[col].max())
        
        # If no spend data found or max_overall_spend is 0, set a default
        if max_overall_spend == 0:
            max_overall_spend = 100000 # Default max spend if no actual spend data

        # Get coefficients from the fitted model
        # Ensure that the number of coefficients matches the number of media columns
        if len(self._model.coef_) != len(self._media_cols):
            st.error(f"Mismatch between number of coefficients ({len(self._model.coef_)}) and media columns ({len(self._media_cols)}). Cannot plot response curves.")
            return None
        
        coefs = self._model.coef_

        for i, channel in enumerate(self._media_cols):
            coef = coefs[i]
            
            # Generate a range of spend values for the channel, extending beyond observed max
            spend_range = np.linspace(0, max_overall_spend * 1.5, 100) # Extend 50% beyond max observed spend

            # For linear regression, the individual channel response is simply coef * spend.
            response = coef * spend_range

            fig.add_trace(go.Scatter(x=spend_range, y=response, mode='lines', name=f'{channel} Response'))

        fig.update_layout(
            title='Media Channel Response Curves (Linear Regression)',
            xaxis_title='Spend',
            yaxis_title='Revenue Response',
            hovermode="x unified",
            template="plotly_white" # Clean theme
        )
        return fig

class Meridien:
    """
    A simplified Marketing Mix Modeling (MMM) class based on linear regression.
    Automatically aggregates data by date for modeling.
    """
    def __init__(self, data, target_column, date_column):
        self.original_data = data.copy() # Keep a copy of original data
        self.target_column = target_column
        self.date_column = date_column
        self.model = None # Stores the fitted sklearn LinearRegression model
        self.model_results_df = None # Stores coefficients, ROI, etc.
        self.r2 = None # Stores the R2 score
        self.aggregated_data = None # Will store the preprocessed data

        # Basic validation on original data
        if self.target_column not in self.original_data.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in data.")
        if self.date_column not in self.original_data.columns:
            raise ValueError(f"Date column '{self.date_column}' not found in data.")
        if not pd.api.types.is_datetime64_any_dtype(self.original_data[self.date_column]):
            raise ValueError(f"Date column '{self.date_column}' is not in datetime format.")

        self._preprocess_data() # Call preprocessing during initialization

        # Identify media columns from the aggregated data
        # Assume columns containing '_spend' are media columns, excluding the target and date columns.
        self.media_cols = [col for col in self.aggregated_data.columns if '_spend' in col.lower() and col not in [self.target_column, self.date_column]]

        if not self.media_cols:
            st.warning("No media spend columns identified after aggregation. Please ensure your CSV has 'media_channel' and 'media_spend' columns, or columns containing '_spend' in their names after aggregation.")
            # If no media columns, we can't run the model, but let's not raise an error immediately
            # The fit method will handle this.

        st.info(f"Meridien initialized with target: '{target_column}', date: '{date_column}'. Data will be aggregated for modeling.")

    def _preprocess_data(self):
        """
        Aggregates the raw data by date, summing media spend for each channel
        and taking the mean of revenue (assuming revenue is constant per date).
        """
        # Ensure 'media_channel' and 'media_spend' columns exist for aggregation
        if 'media_channel' not in self.original_data.columns or 'media_spend' not in self.original_data.columns:
            raise ValueError("CSV must contain 'media_channel' and 'media_spend' columns for aggregation.")

        # Group by date and media_channel, sum media_spend
        # For 'Revenue', we assume it's constant per date_start and take the first value after grouping by date_start
        # It's crucial that Revenue is truly unique per date_start or consistently duplicated.
        
        # First, ensure Revenue is consistent per date_start
        revenue_check = self.original_data.groupby(self.date_column)[self.target_column].nunique()
        if (revenue_check > 1).any():
            st.warning(f"Multiple unique '{self.target_column}' values found for the same '{self.date_column}'. Taking the mean for aggregation.")

        # Aggregate spend by date_start and media_channel
        temp_df = self.original_data.groupby([self.date_column, 'media_channel'])['media_spend'].sum().reset_index()

        # Pivot media_channel to columns
        pivot_df = temp_df.pivot_table(index=self.date_column, columns='media_channel', values='media_spend', fill_value=0).reset_index()
        
        # Rename pivoted columns to include '_spend' suffix if not already present, for consistency
        pivot_df.columns.name = None # Remove the 'media_channel' column name
        new_cols = []
        for col in pivot_df.columns:
            if col != self.date_column and '_spend' not in col.lower():
                new_cols.append(f"{col}_spend")
            else:
                new_cols.append(col)
        pivot_df.columns = new_cols

        # Get unique revenue per date_start
        # Use .mean() to handle potential (but hopefully rare) slight variations or duplicates
        revenue_df = self.original_data.groupby(self.date_column)[self.target_column].mean().reset_index()

        # Merge pivoted spend data with revenue data
        self.aggregated_data = pd.merge(pivot_df, revenue_df, on=self.date_column, how='left')
        
        # Drop rows where target column is NaN after merge (e.g., if a date had spend but no revenue)
        self.aggregated_data.dropna(subset=[self.target_column], inplace=True)
        
        st.success("Data successfully preprocessed and aggregated by date!")
        st.dataframe(self.aggregated_data.head()) # Show head of aggregated data

    def _run_meridian_model(self):
        """
        Runs the linear regression model to determine media channel coefficients and R2 score
        on the aggregated data.
        """
        if self.aggregated_data is None or self.aggregated_data.empty:
            st.error("Aggregated data is empty. Cannot run model.")
            return

        # Re-identify media_cols based on aggregated data's columns, as they might have changed due to pivot
        self.media_cols = [col for col in self.aggregated_data.columns if '_spend' in col.lower() and col not in [self.target_column, self.date_column]]
        
        if not self.media_cols:
            st.warning("No media spend columns found in aggregated data. Cannot run model.")
            return

        X = self.aggregated_data[self.media_cols]
        y = self.aggregated_data[self.target_column]

        # Handle potential NaN values by filling with 0 (should be minimal after preprocessing)
        X = X.fillna(0)
        y = y.fillna(0) 

        # Check for zero variance in independent variables (media spend)
        if X.var().sum() == 0:
            st.warning("All media spend columns have zero variance. Linear regression cannot be performed as there's no variability in spend to explain revenue changes.")
            self.model = None
            self.model_results_df = pd.DataFrame({
                "media_channel": self.media_cols,
                "coefficient": 0.0,
                "estimated_roi": 0.0,
                "normalized_contribution": 0.0
            })
            self.r2 = 0.0
            return

        model = LinearRegression()
        model.fit(X, y)

        # Calculate R2 score
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)

        coefs = model.coef_

        # Calculate total contribution for normalization
        # Ensure coefs and X columns align
        contributions = pd.Series(0.0, index=self.media_cols)
        for i, col in enumerate(self.media_cols):
            contributions[col] = (X[col] * coefs[i]).sum()

        total_contribution = contributions.sum()

        if total_contribution == 0:
            st.warning("Total contribution is zero. Normalized contribution will be 0 for all channels.")
            normalized_contribution = pd.Series(0.0, index=self.media_cols)
        else:
            normalized_contribution = contributions / total_contribution

        # Calculate estimated_roi. Handle division by zero if mean spend is zero.
        estimated_roi = []
        for i, col in enumerate(self.media_cols):
            mean_spend = X[col].mean()
            if mean_spend != 0:
                estimated_roi.append(coefs[i] / mean_spend)
            else:
                estimated_roi.append(0.0) # ROI is 0 if average spend for the channel is 0

        results_df = pd.DataFrame({
            "media_channel": self.media_cols,
            "coefficient": coefs,
            "estimated_roi": estimated_roi,
            "normalized_contribution": normalized_contribution
        }).round(3)

        self.model = model
        self.model_results_df = results_df
        self.r2 = r2
        st.success("Meridien model fitted successfully!")

    def fit(self):
        """
        Public method to fit the Meridien model.
        """
        self._run_meridian_model()

    def _recommend_budget_allocation(self, input_value, plan_type):
        """
        Recommends budget allocation based on input value and plan type.
        """
        if self.model_results_df is None or self.model_results_df.empty:
            raise RuntimeError("Model results are not available. Please ensure the model was fitted successfully.")

        results = self.model_results_df.copy()

        # Calculate base recommended spend based on ROI
        roi_sum = results["estimated_roi"].sum()
        
        required_total_spend = 0.0
        if plan_type == "Enter Revenue Target":
            # Estimate required total spend to hit revenue target based on overall average ROI
            # This is a simplification for linear regression.
            # A more robust approach would involve iterating or using optimization solvers.
            avg_roi_overall = results["estimated_roi"].mean()
            if avg_roi_overall == 0:
                st.warning("Average ROI is zero. Cannot estimate required spend for revenue target.")
                required_total_spend = 0
            else:
                # This is a very simplistic way to estimate total spend for a revenue target
                # A more accurate way would involve the model's intercept and solving for total spend
                # For now, we'll use the average ROI as a proxy.
                required_total_spend = input_value / avg_roi_overall
        else: # "Enter Fixed Budget"
            required_total_spend = input_value

        if roi_sum == 0:
            st.warning("Sum of estimated ROI is zero. Cannot distribute budget based on ROI. Allocating zero to all channels.")
            results["recommended_spend"] = 0.0 # Assign 0 if sum of ROI is 0
        else:
            results["recommended_spend"] = (results["estimated_roi"] / roi_sum) * required_total_spend

        # --- Scenario Plans ---
        scenarios = {
            "🟡 Balanced Plan": results.copy(), # This is the base ROI-weighted allocation
            "🟢 Aggressive Plan": results.copy(),
            "🔵 Conservative Plan": results.copy()
        }

        # Aggressive Plan: Distribute based on square of ROI (favors higher ROI channels more)
        roi_squared_sum = (results["estimated_roi"] ** 2).sum()
        if roi_squared_sum != 0:
            scenarios["🟢 Aggressive Plan"]["recommended_spend"] = (results["estimated_roi"] ** 2 / roi_squared_sum) * scenarios["🟢 Aggressive Plan"]["recommended_spend"].sum()
        else:
            scenarios["🟢 Aggressive Plan"]["recommended_spend"] = 0.0

        # Conservative Plan: Only allocate to channels with ROI > average ROI
        mean_roi = results["estimated_roi"].mean()
        conservative_channels_df = results[results["estimated_roi"] > mean_roi]
        conservative_roi_sum = conservative_channels_df["estimated_roi"].sum()

        if conservative_roi_sum != 0 and not conservative_channels_df.empty:
            # Distribute the *total* input_value (or required_total_spend) among conservative channels
            total_spend_for_conservative = required_total_spend # Use the calculated total spend for the plan
            scenarios["🔵 Conservative Plan"]["recommended_spend"] = 0.0 # Start with all zero
            for index, row in conservative_channels_df.iterrows():
                channel = row["media_channel"]
                # Calculate allocation for this channel
                allocation = (row["estimated_roi"] / conservative_roi_sum) * total_spend_for_conservative
                # Update the recommended_spend in the scenario dataframe for this channel
                scenarios["🔵 Conservative Plan"].loc[scenarios["🔵 Conservative Plan"]["media_channel"] == channel, "recommended_spend"] = allocation
        else:
            st.warning("No channels with ROI greater than average for Conservative Plan. Allocating zero to all channels in Conservative Plan.")
            scenarios["🔵 Conservative Plan"]["recommended_spend"] = 0.0 # No allocation if no conservative channels or sum is zero

        # Ensure all scenario dataframes have all media channels, even if spend is 0
        for label, df_scenario in scenarios.items():
            # Add missing channels with 0 spend, if any were filtered out (e.g., in conservative)
            existing_channels = df_scenario["media_channel"].tolist()
            missing_channels = [mc for mc in self.media_cols if mc not in existing_channels]
            
            if missing_channels:
                missing_df = pd.DataFrame({
                    "media_channel": missing_channels,
                    "coefficient": 0.0, # Default to 0
                    "estimated_roi": 0.0, # Default to 0
                    "normalized_contribution": 0.0, # Default to 0
                    "recommended_spend": 0.0
                })
                scenarios[label] = pd.concat([df_scenario, missing_df], ignore_index=True)
            
            # Sort to maintain consistent order across scenarios
            scenarios[label] = scenarios[label].set_index("media_channel").loc[self.media_cols].reset_index()
            scenarios[label]["recommended_spend"] = scenarios[label]["recommended_spend"].round(2)
            scenarios[label]["estimated_roi"] = scenarios[label]["estimated_roi"].round(3) # Keep ROI consistent
            scenarios[label]["coefficient"] = scenarios[label]["coefficient"].round(3) # Keep coefficient consistent
            scenarios[label]["normalized_contribution"] = scenarios[label]["normalized_contribution"].round(3) # Keep contribution consistent


        results["recommended_spend"] = results["recommended_spend"].round(2)
        return results[["media_channel", "estimated_roi", "recommended_spend"]], scenarios

    def optimize(self, input_value, plan_type):
        """
        Optimizes the media plan based on input value and plan type,
        and returns a MeridienResults object.
        """
        # Ensure the model is fitted before optimization
        if self.model is None:
            st.error("Model has not been fitted. Please run .fit() first.")
            return None

        optimized_spend_df, scenarios_dict = self._recommend_budget_allocation(input_value, plan_type)
        # Pass all necessary data to MeridienResults, including the aggregated data for plotting ranges
        return MeridienResults(optimized_spend_df, self.model_results_df[['media_channel', 'estimated_roi', 'coefficient', 'normalized_contribution']], self.r2, scenarios_dict, self.model, self.media_cols, self.aggregated_data)

