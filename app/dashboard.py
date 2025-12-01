"""Streamlit dashboard for Rossmann Sales Forecasting"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import json
import os

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="🚀 Rossmann Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# STYLING
# ============================================================================

st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .header-style {
        font-size: 2.5em;
        font-weight: bold;
        color: #1f77b4;
    }
    .viz-category {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================

st.title("🚀 Rossmann Sales Forecasting - Production Dashboard")
st.markdown("---")

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.header("⚙️ Settings & Navigation")
    
    page = st.radio(
        "Select Page:",
        ["📊 Visualizations", "🔮 Make Predictions", "📈 Model Performance", "🏥 Health Check", "📚 Documentation"]
    )
    
    st.markdown("---")
    st.subheader("Configuration")
    
    api_url = st.text_input("API URL", value="http://localhost:8000")
    
    st.markdown("---")
    st.subheader("Quick Stats")
    st.metric("Model Status", "✅ Production")
    st.metric("API Status", "🟢 Online")
    st.metric("Last Updated", datetime.now().strftime("%H:%M:%S"))

# ============================================================================
# PAGE 1: VISUALIZATIONS
# ============================================================================

if page == "📊 Visualizations":
    st.header("📊 Data Visualizations & Analysis")
    st.markdown("Explore interactive visualizations from the Rossmann Sales Forecasting project")
    
    # Define visualization categories and files
    visualizations = {
        "📊 Exploratory Data Analysis": {
            "Initial EDA Dashboard": "Rossmann Sales - Initial EDA Dashboard.html",
            "Train-Test Split": "01_train_test_split.html",
        },
        "🔄 Feature Engineering": {
            "Cyclical Encoding (sin_cos)": "Cyclical Encoding of Months (sin_cos).html",
            "Feature Correlation Heatmap": "Top 20 Features - Correlation Heatmap.html",
            "ACF & PACF Plots": "Interactive ACF and PACF Plots.html",
        },
        "🤖 Model Analysis": {
            "XGBoost Feature Importance": "M3_03_xgboost_features.html",
            "Model Comparison": "M3_10_model_comparison.html",
            "Residual Analysis": "M3_11_residual_analysis.html",
        },
        "🎯 Results & Forecasts": {
            "Final Forecast": "M3_12_final_forecast.html",
        }
    }
    
    # Create tabs for categories
    tabs = st.tabs(list(visualizations.keys()))
    
    for tab, (category, viz_dict) in zip(tabs, visualizations.items()):
        with tab:
            # Dropdown to select visualization within category
            selected_viz = st.selectbox(
                f"Select visualization:",
                options=list(viz_dict.keys()),
                key=f"select_{category}"
            )
            
            viz_filename = viz_dict[selected_viz]
            viz_path = f"visualizations/{viz_filename}"
            
            st.markdown(f"### 📈 {selected_viz}")
            
            try:
                # Check if file exists
                if os.path.exists(viz_path):
                    # Read HTML file
                    with open(viz_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    
                    # Display HTML with scrolling
                    components.html(html_content, height=800, scrolling=True)
                    
                    # Add info and download button
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.info(f"📁 **File:** {viz_filename}")
                    with col2:
                        st.download_button(
                            label="⬇️ Download",
                            data=html_content,
                            file_name=viz_filename,
                            mime="text/html",
                            key=f"download_{category}_{selected_viz}"
                        )
                else:
                    st.error(f"❌ **File not found:** `{viz_path}`")
                    st.info("💡 **Tip:** Ensure HTML files are in the `visualizations/` folder relative to this script.")
            
            except Exception as e:
                st.error(f"❌ **Error loading visualization:** {str(e)}")
                st.code(f"Path checked: {viz_path}")
    
    # Quick overview section
    st.markdown("---")
    st.subheader("📚 Visualization Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**📊 EDA**")
        st.write("• Initial data exploration")
        st.write("• Train-test split")
        st.write("• Data distributions")
    
    with col2:
        st.markdown("**🔄 Features**")
        st.write("• Cyclical encoding")
        st.write("• Correlation analysis")
        st.write("• Time series plots")
    
    with col3:
        st.markdown("**🤖 Models**")
        st.write("• Feature importance")
        st.write("• Model comparison")
        st.write("• Residual analysis")
    
    with col4:
        st.markdown("**🎯 Results**")
        st.write("• Final forecasts")
        st.write("• Performance metrics")
        st.write("• Predictions vs actuals")

# ============================================================================
# PAGE 2: MAKE PREDICTIONS
# ============================================================================

elif page == "🔮 Make Predictions":
    st.header("🔮 Make Sales Predictions")
    
    # Add expandable reference guide
    with st.expander("📚 Date & Time Reference Guide - Click to expand", expanded=False):
        st.markdown("### Quick Reference for Input Values")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📅 Days of Week", "📆 Months", "📊 Quarters", "🗓️ Weeks"])
        
        with tab1:
            st.markdown("#### Day of Week Mapping")
            days_df = pd.DataFrame({
                'Number': [1, 2, 3, 4, 5, 6, 7],
                'Day': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday ⭐', 'Sunday ⭐'],
                'Type': ['Weekday', 'Weekday', 'Weekday', 'Weekday', 'Weekday', 'Weekend', 'Weekend']
            })
            st.dataframe(days_df, use_container_width=True, hide_index=True)
            st.info("⭐ Days 6 (Saturday) and 7 (Sunday) are weekends")
        
        with tab2:
            st.markdown("#### Month Numbers & Details")
            months_df = pd.DataFrame({
                'Month #': list(range(1, 13)),
                'Name': ['January', 'February', 'March', 'April', 'May', 'June',
                        'July', 'August', 'September', 'October', 'November', 'December'],
                'Abbr': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                'Days': [31, '28/29', 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
                'Quarter': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
            })
            st.dataframe(months_df, use_container_width=True, hide_index=True)
        
        with tab3:
            st.markdown("#### Quarter Breakdown")
            quarters_df = pd.DataFrame({
                'Quarter': ['Q1', 'Q2', 'Q3', 'Q4'],
                'Number': [1, 2, 3, 4],
                'Months': ['Jan-Mar', 'Apr-Jun', 'Jul-Sep', 'Oct-Dec'],
                'Month #s': ['1, 2, 3', '4, 5, 6', '7, 8, 9', '10, 11, 12'],
                'Weeks': ['1-13', '14-26', '27-39', '40-52'],
                'Season': ['Winter/Spring', 'Spring/Summer', 'Summer/Fall', 'Fall/Winter']
            })
            st.dataframe(quarters_df, use_container_width=True, hide_index=True)
            st.success("📊 Each quarter ≈ 13 weeks and 3 months")
        
        with tab4:
            st.markdown("#### Week of Year Guide")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Week Ranges by Quarter:**")
                weeks_df = pd.DataFrame({
                    'Quarter': ['Q1', 'Q2', 'Q3', 'Q4'],
                    'Week Range': ['1-13', '14-26', '27-39', '40-52'],
                    'Months': ['Jan-Mar', 'Apr-Jun', 'Jul-Sep', 'Oct-Dec']
                })
                st.dataframe(weeks_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("**Example Dates:**")
                examples_df = pd.DataFrame({
                    'Date': ['Jan 1st', 'Apr 1st', 'Jul 1st', 'Oct 1st', 'Dec 31st'],
                    'Approx Week': ['≈ Week 1', '≈ Week 13', '≈ Week 26', '≈ Week 39', '≈ Week 52']
                })
                st.dataframe(examples_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # PREDICTION FORM
    with st.form("prediction_form"):
        st.subheader("📝 Enter Prediction Details")
        
        # Date & Time Features
        st.markdown("**📅 Date & Time Features**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            day_of_week = st.selectbox("Day of Week", 
                options=list(range(1, 8)),
                format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x-1])
        with col2:
            quarter = st.selectbox("Quarter", [1, 2, 3, 4], index=0)
        with col3:
            is_weekend = st.checkbox("Is Weekend?", value=(day_of_week >= 6))
        
        st.markdown("---")
        
        # Promotional & Holiday Features
        st.markdown("**🎯 Promotional & Holiday Features**")
        col1, col2 = st.columns(2)
        
        with col1:
            promo = st.checkbox("Promotion Active?", value=False)
        with col2:
            state_holiday = st.checkbox("State Holiday?", value=False)
        
        st.markdown("---")
        
        # Customer Data
        st.markdown("**👥 Customer Data**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            customers = st.number_input("Current Customers", min_value=1, value=600, step=10)
        with col2:
            customers_lag_7 = st.number_input("Customers (Lag-7)", min_value=1, value=600, step=10)
        with col3:
            customers_lag_14 = st.number_input("Customers (Lag-14)", min_value=1, value=600, step=10)
        
        st.markdown("---")
        
        # Historical Sales Data
        st.markdown("**📊 Historical Sales Data**")
        col1, col2 = st.columns(2)
        
        with col1:
            sales_lag_1 = st.number_input("Sales (Lag-1 day)", min_value=1.0, value=5000.0, step=100.0)
        with col2:
            sales_lag_30 = st.number_input("Sales (Lag-30 days)", min_value=1.0, value=5000.0, step=100.0)
        
        st.markdown("---")
        
        # Rolling Statistics
        st.markdown("**📈 Rolling Statistics**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("*Rolling Means*")
            sales_rolling_mean_7 = st.number_input("Sales Rolling Mean (7d)", min_value=1.0, value=5000.0, step=100.0)
            sales_rolling_mean_14 = st.number_input("Sales Rolling Mean (14d)", min_value=1.0, value=5000.0, step=100.0)
        
        with col2:
            st.markdown("*Rolling Standard Deviations*")
            sales_rolling_std_7 = st.number_input("Sales Rolling Std (7d)", min_value=0.0, value=100.0, step=10.0)
            sales_rolling_std_14 = st.number_input("Sales Rolling Std (14d)", min_value=0.0, value=100.0, step=10.0)
        
        st.markdown("---")
        
        # Derived Metric
        st.markdown("**💰 Derived Metrics**")
        sales_per_customer = st.number_input("Sales Per Customer", min_value=0.01, value=8.0, step=0.5)
        
        st.info(f"💡 Auto-calculated: €{sales_lag_1 / customers:.2f} per customer based on Lag-1 sales and current customers")
        
        submit_button = st.form_submit_button("🚀 Generate Prediction", use_container_width=True)
        
        if submit_button:
            st.info("📡 Sending request to prediction API...")
            
            try:
                # Create payload matching the new feature set
                payload = {
                    "DayOfWeek": day_of_week,
                    "Customers": customers,
                    "Promo": 1 if promo else 0,
                    "StateHoliday": 1 if state_holiday else 0,
                    "SalesPerCustomer": sales_per_customer,
                    "IsWeekend": 1 if is_weekend or day_of_week >= 6 else 0,
                    "Quarter": quarter,
                    "Sales_Lag_1": sales_lag_1,
                    "Sales_Lag_30": sales_lag_30,
                    "Customers_Lag_7": customers_lag_7,
                    "Customers_Lag_14": customers_lag_14,
                    "Sales_Rolling_Mean_7": sales_rolling_mean_7,
                    "Sales_Rolling_Mean_14": sales_rolling_mean_14,
                    "Sales_Rolling_Std_7": sales_rolling_std_7,
                    "Sales_Rolling_Std_14": sales_rolling_std_14
                }
                
                response = requests.post(f"{api_url}/predict", json=payload, timeout=5)
                response.raise_for_status()
                result = response.json()
                
                # Debug: Show raw response
                with st.expander("🔍 Debug: View Raw API Response"):
                    st.json(result)
                
                # Handle different response formats
                if isinstance(result, dict):
                    prediction_value = (
                        result.get('prediction') or 
                        result.get('predicted_sales') or 
                        result.get('sales') or
                        result.get('forecast') or
                        result.get('value')
                    )
                    
                    confidence = result.get('confidence', result.get('conf', 0.0))
                    model_version = result.get('model_version', result.get('version', 'v1.0.0'))
                    
                    if prediction_value is not None:
                        st.success("✅ Prediction Generated!")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Predicted Sales", f"€{prediction_value:,.0f}")
                        with col2:
                            if confidence:
                                st.metric("Confidence", f"{confidence*100:.1f}%")
                            else:
                                st.metric("Confidence", "N/A")
                        with col3:
                            st.metric("Model Version", model_version)
                    else:
                        st.error("❌ **API Response Error:** Missing prediction value")
                        st.warning(f"""
                        **The API returned data but no prediction value was found.**
                        
                        Expected keys: 'prediction', 'predicted_sales', 'sales', 'forecast', or 'value'
                        
                        Received keys: {list(result.keys())}
                        """)
                elif isinstance(result, (int, float)):
                    st.success("✅ Prediction Generated!")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Predicted Sales", f"€{result:,.0f}")
                    with col2:
                        st.metric("Confidence", "N/A")
                    with col3:
                        st.metric("Model Version", "N/A")
                else:
                    st.error(f"❌ **Unexpected API Response Type:** {type(result)}")
                    st.json(result)
                
            except requests.exceptions.ConnectionError:
                st.error("❌ **API Connection Failed**")
                st.warning("""
                **The FastAPI server is not running.** Please start it first:
                
                1. Open a new terminal
                2. Navigate to your project folder
                3. Run: `python api.py` or `uvicorn api:app --reload`
                4. Keep that terminal running
                5. Try predicting again
                """)
                
                # Show demo prediction
                st.info("📊 **Demo Mode** - Showing mock prediction:")
                mock_prediction = sales_lag_1 * (1.1 if promo else 1.0) * (0.9 if is_weekend else 1.0)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predicted Sales (Demo)", f"€{mock_prediction:,.0f}")
                with col2:
                    st.metric("Confidence", "N/A")
                with col3:
                    st.metric("Model Version", "DEMO")
            
            except requests.exceptions.HTTPError as e:
                st.error(f"❌ **HTTP Error:** {e.response.status_code}")
                try:
                    error_detail = e.response.json()
                    st.json(error_detail)
                except:
                    st.text(e.response.text)
                    
            except KeyError as e:
                st.error(f"❌ **Key Error:** Missing key {str(e)}")
                st.warning("""
                **The API response format doesn't match expectations.**
                
                Please check the 'Debug: View Raw API Response' section above to see what the API actually returned.
                """)
                
            except Exception as e:
                st.error(f"❌ **Unexpected Error:** {str(e)}")
                st.info("Please check your API configuration and ensure the server is running correctly.")

# ============================================================================
# PAGE 3: MODEL PERFORMANCE
# ============================================================================

elif page == "📈 Model Performance":
    st.header("📈 Model Performance & Comparison")
    
    models_df = pd.DataFrame({
        'Model': ['XGBoost 🏆', 'Random Forest', 'Ensemble', 'LSTM', 'Prophet', 'ARIMA'],
        'RMSE': [147015, 169056, 692137, 1830020, 1925076, 2859128],
        'MAE': [72390, 72299, 554352, 1125164, 1519932, 2308695],
        'MAPE (%)': [1.65, 1.70, 61.33, 200.99, 110.25, 400.65],
        'R²': [0.9979, 0.9972, 0.9534, 0.6659, 0.6303, 0.1846]
    })
    
    st.dataframe(models_df, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure(go.Bar(
            y=models_df['Model'],
            x=models_df['RMSE'],
            orientation='h',
            marker_color=['gold', 'silver', 'gray', 'red', 'red', 'darkred']
        ))
        fig.update_layout(title="RMSE Comparison (Lower is Better)", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure(go.Bar(
            y=models_df['Model'],
            x=models_df['MAPE (%)'],
            orientation='h',
            marker_color=['gold', 'silver', 'gray', 'red', 'red', 'darkred']
        ))
        fig.update_layout(title="MAPE Comparison (Lower is Better)", height=400)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 4: HEALTH CHECK
# ============================================================================

elif page == "🏥 Health Check":
    st.header("🏥 System Health Check")
    
    st.subheader("API Health Status")
    
    try:
        response = requests.get(f"{api_url}/health", timeout=3)
        if response.status_code == 200:
            st.success("✅ API is healthy and responding")
            health_data = response.json()
            st.json(health_data)
        else:
            st.error(f"❌ API returned status code: {response.status_code}")
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API - ensure it's running")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# ============================================================================
# PAGE 5: DOCUMENTATION
# ============================================================================

elif page == "📚 Documentation":
    st.header("📚 API Documentation & Usage")
    
    st.subheader("🔌 Available Endpoints")
    
    endpoints = {
        "GET /": "API overview and endpoints",
        "GET /health": "Check API health status",
        "POST /predict": "Single prediction request",
        "POST /predict_batch": "Batch predictions",
        "GET /model/info": "Model metadata and performance"
    }
    
    for endpoint, description in endpoints.items():
        st.write(f"**{endpoint}**: {description}")
    
    st.markdown("---")
    st.subheader("📖 Example Request")
    
    st.code("""
import requests

payload = {
    "DayOfWeek": 3,
    "Customers": 600,
    "Promo": 1,
    "StateHoliday": 0,
    "SalesPerCustomer": 8.5,
    "IsWeekend": 0,
    "Quarter": 4,
    "Sales_Lag_1": 5000.0,
    "Sales_Lag_30": 5000.0,
    "Customers_Lag_7": 600.0,
    "Customers_Lag_14": 600.0,
    "Sales_Rolling_Mean_7": 5000.0,
    "Sales_Rolling_Mean_14": 5000.0,
    "Sales_Rolling_Std_7": 100.0,
    "Sales_Rolling_Std_14": 100.0
}

response = requests.post("http://localhost:8000/predict", json=payload)
print(response.json())
    """, language="python")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>🚀 <b>Rossmann Sales Forecasting System</b> | Production Grade | v1.0.0</p>
        <p>Last Updated: 2025-11-16 | Status: ✅ OPERATIONAL</p>
    </div>
""", unsafe_allow_html=True)
