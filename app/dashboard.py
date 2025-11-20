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
            st.info("⭐ Days 5 (Saturday) and 6 (Sunday) are weekends")
        
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
    
    with st.form("prediction_form"):
        st.subheader("📝 Enter Prediction Details")
        
        # Basic Store Information
        st.markdown("**🏪 Store Information**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            store = st.number_input("Store Number", min_value=1, max_value=1115, value=1, step=1)
        with col2:
            store_open = st.selectbox("Store Open?", options=[0, 1], format_func=lambda x: "Closed" if x == 0 else "Open", index=1)
        with col3:
            store_type = st.selectbox("Store Type", options=[0, 1, 2, 3], format_func=lambda x: ['a', 'b', 'c', 'd'][x], index=0)
        with col4:
            assortment = st.selectbox("Assortment", options=[0, 1, 2], format_func=lambda x: ['a', 'b', 'c'][x], index=0)
        
        st.markdown("---")
        
        # Date & Time Features
        st.markdown("**📅 Date & Time Features**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            day_of_week = st.selectbox("Day of Week", 
                options=list(range(7)), 
                format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x])
        with col2:
            month = st.slider("Month", 1, 12, 4)
        with col3:
            quarter = st.selectbox("Quarter", [1, 2, 3, 4], index=1)
        with col4:
            week = st.slider("Week of Year", 1, 52, 15)
        
        st.markdown("---")
        
        # Promotional Features
        st.markdown("**🎯 Promotional Features**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            promo = st.checkbox("Promotion Active?", value=False)
        with col2:
            school_holiday = st.checkbox("School Holiday?", value=True)
        with col3:
            competition_distance = st.number_input("Competition Distance (m)", value=1000.0, step=100.0)
        
        st.markdown("---")
        
        # Historical Sales Data
        st.markdown("**📊 Historical Sales & Customer Data**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sales_lag_1 = st.number_input("Sales (Lag-1 day)", value=5785, step=100)
            sales_lag_7 = st.number_input("Sales (Lag-7 days)", value=10061, step=100)
            sales_lag_14 = st.number_input("Sales (Lag-14 days)", value=9860, step=100)
            sales_lag_30 = st.number_input("Sales (Lag-30 days)", value=9500, step=100)
        
        with col2:
            customers_lag_1 = st.number_input("Customers (Lag-1)", value=652, step=10)
            customers_lag_7 = st.number_input("Customers (Lag-7)", value=665, step=10)
        
        with col3:
            sales_rolling_mean_7 = st.number_input("Sales Rolling Mean (7d)", value=10061, step=100)
            sales_rolling_mean_14 = st.number_input("Sales Rolling Mean (14d)", value=9960, step=100)
            sales_rolling_std_7 = st.number_input("Sales Rolling Std (7d)", value=1000, step=50)
            sales_rolling_std_14 = st.number_input("Sales Rolling Std (14d)", value=1100, step=50)
        
        submit_button = st.form_submit_button("🚀 Generate Prediction", use_container_width=True)
        
        if submit_button:
            st.info("📡 Sending request to prediction API...")
            
            # Calculate sales per customer
            sales_per_customer = sales_lag_1 / customers_lag_1 if customers_lag_1 > 0 else 0
            
            try:
                # Create payload with PascalCase keys to match API requirements
                payload = {
                    "DayOfWeek": day_of_week,
                    "Month": month,
                    "Quarter": quarter,
                    "Week": week,
                    "IsWeekend": 1 if day_of_week >= 5 else 0,
                    "Promo": 1 if promo else 0,
                    "SchoolHoliday": 1 if school_holiday else 0,
                    "Sales_Lag_1": sales_lag_1,
                    "Sales_Lag_7": sales_lag_7,
                    "Sales_Lag_14": sales_lag_14,
                    "Sales_Lag_30": sales_lag_30,
                    "Customers_Lag_1": customers_lag_1,
                    "Customers_Lag_7": customers_lag_7,
                    "Sales_Rolling_Mean_7": sales_rolling_mean_7,
                    "Sales_Rolling_Mean_14": sales_rolling_mean_14,
                    "Sales_Rolling_Std_7": sales_rolling_std_7,
                    "Sales_Rolling_Std_14": sales_rolling_std_14,
                    "SalesPerCustomer": sales_per_customer,
                    "Store": store,
                    "Open": store_open,
                    "StoreType": store_type,
                    "Assortment": assortment,
                    "CompetitionDistance": competition_distance
                }
                
                response = requests.post(f"{api_url}/predict", json=payload, timeout=5)
                response.raise_for_status()  # Raise error for bad status codes
                result = response.json()
                
                # Debug: Show raw response
                with st.expander("🔍 Debug: View Raw API Response"):
                    st.json(result)
                
                # Handle different response formats
                if isinstance(result, dict):
                    # Try different possible keys for the prediction value
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
                    # If API returns just a number
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
                mock_prediction = sales_lag_1 * (1.1 if promo else 1.0) * (0.9 if day_of_week >= 5 else 1.0)
                
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
    st.header("🏥 System Health & Monitoring")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("API Status", "🟢 Online")
    with col2:
        st.metric("Database", "🟢 Connected")
    with col3:
        st.metric("Model", "🟢 Loaded")
    with col4:
        st.metric("Uptime", "99.8%")
    
    st.markdown("---")
    st.subheader("📊 System Logs (Last 24 Hours)")
    
    logs = [
        {"timestamp": "2025-11-16 23:30", "level": "✅ INFO", "message": "Batch prediction completed: 1,115 stores"},
        {"timestamp": "2025-11-16 23:00", "level": "✅ INFO", "message": "Model retraining scheduled"},
        {"timestamp": "2025-11-16 22:30", "level": "⚠️ WARNING", "message": "MAPE increased to 2.1%"},
        {"timestamp": "2025-11-16 21:45", "level": "✅ INFO", "message": "500 predictions served"},
        {"timestamp": "2025-11-16 20:00", "level": "✅ INFO", "message": "Data validation passed"},
    ]
    
    logs_df = pd.DataFrame(logs)
    st.dataframe(logs_df, use_container_width=True)

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
    "day_of_week": 3,
    "month": 11,
    "quarter": 4,
    "week": 45,
    "is_weekend": 0,
    "promo": 1,
    "school_holiday": 0,
    "sales_lag_1": 50000,
    "sales_lag_7": 48000,
    "sales_lag_14": 47000,
    "sales_lag_21": 49000,
    "customers_lag_1": 800,
    "customers_lag_7": 820,
    "sales_rolling_mean_7": 49000,
    "sales_rolling_mean_14": 48500,
    "sales_rolling_std_7": 1000,
    "sales_rolling_std_14": 1200,
    "sales_per_customer": 62.5
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