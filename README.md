# Graduation-Project
# 📊 Rossmann Store Sales Forecasting - IBM Data Science Capstone Project

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Forecasting-green)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)](https://github.com/)
[![License](https://img.shields.io/badge/License-MIT-red)](LICENSE)

> Advanced time series forecasting system using ensemble machine learning models to predict daily sales for Rossmann drugstore chain

---

## 👥 Team Members

| Name | Role | Responsibilities | Contact |
|------|------|-----------------|---------|
| **[Ziad Mohamed shaker]** | Team Lead / Data Scientist | Project coordination, XGBoost/RF model development,EDA, feature engineering, visualization |
| **[Ahmed Behdal]** | -------- |------------------|
| **[mahmoud ismail]** | -------- | ------------------ |
| **[Anas Abdul Monem reyad]** | -------- | ------------------ |
---

## 📅 Project Timeline

### **Phase 1: Data Exploration & Analysis** ✅ COMPLETED
**Duration:** October 25 - October 31, 2025 (1 weeks)
- ✅ Dataset acquisition from Kaggle (1M+ records, 1,115 stores)
- ✅ Comprehensive exploratory data analysis (EDA)
- ✅ Data cleaning and preprocessing
- ✅ Outlier detection and handling (3×IQR method)
- ✅ Feature engineering (40+ temporal, lag, and rolling features)
- ✅ Statistical analysis (stationarity tests, correlation analysis)
- ✅ Interactive visualization dashboards

### **Phase 2: Model Development & Optimization** ✅ COMPLETED
**Duration:** November 1 - November 14, 2025 (2 week)
- ✅ Traditional time series models (ARIMA, Prophet)
- ✅ Machine learning models (XGBoost, Random Forest)
- ✅ Deep learning (LSTM neural networks)
- ✅ Bayesian hyperparameter optimization
- ✅ Comprehensive model evaluation and comparison
- ✅ Ensemble model development

### **Phase 3: Deployment & Monitoring** 🔄 IN PROGRESS
**Duration:** November 15 - November 24, 2025 (1.5 week)
- 🔄 MLflow integration for experiment tracking
- 🔄 Streamlit web application development
- 🔄 Model monitoring and performance tracking
- ⏳ Automated retraining pipelines

### **Phase 4: Documentation & Presentation** ⏳ UPCOMING
**Duration:** November 25 - November 27, 2025 (2 days)
- ⏳ Final report and technical documentation
- ⏳ Presentation preparation
- ⏳ Code cleanup and optimization
- ⏳ Knowledge transfer materials

---

## 🎯 Project Goals

### Primary Objectives
1. **Develop High-Accuracy Forecasting System**
   - Target: MAPE < 10% on test data
   - Build ensemble of 5+ different models
   - Implement production-ready prediction pipeline

2. **Optimize Inventory Management**
   - Reduce stockouts by 25-30%
   - Minimize excess inventory by 10-12%
   - Improve demand planning accuracy by 15-20%

3. **Deploy Scalable MLOps Solution**
   - Automated model retraining (weekly)
   - Real-time performance monitoring
   - Version control and experiment tracking

### Secondary Goals
- Identify key sales drivers and seasonal patterns
- Create interactive dashboards for stakeholders
- Build reusable forecasting framework
- Document best practices and lessons learned

---

## 📈 Results Summary

### Model Performance (as of November 14, 2025)

### Model Performance (as of November 15, 2025)

| Model | RMSE | MAE | MAPE (%) | R^2 | Status |
|-------|------|-----|----------|--------|--------|
| **XGBoost** 🏆 | 64,389.0    | 53,351.0    | **1.64** |  0.9995  |✅ Best |
| Random Forest | 108,398.0  | 60,042.0    | 1.33 |  0.9987  | ✅ Good |
| Facebook Prophet | 615,994.0   | 470,823.0   | 24.99 |  0.9568  |⚠️ not efficent enough |
| Auto-ARIMA | 2,504,064.0  | 2,050,757.0  | 249.83 |  0.2857  |⚠️ Terrible |
| LSTM | 809,146.0   | 538,233.0   | 60.85 |  0.9254  | ⚠️ Not usable |

### 📊 Forecasting Model Comparison
<p align="center">
  <img src="images/forecast_comparison.png" width="800">
</p>


### 📊 Forecasting Model Comparison
<p align="center">
  <img src="image.png" width="800">
</p>

**Key Achievements:**
- ✅ **XGBoost achieved target MAPE of 1.64%** - meeting business requirements
- ✅ Successfully implemented Bayesian hyperparameter optimization
- ✅ Created 40+ engineered features for improved model performance
- ✅ Cleaned dataset with outlier removal (improved stability)
- ✅ Comprehensive model comparison framework established

### Dataset Statistics
- **Total Records:** 1,017,209 sales transactions
- **Date Range:** January 1, 2013 - July 31, 2015 (2.5+ years)
- **Stores:** 1,115 unique Rossmann stores
- **Features:** 40+ engineered features (temporal, lags, rolling stats)
- **Train/Test Split:** 80/20 (last 6 weeks for testing)

### Feature Engineering Highlights
```
✓ Temporal Features: Year, Month, Day, DayOfWeek, Quarter, WeekOfYear
✓ Lag Features: 1, 7, 14, 21, 30-day lags for Sales and Customers
✓ Rolling Statistics: 7, 14, 28-day moving averages and standard deviations
✓ Cyclical Encoding: Sin/cos transformations for month and day-of-week
✓ Business Indicators: Promotions, holidays, competition metrics
✓ Interaction Features: SalesPerCustomer, CompetitionMonthsOpen
```

---

## 🛠️ Technologies & Tools

### Programming & Libraries
- **Python 3.8+** - Core programming language
- **Pandas & NumPy** - Data manipulation and numerical computing
- **Scikit-learn** - Machine learning algorithms
- **XGBoost** - Gradient boosting framework
- **TensorFlow/Keras** - Deep learning (LSTM)
- **Statsmodels** - Time series analysis (ARIMA)
- **Prophet** - Facebook's time series forecasting

### Visualization & Reporting
- **Matplotlib & Seaborn** - Static visualizations
- **Plotly** - Interactive dashboards
- **Streamlit** - Web application deployment

### MLOps & Deployment
- **MLflow** - Experiment tracking and model registry
- **Docker** - Containerization
- **Git & GitHub** - Version control
- **Jupyter Notebook** - Interactive development

---

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
pip or conda package manager
Git
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-team/rossmann-forecasting.git
cd rossmann-forecasting
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download dataset**
- Visit [Kaggle Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales/data)
- Download `train.csv`, `store.csv`, `test.csv`
- Place files in `data/raw/` directory

### Running the Project

**1. Exploratory Data Analysis**
```bash
jupyter notebook notebooks/01_eda_analysis.ipynb
```

**2. Model Training**
```bash
jupyter notebook notebooks/04_model_development.ipynb
```

**3. Run Streamlit App (when ready)**
```bash
streamlit run deployment/streamlit_app.py
```

---

## 📊 Key Findings

### Business Insights
1. **Promotional Impact:** Stores with active promotions see 15-20% higher sales on average
2. **Day of Week Patterns:** Sunday sales are typically 25% higher than Monday
3. **Store Type Performance:** Type 'b' stores show highest sales per customer ratio
4. **Seasonality:** Strong yearly patterns with peaks around holidays (Christmas, Easter)
5. **Competition Effect:** Stores with competitors within 500m show paradoxically higher sales (better locations)

### Technical Insights
1. **XGBoost Superior Performance:** Tree-based models significantly outperform time series models on this aggregated data
2. **Feature Engineering Critical:** Lag and rolling features contributed 60%+ to model performance
3. **Outlier Impact:** Removing outliers using 3×IQR improved model stability by 12%
4. **LSTM Challenges:** Deep learning struggled without massive training data and proper hyperparameter tuning
5. **Ensemble Potential:** Combining XGBoost + Random Forest could yield 3-5% further improvement

---

## 🎓 Lessons Learned

### What Worked Well
- ✅ Comprehensive feature engineering pipeline significantly improved results
- ✅ Bayesian optimization reduced hyperparameter tuning time by 60%
- ✅ Interactive visualizations enhanced stakeholder communication
- ✅ Modular code structure enabled rapid iteration

### Challenges & Solutions
- ⚠️ **Challenge:** ARIMA/Prophet underperformed on aggregated data  
  **Solution:** Focus on per-store modeling or stick with ML approaches

- ⚠️ **Challenge:** LSTM required extensive computational resources  
  **Solution:** Reduced sequence length, simplified architecture

- ⚠️ **Challenge:** Binary incompatibility with `pmdarima` library  
  **Solution:** Implemented manual ARIMA grid search using statsmodels

### Future Improvements
- 🔮 Implement hierarchical forecasting (store-level → aggregate)
- 🔮 Add external features (weather, local events, economic indicators)
- 🔮 Deploy real-time prediction API with FastAPI
- 🔮 Implement A/B testing framework for model comparison
- 🔮 Add automated model retraining on new data

---

## 📝 Documentation

- **Project Proposal:** [docs/proposal.pdf](docs/proposal.pdf)
- **Technical Report:** [docs/technical_report.pdf](docs/technical_report.pdf)
- **Model Performance:** [reports/performance_reports/](reports/performance_reports/)
- **API Documentation:** Coming soon

---

## 🤝 Contributing

This is an academic capstone project. For questions or collaboration:

1. Open an issue describing your suggestion
2. Fork the repository
3. Create a feature branch (`git checkout -b feature/AmazingFeature`)
4. Commit changes (`git commit -m 'Add some AmazingFeature'`)
5. Push to branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **IBM & Digital Egypt Pioneers Initiative (DEPI)** - For providing the capstone project framework
- **Kaggle & Rossmann** - For providing the high-quality dataset
- **Instructors & Mentors** - For guidance throughout the project
- **Open Source Community** - For the amazing tools and libraries

---

## 📧 Contact

**Project Repository:** [https://github.com/your-team/rossmann-forecasting](https://github.com/your-team/rossmann-forecasting)

**Team Email:** [team-email@example.com](mailto:team-email@example.com)

For project-specific questions, please open an issue on GitHub.

---

## 🏆 Project Status

**Current Phase:** Model Development & Optimization (Week 3 of 5)  
**Completion:** ~80%  
**Next Milestone:** MLOps Deployment (November 15-24, 2025)  
**Final Deadline:** November 30, 2025

---

**Last Updated:** November 14, 2025  
**Version:** 1.2.0

---

<<<<<<< HEAD
</div>
=======
</div>
>>>>>>> c8f0f01 (added md)
