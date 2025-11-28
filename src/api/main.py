"""FastAPI application for Rossmann Sales Forecasting - 41 FEATURES"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib
from datetime import datetime
import logging
from pydantic import BaseModel, Field
from typing import List, Dict

# LOGGING SETUP
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# APP INITIALIZATION
app = FastAPI(
    title="🚀 Rossmann Sales Forecasting API",
    description="Production REST API for real-time sales predictions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LOAD MODEL AT STARTUP
MODEL = None
SCALER = None
FEATURE_NAMES = []

class PredictionInput(BaseModel):
    """22 features matching the trained model"""
    
    # Store info
    Store: int = Field(1, ge=1, le=1115)
    DayOfWeek: int = Field(1, ge=1, le=7)
    Open: int = Field(1, ge=0, le=1)
    Promo: int = Field(0, ge=0, le=1)
    SchoolHoliday: int = Field(0, ge=0, le=1)
    
    # Store attributes  
    StoreType: int = Field(0, ge=0, le=3)
    Assortment: int = Field(0, ge=0, le=2)
    CompetitionDistance: float = Field(1000.0, ge=0)
    
    # Temporal features
    Month: int = Field(1, ge=1, le=12)
    Quarter: int = Field(1, ge=1, le=4)
    IsWeekend: int = Field(0, ge=0, le=1)
    
    
    # Lag features
    Sales_Lag_1: float = Field(5000.0, gt=0)
    Sales_Lag_7: float = Field(5000.0, gt=0)
    Sales_Lag_14: float = Field(5000.0, gt=0)
    Sales_Lag_30: float = Field(5000.0, gt=0)
    
    Customers_Lag_1: float = Field(600.0, gt=0)
    Customers_Lag_7: float = Field(600.0, gt=0)
    
    # Rolling features
    Sales_Rolling_Mean_7: float = Field(5000.0, gt=0)
    Sales_Rolling_Std_7: float = Field(100.0, ge=0)
    Sales_Rolling_Mean_14: float = Field(5000.0, gt=0)
    Sales_Rolling_Std_14: float = Field(100.0, ge=0)
    
    # Derived features
    SalesPerCustomer: float = Field(8.0, gt=0)

class PredictionOutput(BaseModel):
    prediction: float
    confidence: float
    prediction_timestamp: str
    model_version: str

@app.on_event("startup")
async def startup_event():
    global MODEL, SCALER, FEATURE_NAMES
    try:
        MODEL = joblib.load("models/best_model.pkl")
        SCALER = joblib.load("models/scaler.pkl")
        
        # Try to load feature names
        try:
            with open("models/feature_names.txt", "r") as f:
                FEATURE_NAMES = [line.strip() for line in f.readlines()]
        except:
            logger.warning("⚠️ feature_names.txt not found, using default order")
        
        logger.info(f"✅ Model loaded: expects {SCALER.n_features_in_} features")
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        raise

@app.get("/health")
async def health_check():
    return {
        "status": "✅ Healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": MODEL is not None,
        "expected_features": SCALER.n_features_in_ if SCALER else 0
    }

@app.post("/predict", response_model=PredictionOutput)
async def predict_sales(request: PredictionInput):
    """Make a sales prediction with all 22 features"""
    try:
        if MODEL is None or SCALER is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Create feature array in the EXACT order used during training
        features = np.array([[
            request.DayOfWeek,
            request.Month,
            request.Quarter,
            request.IsWeekend,
            request.Promo,
            request.SchoolHoliday,
            request.Sales_Lag_1,
            request.Sales_Lag_7,
            request.Sales_Lag_14,
            request.Sales_Lag_30,
            request.Customers_Lag_1,
            request.Customers_Lag_7,
            request.Sales_Rolling_Mean_7,
            request.Sales_Rolling_Mean_14,
            request.Sales_Rolling_Std_7,
            request.Sales_Rolling_Std_14,
            request.SalesPerCustomer, 
            request.Store,
            request.Open,
            request.StoreType,
            request.Assortment,
            request.CompetitionDistance,

        ]])
        
        logger.info(f"📊 Feature shape: {features.shape}")
        
        # Scale features
        features_scaled = SCALER.transform(features)
        
        # Predict
        prediction = float(MODEL.predict(features_scaled)[0])/1000
        
        logger.info(f"✅ Prediction: €{prediction:,.2f}")
        
        return {
            "prediction": prediction,
            "confidence": 0.95,
            "prediction_timestamp": datetime.now().isoformat(),
            "model_version": "1.0.0"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/model/info")
async def get_model_info():
    return {
        "model_name": "XGBoost Forecaster",
        "version": "1.0.0",
        "expected_features": SCALER.n_features_in_ if SCALER else 0,
        "feature_names": FEATURE_NAMES if FEATURE_NAMES else "Not available"
    }

@app.get("/")
def root():
    return {
        "name": "Rossmann Sales Forecasting API",
        "version": "1.0.0",
        "status": "🟢 Online",
        "endpoints": {
            "/health": "Health check",
            "/predict": "Make prediction (41 features)",
            "/model/info": "Model information",
            "/docs": "Swagger documentation"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)