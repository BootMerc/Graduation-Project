"""FastAPI application for Rossmann Sales Forecasting - 22 FEATURES"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib
from datetime import datetime
import logging
from .models import (
    PredictionInput, PredictionOutput, HealthCheckResponse,
    BatchPredictionRequest, ModelInfoResponse
)

# LOGGING SETUP
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api.log'),
        logging.StreamHandler()
    ]
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

# Add CORS middleware
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

# THE CORRECT 22 FEATURES (17 from scaler + 5 missing)
FEATURE_NAMES = [
    'DayOfWeek', 'Month', 'Quarter', 'IsWeekend', 'Promo', 'SchoolHoliday',
    'Sales_Lag_1', 'Sales_Lag_7', 'Sales_Lag_14', 'Sales_Lag_30',
    'Customers_Lag_1', 'Customers_Lag_7', 'Sales_Rolling_Mean_7',
    'Sales_Rolling_Mean_14', 'Sales_Rolling_Std_7', 'Sales_Rolling_Std_14',
    'SalesPerCustomer', 'Store', 'Open', 'StoreType', 'Assortment',
    'CompetitionDistance'
]

@app.on_event("startup")
async def startup_event():
    """Load model on application startup"""
    global MODEL, SCALER
    try:
        MODEL = joblib.load("models/best_model.pkl")
        SCALER = joblib.load("models/scaler.pkl")
        
        logger.info("✅ Model and scaler loaded successfully")
        logger.info(f"📊 Expected features: {len(FEATURE_NAMES)}")
        logger.info(f"⚠️  NOTE: Using 17 features from scaler + 5 unscaled features")
        
    except FileNotFoundError as e:
        logger.error(f"❌ Model file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 API shutting down...")

# HEALTH CHECK ENDPOINT
@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Check API health and model status"""
    logger.info("📡 Health check requested")
    return {
        "status": "✅ Healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": MODEL is not None,
        "api_version": "1.0.0"
    }

# SINGLE PREDICTION ENDPOINT
@app.post("/predict", response_model=PredictionOutput)
async def predict_sales(request: PredictionInput):
    """
    Make a single sales prediction
    
    Requires 22 features (17 scaled + 5 unscaled store features)
    """
    try:
        if MODEL is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # First 17 features (to be scaled)
        features_to_scale = np.array([[
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
            request.SalesPerCustomer
        ]])
        
        # Last 5 features (store-related, NOT scaled)
        features_not_scaled = np.array([[
            request.Store,
            request.Open,
            request.StoreType,
            request.Assortment,
            request.CompetitionDistance
        ]])
        
        logger.info(f"📊 Features to scale: {features_to_scale.shape}")
        logger.info(f"📊 Features NOT scaled: {features_not_scaled.shape}")
        
        # Scale the first 17 features
        features_scaled = SCALER.transform(features_to_scale)
        
        # Combine scaled + unscaled features
        features_final = np.concatenate([features_scaled, features_not_scaled], axis=1)
        
        logger.info(f"📊 Final feature shape: {features_final.shape}")
        logger.info(f"📊 Final features (first 5): {features_final[0][:5]}")
        
        # Make prediction
        prediction_array = MODEL.predict(features_final)
        prediction_value = float(prediction_array[0])/1000
        confidence_value = 0.95
        
        logger.info(f"✅ Raw prediction: {prediction_value}")
        logger.info(f"✅ Formatted prediction: €{prediction_value:,.2f}")
        
        # Create response
        response = {
            "prediction": prediction_value,
            "confidence": confidence_value,
            "prediction_timestamp": datetime.now().isoformat(),
            "model_version": "1.0.0"
        }
        
        logger.info(f"📤 Final response: {response}")
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# BATCH PREDICTION ENDPOINT
@app.post("/predict_batch")
async def predict_batch(request: BatchPredictionRequest):
    """Make batch predictions for multiple records"""
    try:
        if MODEL is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        predictions = []
        errors = []
        
        for idx, item in enumerate(request.data):
            try:
                # First 17 features (scaled)
                features_to_scale = np.array([[
                    item.DayOfWeek, item.Month, item.Quarter, item.IsWeekend,
                    item.Promo, item.SchoolHoliday, item.Sales_Lag_1,
                    item.Sales_Lag_7, item.Sales_Lag_14, item.Sales_Lag_30,
                    item.Customers_Lag_1, item.Customers_Lag_7,
                    item.Sales_Rolling_Mean_7, item.Sales_Rolling_Mean_14,
                    item.Sales_Rolling_Std_7, item.Sales_Rolling_Std_14,
                    item.SalesPerCustomer
                ]])
                
                # Last 5 features (not scaled)
                features_not_scaled = np.array([[
                    item.Store, item.Open, item.StoreType,
                    item.Assortment, item.CompetitionDistance
                ]])
                
                features_scaled = SCALER.transform(features_to_scale)
                features_final = np.concatenate([features_scaled, features_not_scaled], axis=1)
                
                pred = float(MODEL.predict(features_final)[0])
                predictions.append(pred)
            
            except Exception as e:
                logger.warning(f"⚠️ Error predicting item {idx}: {str(e)}")
                errors.append({"index": idx, "error": str(e)})
        
        logger.info(f"✅ Batch predictions completed: {len(predictions)} successful, {len(errors)} errors")
        
        return {
            "batch_size": len(request.data),
            "successful": len(predictions),
            "failed": len(errors),
            "predictions": predictions,
            "errors": errors,
            "timestamp": datetime.now().isoformat(),
            "model_version": "1.0.0"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# MODEL METADATA ENDPOINT
@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get model metadata and performance statistics"""
    return {
        "model_name": "XGBoost Forecaster",
        "version": "1.0.0",
        "status": "Production",
        "performance_metrics": {
            "rmse": 147015.0,
            "mae": 72390.0,
            "mape": 1.65,
            "r2": 0.9979
        },
        "features_count": 22,
        "last_updated": "2025-11-16T22:00:00"
    }

# FEATURES ENDPOINT
@app.get("/model/features")
async def get_model_features():
    """Get list of features expected by the model"""
    return {
        "features": FEATURE_NAMES,
        "count": len(FEATURE_NAMES),
        "scaled_features": FEATURE_NAMES[:17],
        "unscaled_features": FEATURE_NAMES[17:],
        "description": "First 17 features are scaled, last 5 are not"
    }

# ROOT ENDPOINT
@app.get("/")
def root():
    """API documentation and endpoints overview"""
    return {
        "name": "Rossmann Sales Forecasting API",
        "version": "1.0.0",
        "status": "🟢 Online",
        "endpoints": {
            "/health": "GET - Health check",
            "/predict": "POST - Single prediction",
            "/predict_batch": "POST - Batch predictions",
            "/model/info": "GET - Model information",
            "/model/features": "GET - Feature list",
            "/docs": "GET - Swagger UI documentation",
            "/redoc": "GET - ReDoc documentation"
        },
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc"
        }
    }

# ERROR HANDLERS
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors"""
    logger.error(f"Validation error: {str(exc)}")
    return {
        "error": "Validation Error",
        "detail": str(exc)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)