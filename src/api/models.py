"""Pydantic models for API requests and responses - 22 FEATURES"""

from pydantic import BaseModel, Field
from typing import List, Dict

class PredictionInput(BaseModel):
    """Single prediction request schema - 22 features to match model"""
    
    # Basic temporal features
    DayOfWeek: int = Field(..., ge=1, le=7, description="Day of week (1-7)")
    Month: int = Field(..., ge=1, le=12, description="Month (1-12)")
    Quarter: int = Field(..., ge=1, le=4, description="Quarter (1-4)")
    IsWeekend: int = Field(..., ge=0, le=1, description="Is weekend (0/1)")
    
    # Store/Promo features
    Promo: float = Field(..., ge=0, le=1, description="Promotion active (0/1)")
    SchoolHoliday: int = Field(..., ge=0, le=1, description="School holiday (0/1)")
    
    # Lag features
    Sales_Lag_1: float = Field(..., gt=0, description="Sales lag 1 day")
    Sales_Lag_7: float = Field(..., gt=0, description="Sales lag 7 days")
    Sales_Lag_14: float = Field(..., gt=0, description="Sales lag 14 days")
    Sales_Lag_30: float = Field(..., gt=0, description="Sales lag 30 days")
    
    Customers_Lag_1: float = Field(..., gt=0, description="Customers lag 1 day")
    Customers_Lag_7: float = Field(..., gt=0, description="Customers lag 7 days")
    
    # Rolling features
    Sales_Rolling_Mean_7: float = Field(..., gt=0, description="7-day rolling mean sales")
    Sales_Rolling_Mean_14: float = Field(..., gt=0, description="14-day rolling mean sales")
    Sales_Rolling_Std_7: float = Field(..., ge=0, description="7-day rolling std sales")
    Sales_Rolling_Std_14: float = Field(..., ge=0, description="14-day rolling std sales")
    
    # Derived feature
    SalesPerCustomer: float = Field(..., gt=0, description="Sales per customer")
    
    # THE 5 MISSING FEATURES
    Store: int = Field(..., ge=1, description="Store ID")
    Open: int = Field(..., ge=0, le=1, description="Store is open (0/1)")
    StoreType: int = Field(..., ge=0, description="Store type (encoded)")
    Assortment: int = Field(..., ge=0, description="Assortment type (encoded)")
    CompetitionDistance: float = Field(..., ge=0, description="Distance to competitor in meters")

    class Config:
        schema_extra = {
            "example": {
                "DayOfWeek": 3,
                "Month": 11,
                "Quarter": 4,
                "IsWeekend": 0,
                "Promo": 1.0,
                "SchoolHoliday": 0,
                "Sales_Lag_1": 5000.0,
                "Sales_Lag_7": 4800.0,
                "Sales_Lag_14": 4700.0,
                "Sales_Lag_30": 4900.0,
                "Customers_Lag_1": 800.0,
                "Customers_Lag_7": 820.0,
                "Sales_Rolling_Mean_7": 4900.0,
                "Sales_Rolling_Mean_14": 4850.0,
                "Sales_Rolling_Std_7": 100.0,
                "Sales_Rolling_Std_14": 120.0,
                "SalesPerCustomer": 6.25,
                "Store": 1,
                "Open": 1,
                "StoreType": 0,
                "Assortment": 0,
                "CompetitionDistance": 1000.0
            }
        }

class PredictionOutput(BaseModel):
    """Prediction response schema"""
    prediction: float = Field(..., description="Predicted sales value")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence score")
    prediction_timestamp: str = Field(..., description="ISO format timestamp")
    model_version: str = Field(..., description="Model version used")

    class Config:
        schema_extra = {
            "example": {
                "prediction": 4875.25,
                "confidence": 0.95,
                "prediction_timestamp": "2025-11-20T10:30:00",
                "model_version": "1.0.0"
            }
        }

class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    model_loaded: bool
    api_version: str

class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    data: List[PredictionInput]

class ModelInfoResponse(BaseModel):
    """Model information response"""
    model_name: str
    version: str
    status: str
    performance_metrics: Dict[str, float]
    features_count: int
    last_updated: str