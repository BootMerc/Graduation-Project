"""
Complete training pipeline with MLflow tracking + DVC versioning
"""

import os
import logging
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

from src.mlops.mlflow_tracker import MLFlowTracker
from src.mlops.dvc_handler import DVCHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingPipeline:
    """End-to-end training pipeline with MLOps"""
    
    def __init__(self, data_path: str = "Data/cleaned_sales_features.csv"):
        self.data_path = data_path
        self.tracker = MLFlowTracker()
        self.dvc = DVCHandler()
        
        # Model and scaler will be stored here
        self.model = None
        self.scaler = None
    
    def load_data(self):
        """Load training data"""
        logger.info(f"📂 Loading data from {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        
        logger.info(f"✅ Data loaded: {len(df)} samples, {len(df.columns)} features")
        
        # Drop non-numeric columns first
        if 'Date' in df.columns:
            logger.info("⚠️ Dropping 'Date' column")
            df = df.drop(columns=['Date'])
        
        # Drop any remaining object columns
        object_cols = df.select_dtypes(include=['object']).columns.tolist()
        if object_cols:
            logger.info(f"⚠️ Dropping non-numeric columns: {object_cols}")
            df = df.drop(columns=object_cols)
        
        # Separate features and target
        if 'Sales' not in df.columns:
            raise ValueError("❌ 'Sales' column not found in dataset")
        
        # Drop Store if it exists (it's an ID, not a feature)
        cols_to_drop = ['Sales']
        if 'Store' in df.columns:
            cols_to_drop.append('Store')
        
        X = df.drop(columns=cols_to_drop)
        y = df['Sales']
        
        logger.info(f"✅ Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train/test"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"✅ Split: {len(X_train)} train, {len(X_test)} test")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test):
        """Scale features using StandardScaler"""
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info("✅ Features scaled")
        
        return X_train_scaled, X_test_scaled
    
    def train_model(self, X_train, y_train, X_test, y_test, params: dict = None):
        """Train XGBoost model"""
        
        if params is None:
            params = {
                "n_estimators": 1000,
                "max_depth": 7,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "objective": "reg:squarederror",
                "random_state": 42
            }
        
        logger.info("🚀 Training XGBoost model...")
        
        self.model = xgb.XGBRegressor(**params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        logger.info("✅ Model trained")
        
        return self.model
    
    def evaluate_model(self, X_test, y_test, y_pred):
        """
        Calculate evaluation metrics
        
        Args:
            X_test: Test features (not used, just for API consistency)
            y_test: True target values
            y_pred: Predicted target values
        
        Returns:
            dict: Dictionary of metrics
        """
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1))) * 100
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            "rmse": float(rmse),
            "mae": float(mae),
            "mape": float(mape),
            "r2": float(r2)
        }
        
        logger.info(f"📊 Metrics: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%, R²={r2:.4f}")
        
        return metrics
    
    def save_artifacts(self, output_dir: str = "models"):
        """Save model and scaler"""
        Path(output_dir).mkdir(exist_ok=True)
        
        model_path = f"{output_dir}/best_model.pkl"
        scaler_path = f"{output_dir}/scaler.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        logger.info(f"✅ Saved model: {model_path}")
        logger.info(f"✅ Saved scaler: {scaler_path}")
        
        return model_path, scaler_path
    
    def run(self, experiment_name: str = "XGBoost-Baseline", params: dict = None):
        """Run complete training pipeline"""
        
        logger.info("="*80)
        logger.info("🚀 STARTING TRAINING PIPELINE")
        logger.info("="*80)
        
        # Start MLflow run
        self.tracker.start_run(
            run_name=experiment_name,
            tags={
                "experiment": experiment_name,
                "timestamp": datetime.now().isoformat(),
                "version": "1.0"
            }
        )
        
        try:
            # Load data
            X, y = self.load_data()
            
            # Split
            X_train, X_test, y_train, y_test = self.split_data(X, y)
            
            # Scale
            X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
            
            # Log parameters
            if params is None:
                params = {
                    "n_estimators": 1000,
                    "max_depth": 7,
                    "learning_rate": 0.1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8
                }
            
            self.tracker.log_parameters(params)
            
            # Train
            self.train_model(X_train_scaled, y_train, X_test_scaled, y_test, params)
            
            # Predict
            y_pred = self.model.predict(X_test_scaled)
            
            # Evaluate - FIXED: Pass X_test_scaled, y_test, y_pred
            metrics = self.evaluate_model(X_test_scaled, y_test, y_pred)
            
            # Log metrics
            self.tracker.log_metrics(metrics)
            
            # Save artifacts
            model_path, scaler_path = self.save_artifacts()
            
            # Log to MLflow
            self.tracker.log_model(self.model)
            self.tracker.log_artifact(model_path)
            self.tracker.log_artifact(scaler_path)
            
            # Track with DVC
            self.dvc.add_artifact(model_path)
            self.dvc.add_artifact(scaler_path)
            
            # End run
            self.tracker.end_run("FINISHED")
            
            logger.info("="*80)
            logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("="*80)
            logger.info(f"\n📊 Final Metrics:")
            for metric, value in metrics.items():
                logger.info(f"  {metric.upper()}: {value:.4f}")
            
            return self.model, self.scaler, metrics
        
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            self.tracker.end_run("FAILED")
            raise


# ============================================================================
# RUN FROM COMMAND LINE
# ============================================================================

if __name__ == "__main__":
    pipeline = TrainingPipeline(data_path="Data/cleaned_sales_features.csv")
    
    model, scaler, metrics = pipeline.run(
        experiment_name="Rossmann-XGBoost-v1-MLOps"
    )
    
    logger.info("\n✅ Training completed with MLOps tracking!")