"""
MLflow experiment tracking for Rossmann Sales Forecasting
Logs all model training experiments, metrics, and parameters
"""

import mlflow
import mlflow.xgboost
import mlflow.sklearn
import numpy as np
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class MLFlowTracker:
    """Centralized MLflow tracking for experiments"""
    
    def __init__(self, tracking_uri: str = "mlruns", experiment_name: str = "Rossmann-Sales"):
        """
        Initialize MLflow tracker
        
        Args:
            tracking_uri: Path to store MLflow runs (default: ./mlruns)
            experiment_name: Name of the experiment
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        
        # Set tracking URI (local file system by default)
        mlflow.set_tracking_uri(tracking_uri)
        
        # Set or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(f"✅ Created experiment: {experiment_name} (ID: {experiment_id})")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"✅ Using existing experiment: {experiment_name}")
        except Exception as e:
            logger.error(f"❌ Error setting up experiment: {e}")
            raise
        
        self.experiment_id = experiment_id
        mlflow.set_experiment(experiment_name)
    
    def start_run(self, run_name: str, tags: dict = None):
        """
        Start a new MLflow run
        
        Args:
            run_name: Human-readable name for this run
            tags: Dictionary of tags (e.g., {"model": "XGBoost", "version": "1.0"})
        """
        mlflow.start_run(run_name=run_name)
        
        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, value)
        
        logger.info(f"🚀 Started run: {run_name}")
    
    def log_parameters(self, params: dict):
        """
        Log model hyperparameters
        
        Args:
            params: Dictionary of parameters
        """
        mlflow.log_params(params)
        logger.info(f"📝 Logged {len(params)} parameters")
    
    def log_metrics(self, metrics: dict, step: int = None):
        """
        Log model performance metrics
        
        Args:
            metrics: Dictionary of metrics (RMSE, MAE, MAPE, R²)
            step: Training step/epoch
        """
        mlflow.log_metrics(metrics, step=step)
        
        # Pretty print metrics
        metric_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        logger.info(f"📊 Logged metrics: {metric_str}")
    
    def log_model(self, model, model_name: str = "xgboost_model"):
        """
        Log trained model to MLflow
        
        Args:
            model: Trained XGBoost model
            model_name: Name for the model
        """
        mlflow.xgboost.log_model(model, model_name)
        logger.info(f"✅ Logged model: {model_name}")
    
    def log_artifact(self, file_path: str, artifact_path: str = None):
        """
        Log artifacts (plots, data, configs)
        
        Args:
            file_path: Path to file to log
            artifact_path: Destination path in MLflow
        """
        mlflow.log_artifact(file_path, artifact_path)
        logger.info(f"📎 Logged artifact: {file_path}")
    
    def log_dataset_info(self, dataset_info: dict):
        """
        Log dataset information
        
        Args:
            dataset_info: Dictionary with dataset stats
        """
        mlflow.log_dict(dataset_info, "dataset_info.json")
        logger.info(f"📂 Logged dataset info")
    
    def end_run(self, status: str = "FINISHED"):
        """
        End current MLflow run
        
        Args:
            status: Run status (FINISHED, FAILED, KILLED)
        """
        mlflow.end_run(status)
        logger.info(f"✅ Ended run with status: {status}")
    
    def get_best_run(self, metric: str = "rmse"):
        """
        Get best run based on metric
        
        Args:
            metric: Metric to optimize (lower is better)
        
        Returns:
            Best run info
        """
        experiment = mlflow.get_experiment(self.experiment_id)
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        
        if runs.empty:
            logger.warning("⚠️ No runs found")
            return None
        
        best_run = runs.loc[runs[f"metrics.{metric}"].idxmin()]
        logger.info(f"🏆 Best run: {best_run['run_id']} with {metric}={best_run[f'metrics.{metric}']:.4f}")
        
        return best_run
    
    @staticmethod
    def compare_runs(experiment_id: str):
        """
        Compare all runs in experiment
        
        Args:
            experiment_id: Experiment ID to compare
        """
        runs = mlflow.search_runs(experiment_ids=[experiment_id])
        
        if runs.empty:
            logger.warning("⚠️ No runs to compare")
            return None
        
        logger.info("\n📊 EXPERIMENT COMPARISON:")
        logger.info("="*80)
        
        # Display key metrics
        comparison_cols = [
            'run_id', 'start_time', 'status',
            'metrics.rmse', 'metrics.mae', 'metrics.mape', 'metrics.r2'
        ]
        
        existing_cols = [col for col in comparison_cols if col in runs.columns]
        print(runs[existing_cols].to_string())
        
        return runs


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_training_with_mlflow():
    """
    Example: How to use MLflow in your training pipeline
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb
    import pandas as pd
    
    # Initialize tracker
    tracker = MLFlowTracker(
        tracking_uri="./mlruns",
        experiment_name="Rossmann-Sales-Forecasting"
    )
    
    # Load your data
    df = pd.read_csv("Data/cleaned_sales_features.csv")
    X = df.drop(['Sales', 'Store'], axis=1)
    y = df['Sales']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Start MLflow run
    tracker.start_run(
        run_name="XGBoost-v1-baseline",
        tags={
            "model_type": "XGBoost",
            "version": "1.0",
            "author": "Zyad",
            "date": datetime.now().isoformat()
        }
    )
    
    # Log parameters
    params = {
        "n_estimators": 1000,
        "max_depth": 7,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror"
    }
    tracker.log_parameters(params)
    
    # Train model
    model = xgb.XGBRegressor(**params, random_state=42)
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False
    )
    
    # Calculate metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    y_pred = model.predict(X_test_scaled)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    r2 = model.score(X_test_scaled, y_test)
    
    metrics = {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2
    }
    
    # Log metrics
    tracker.log_metrics(metrics)
    
    # Log model
    tracker.log_model(model, "xgboost_regressor")
    
    # Log dataset info
    dataset_info = {
        "n_samples": len(X_train),
        "n_features": X_train.shape[1],
        "train_size": len(X_train),
        "test_size": len(X_test),
        "target": "Sales"
    }
    tracker.log_dataset_info(dataset_info)
    
    # End run
    tracker.end_run("FINISHED")
    
    logger.info(f"✅ Training complete! Run tracked in MLflow")
    
    return model, scaler


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model, scaler = example_training_with_mlflow()