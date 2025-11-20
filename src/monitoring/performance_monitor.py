"""Model and system performance monitoring"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor model performance and system health"""
    
    def __init__(self, baseline_rmse=147015, baseline_mape=1.65):
        self.baseline_rmse = baseline_rmse
        self.baseline_mape = baseline_mape
        self.threshold_degradation = 0.15  # 15% threshold
        self.predictions_log = []
        logger.info("✅ PerformanceMonitor initialized")
    
    def log_prediction(self, actual, predicted, store_id, timestamp=None):
        """Log a prediction for monitoring"""
        if timestamp is None:
            timestamp = datetime.now()
        
        error = abs(actual - predicted)
        error_pct = (error / actual) * 100 if actual > 0 else 0
        
        record = {
            'timestamp': timestamp,
            'store_id': store_id,
            'actual': actual,
            'predicted': predicted,
            'error': error,
            'error_pct': error_pct
        }
        
        self.predictions_log.append(record)
        logger.info(f"📝 Prediction logged: Store {store_id}, Error: {error_pct:.2f}%")
        
        return record
    
    def check_model_performance(self):
        """Check if model performance is acceptable"""
        if len(self.predictions_log) < 10:
            return {"status": "⏳ Insufficient data"}
        
        recent = pd.DataFrame(self.predictions_log[-100:])
        
        rmse = np.sqrt(np.mean(recent['error']**2))
        mape = recent['error_pct'].mean()
        
        rmse_degradation = (rmse - self.baseline_rmse) / self.baseline_rmse
        mape_degradation = (mape - self.baseline_mape) / self.baseline_mape
        
        status = {
            'current_rmse': rmse,
            'baseline_rmse': self.baseline_rmse,
            'rmse_degradation_pct': rmse_degradation * 100,
            'current_mape': mape,
            'baseline_mape': self.baseline_mape,
            'mape_degradation_pct': mape_degradation * 100,
            'alert': False
        }
        
        if abs(rmse_degradation) > self.threshold_degradation:
            status['alert'] = True
            logger.warning(f"🚨 ALERT: RMSE degradation {rmse_degradation*100:.2f}%")
        
        return status
    
    def generate_report(self):
        """Generate monitoring report"""
        if not self.predictions_log:
            return {"message": "No predictions logged yet"}
        
        df = pd.DataFrame(self.predictions_log)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_predictions': len(df),
            'performance': {
                'rmse': float(np.sqrt(np.mean(df['error']**2))),
                'mae': float(np.mean(df['error'])),
                'mape': float(df['error_pct'].mean())
            }
        }
        
        logger.info(f"📊 Report generated: {len(df)} predictions")
        return report
    
    def save_logs(self, filepath='logs/predictions.json'):
        """Save prediction logs to file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.predictions_log, f, default=str, indent=2)
            logger.info(f"✅ Logs saved to {filepath}")
        except Exception as e:
            logger.error(f"❌ Error saving logs: {e}")