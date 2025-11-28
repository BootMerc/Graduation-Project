import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json

# ============================================================================
# LOGGER SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/monitoring.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ============================================================================
# MODEL MONITORING CLASS
# ============================================================================

class ModelMonitor:
    """Real-time model performance monitoring"""
    
    def __init__(self, baseline_rmse=147015, baseline_mape=1.65):
        self.baseline_rmse = baseline_rmse
        self.baseline_mape = baseline_mape
        self.threshold_degradation = 0.15  # 15% degradation triggers alert
        self.predictions_log = []
        
        logger.info("✅ ModelMonitor initialized")
    
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
        
        return record
    
    def check_data_quality(self, data):
        """Check for data quality issues"""
        issues = []
        
        # Check for nulls
        null_pct = (data.isnull().sum() / len(data)) * 100
        if null_pct.max() > 5:
            issues.append(f"⚠️ High null percentage: {null_pct.max():.2f}%")
        
        # Check for outliers (3-sigma rule)
        for col in data.select_dtypes(include=[np.number]).columns:
            mean = data[col].mean()
            std = data[col].std()
            outliers = ((data[col] < mean - 3*std) | (data[col] > mean + 3*std)).sum()
            if outliers > 0:
                issues.append(f"⚠️ {outliers} outliers detected in {col}")
        
        return issues
    
    def check_model_performance(self):
        """Check if model performance is within acceptable range"""
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
        
        # Trigger alert if degradation > threshold
        if abs(rmse_degradation) > self.threshold_degradation:
            status['alert'] = True
            logger.warning(f"🚨 ALERT: RMSE degradation detected! {rmse_degradation*100:.2f}%")
        
        if abs(mape_degradation) > self.threshold_degradation:
            status['alert'] = True
            logger.warning(f"🚨 ALERT: MAPE degradation detected! {mape_degradation*100:.2f}%")
        
        return status
    
    def detect_data_drift(self, current_data, baseline_data):
        """Detect if input data distribution has shifted"""
        drift_detected = False
        
        for col in baseline_data.select_dtypes(include=[np.number]).columns:
            baseline_mean = baseline_data[col].mean()
            current_mean = current_data[col].mean()
            
            # Use Welch's t-test
            from scipy import stats
            stat, pvalue = stats.ttest_ind(
                baseline_data[col].dropna(),
                current_data[col].dropna()
            )
            
            if pvalue < 0.05:  # Significant shift detected
                logger.warning(f"⚠️ Data drift detected in {col}: p-value={pvalue:.4f}")
                drift_detected = True
        
        return drift_detected
    
    def generate_report(self):
        """Generate monitoring report"""
        if not self.predictions_log:
            return {"message": "No predictions logged yet"}
        
        df = pd.DataFrame(self.predictions_log)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_predictions': len(df),
            'period': f"{df['timestamp'].min()} to {df['timestamp'].max()}",
            'performance': {
                'rmse': float(np.sqrt(np.mean(df['error']**2))),
                'mae': float(np.mean(df['error'])),
                'mape': float(df['error_pct'].mean()),
                'median_error': float(df['error'].median())
            },
            'by_store': df.groupby('store_id').agg({
                'error': ['mean', 'min', 'max'],
                'error_pct': 'mean'
            }).to_dict()
        }
        
        return report

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    monitor = ModelMonitor()
    
    # Simulate predictions
    for i in range(50):
        actual = np.random.normal(50000, 5000)
        predicted = actual + np.random.normal(0, 1000)
        store_id = np.random.randint(1, 100)
        
        monitor.log_prediction(actual, predicted, store_id)
    
    # Check performance
    perf = monitor.check_model_performance()
    print(json.dumps(perf, indent=2))
    
    # Generate report
    report = monitor.generate_report()
    print(json.dumps(report, indent=2))