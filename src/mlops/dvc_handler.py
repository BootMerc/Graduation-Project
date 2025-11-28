"""
DVC (Data Version Control) integration for Rossmann project
Manages model and dataset versioning with reproducible pipelines
"""

import os
import yaml
import logging
from pathlib import Path
import subprocess
from datetime import datetime

logger = logging.getLogger(__name__)

class DVCHandler:
    """Manages DVC pipeline and artifact versioning"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.dvc_dir = self.project_root / ".dvc"
        
        # Check if DVC is initialized
        if not self.dvc_dir.exists():
            logger.warning("⚠️ DVC not initialized. Run: dvc init")
        else:
            logger.info("✅ DVC initialized")
    
    def add_artifact(self, file_path: str, artifact_name: str = None):
        """
        Add file/model to DVC tracking
        
        Args:
            file_path: Path to file to track
            artifact_name: Optional name for artifact
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"❌ File not found: {file_path}")
            return False
        
        try:
            # Run: dvc add <file>
            subprocess.run(["dvc", "add", str(file_path)], check=True, cwd=str(self.project_root))
            logger.info(f"✅ Added to DVC: {file_path}")
            
            # This creates a .dvc file
            dvc_file = f"{file_path}.dvc"
            logger.info(f"📄 Created tracking file: {dvc_file}")
            
            return True
        
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Error adding to DVC: {e}")
            return False
    
    def create_pipeline(self, pipeline_config: dict, output_file: str = "dvc.yaml"):
        """
        Create DVC pipeline (reproducible stages)
        
        Args:
            pipeline_config: Pipeline configuration dictionary
            output_file: Output pipeline file
        """
        output_path = self.project_root / output_file
        
        try:
            with open(output_path, 'w') as f:
                yaml.dump(pipeline_config, f, default_flow_style=False)
            
            logger.info(f"✅ Created pipeline: {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"❌ Error creating pipeline: {e}")
            return False
    
    def run_pipeline(self):
        """Run entire DVC pipeline"""
        try:
            subprocess.run(["dvc", "repro"], check=True, cwd=str(self.project_root))
            logger.info("✅ Pipeline executed successfully")
            return True
        
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Pipeline execution failed: {e}")
            return False
    
    def push_artifacts(self):
        """Push tracked artifacts to remote storage"""
        try:
            subprocess.run(["dvc", "push"], check=True, cwd=str(self.project_root))
            logger.info("✅ Artifacts pushed to remote storage")
            return True
        
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Error pushing artifacts: {e}")
            return False
    
    def pull_artifacts(self):
        """Pull tracked artifacts from remote storage"""
        try:
            subprocess.run(["dvc", "pull"], check=True, cwd=str(self.project_root))
            logger.info("✅ Artifacts pulled from remote storage")
            return True
        
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Error pulling artifacts: {e}")
            return False
    
    @staticmethod
    def create_dvc_yaml_template():
        """
        Generate template dvc.yaml for your project
        """
        pipeline = {
            "stages": {
                "prepare": {
                    "cmd": "python src/data/prepare.py",
                    "deps": ["Data/raw.csv"],
                    "outs": ["Data/prepared.csv"]
                },
                "train": {
                    "cmd": "python src/train.py",
                    "deps": ["Data/prepared.csv", "src/train.py"],
                    "params": ["params.yaml:train"],
                    "outs": ["models/model.pkl"],
                    "metrics": [{"metrics.json": {"cache": False}}]
                },
                "evaluate": {
                    "cmd": "python src/evaluate.py",
                    "deps": ["models/model.pkl"],
                    "metrics": [{"evaluation.json": {"cache": False}}]
                }
            }
        }
        
        return pipeline


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_dvc_setup():
    """Example: Set up DVC versioning"""
    
    dvc = DVCHandler(project_root=".")
    
    # Add models to DVC
    dvc.add_artifact("models/best_model.pkl", "xgboost_model")
    dvc.add_artifact("models/scaler.pkl", "standard_scaler")
    dvc.add_artifact("Data/cleaned_sales_features.csv", "training_data")
    
    logger.info("\n✅ All artifacts tracked with DVC")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    example_dvc_setup()