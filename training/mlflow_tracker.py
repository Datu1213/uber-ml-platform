"""
MLflow experiment tracking and model registry wrapper.

Provides simplified interface for:
- Experiment tracking (metrics, parameters, artifacts)
- Model registration and versioning
- Model stage transitions (staging -> production)
- Run management and comparison
"""

from typing import Dict, Any, Optional, List, Union
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import Run, Experiment
from mlflow.models import ModelSignature
import pandas as pd
from datetime import datetime

from common.config import settings
from common.logging import get_logger, audit_logger

logger = get_logger(__name__)


class MLflowTracker:
    """
    Wrapper for MLflow tracking and model registry operations.
    
    Simplifies common MLflow operations with error handling,
    logging, and audit trail generation.
    """
    
    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None
    ):
        """
        Initialize MLflow tracker.
        
        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Name of the experiment
        """
        self.tracking_uri = tracking_uri or settings.mlflow.tracking_uri
        mlflow.set_tracking_uri(self.tracking_uri)
        
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        
        if experiment_name:
            self.set_experiment(experiment_name)
        
        logger.info(f"MLflow tracker initialized: {self.tracking_uri}")
    
    def set_experiment(self, experiment_name: str):
        """
        Set or create experiment.
        
        Args:
            experiment_name: Name of the experiment
        """
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        logger.info(f"Experiment set to: {experiment_name}")
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name for the run
            tags: Dictionary of tags
            
        Returns:
            Run ID
        """
        run = mlflow.start_run(run_name=run_name, tags=tags)
        run_id = run.info.run_id
        
        logger.info(
            f"Started MLflow run: {run_id}",
            extra={
                "extra_fields": {
                    "run_id": run_id,
                    "run_name": run_name,
                    "experiment": self.experiment_name
                }
            }
        )
        
        # Log to audit trail
        audit_logger.log_model_training_start(
            model_name=run_name or "unknown",
            model_version=run_id,
            dataset_id="batch",
            user_id="system"
        )
        
        return run_id
    
    def end_run(self):
        """End the current MLflow run."""
        mlflow.end_run()
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters for the current run.
        
        Args:
            params: Dictionary of parameters
        """
        mlflow.log_params(params)
        logger.debug(f"Logged {len(params)} parameters")
    
    def log_param(self, key: str, value: Any):
        """Log a single parameter."""
        mlflow.log_param(key, value)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics for the current run.
        
        Args:
            metrics: Dictionary of metrics
            step: Training step/epoch number
        """
        mlflow.log_metrics(metrics, step=step)
        logger.debug(f"Logged {len(metrics)} metrics at step {step}")
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a single metric."""
        mlflow.log_metric(key, value, step=step)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log an artifact (file) for the current run.
        
        Args:
            local_path: Local file path
            artifact_path: Path within the artifact store
        """
        mlflow.log_artifact(local_path, artifact_path)
        logger.debug(f"Logged artifact: {local_path}")
    
    def log_model(
        self,
        model: Any,
        artifact_path: str,
        signature: Optional[ModelSignature] = None,
        registered_model_name: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Log a trained model.
        
        Args:
            model: The trained model object
            artifact_path: Path to store model within run
            signature: Model signature (inputs/outputs schema)
            registered_model_name: Name for model registry
            
        Returns:
            Model URI
        """
        # Infer model flavor and log accordingly
        if hasattr(model, 'predict'):
            # Scikit-learn compatible
            model_info = mlflow.sklearn.log_model(
                model,
                artifact_path,
                signature=signature,
                registered_model_name=registered_model_name,
                **kwargs
            )
        else:
            # Generic Python model
            model_info = mlflow.pyfunc.log_model(
                artifact_path,
                python_model=model,
                signature=signature,
                registered_model_name=registered_model_name,
                **kwargs
            )
        
        logger.info(
            f"Model logged: {artifact_path}",
            extra={
                "extra_fields": {
                    "artifact_path": artifact_path,
                    "registered_name": registered_model_name
                }
            }
        )
        
        return model_info.model_uri
    
    def register_model(
        self,
        model_uri: str,
        model_name: str,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Register a model to the model registry.
        
        Args:
            model_uri: URI of the model to register
            model_name: Name for the registered model
            tags: Optional tags
            
        Returns:
            Model version
        """
        result = mlflow.register_model(model_uri, model_name)
        version = result.version
        
        # Add tags if provided
        if tags:
            for key, value in tags.items():
                self.client.set_model_version_tag(model_name, version, key, value)
        
        logger.info(
            f"Model registered: {model_name} v{version}",
            extra={
                "extra_fields": {
                    "model_name": model_name,
                    "version": version
                }
            }
        )
        
        return version
    
    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str,
        archive_existing_versions: bool = False
    ):
        """
        Transition a model to a new stage.
        
        Args:
            model_name: Name of the registered model
            version: Model version
            stage: Target stage (Staging, Production, Archived)
            archive_existing_versions: Whether to archive existing versions
        """
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing_versions
        )
        
        logger.info(
            f"Model transitioned: {model_name} v{version} -> {stage}",
            extra={
                "extra_fields": {
                    "model_name": model_name,
                    "version": version,
                    "stage": stage
                }
            }
        )
        
        # Audit log for production deployments
        if stage == "Production":
            audit_logger.log_model_deployment(
                model_name=model_name,
                model_version=version,
                environment="production",
                deployed_by="system"
            )
    
    def get_best_run(
        self,
        experiment_name: str,
        metric_name: str,
        ascending: bool = False
    ) -> Optional[Run]:
        """
        Get the best run from an experiment based on a metric.
        
        Args:
            experiment_name: Name of the experiment
            metric_name: Metric to optimize
            ascending: Whether lower is better (True) or higher is better (False)
            
        Returns:
            Best run or None if no runs found
        """
        experiment = self.client.get_experiment_by_name(experiment_name)
        if not experiment:
            logger.warning(f"Experiment not found: {experiment_name}")
            return None
        
        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric_name} {'ASC' if ascending else 'DESC'}"],
            max_results=1
        )
        
        if runs:
            best_run = runs[0]
            logger.info(
                f"Best run: {best_run.info.run_id}",
                extra={
                    "extra_fields": {
                        "run_id": best_run.info.run_id,
                        f"best_{metric_name}": best_run.data.metrics.get(metric_name)
                    }
                }
            )
            return best_run
        
        return None
    
    def compare_runs(
        self,
        run_ids: List[str],
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare multiple runs by their metrics.
        
        Args:
            run_ids: List of run IDs to compare
            metrics: Specific metrics to compare (None = all)
            
        Returns:
            DataFrame with comparison data
        """
        comparison_data = []
        
        for run_id in run_ids:
            run = self.client.get_run(run_id)
            run_data = {
                "run_id": run_id,
                "run_name": run.data.tags.get("mlflow.runName", ""),
                "start_time": run.info.start_time,
            }
            
            # Add metrics
            for metric_key, metric_value in run.data.metrics.items():
                if metrics is None or metric_key in metrics:
                    run_data[metric_key] = metric_value
            
            comparison_data.append(run_data)
        
        df = pd.DataFrame(comparison_data)
        
        logger.info(f"Compared {len(run_ids)} runs")
        return df
    
    def load_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[str] = None
    ):
        """
        Load a model from the registry.
        
        Args:
            model_name: Name of the registered model
            version: Specific version (or None to use stage)
            stage: Stage to load from (Production, Staging)
            
        Returns:
            Loaded model
        """
        if version:
            model_uri = f"models:/{model_name}/{version}"
        elif stage:
            model_uri = f"models:/{model_name}/{stage}"
        else:
            raise ValueError("Must provide either version or stage")
        
        model = mlflow.pyfunc.load_model(model_uri)
        
        logger.info(
            f"Model loaded: {model_name}",
            extra={
                "extra_fields": {
                    "model_name": model_name,
                    "version": version,
                    "stage": stage
                }
            }
        )
        
        return model
    
    def delete_experiment(self, experiment_name: str):
        """Delete an experiment by name."""
        experiment = self.client.get_experiment_by_name(experiment_name)
        if experiment:
            self.client.delete_experiment(experiment.experiment_id)
            logger.info(f"Experiment deleted: {experiment_name}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if mlflow.active_run():
            self.end_run()


def get_mlflow_tracker(experiment_name: Optional[str] = None) -> MLflowTracker:
    """Get an MLflow tracker instance."""
    return MLflowTracker(experiment_name=experiment_name)


__all__ = [
    "MLflowTracker",
    "get_mlflow_tracker",
]
