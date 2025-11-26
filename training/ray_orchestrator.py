"""
Ray distributed training orchestrator.

Handles:
- Multi-node GPU training coordination
- Hyperparameter optimization with Ray Tune
- Resource allocation and scheduling
- Distributed data loading
"""

from typing import Dict, Any, Optional, Callable, List
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from ray.air import session, RunConfig, ScalingConfig
from ray.train import Trainer
from ray.train.torch import TorchTrainer
import numpy as np

from common.config import settings
from common.logging import get_logger
from training.mlflow_tracker import MLflowTracker

logger = get_logger(__name__)


class RayTrainingOrchestrator:
    """
    Orchestrates distributed training using Ray.
    
    Provides abstractions for:
    - Single-node multi-GPU training
    - Multi-node distributed training
    - Hyperparameter optimization
    - Resource-efficient scheduling
    """
    
    def __init__(
        self,
        num_workers: int = 1,
        use_gpu: bool = True,
        cpus_per_worker: int = 2,
        gpus_per_worker: float = 1.0
    ):
        """
        Initialize Ray training orchestrator.
        
        Args:
            num_workers: Number of parallel workers
            use_gpu: Whether to use GPU acceleration
            cpus_per_worker: CPUs allocated per worker
            gpus_per_worker: GPUs allocated per worker
        """
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(
                address=settings.ray.address,
                num_cpus=settings.ray.num_cpus,
                num_gpus=settings.ray.num_gpus,
                dashboard_port=settings.ray.dashboard_port
            )
        
        self.num_workers = num_workers
        self.use_gpu = use_gpu
        self.cpus_per_worker = cpus_per_worker
        self.gpus_per_worker = gpus_per_worker if use_gpu else 0
        
        logger.info(
            "Ray orchestrator initialized",
            extra={
                "extra_fields": {
                    "num_workers": num_workers,
                    "use_gpu": use_gpu,
                    "cpus_per_worker": cpus_per_worker,
                    "gpus_per_worker": gpus_per_worker
                }
            }
        )
    
    def train_distributed(
        self,
        train_func: Callable,
        config: Dict[str, Any],
        scaling_config: Optional[ScalingConfig] = None
    ) -> Dict[str, Any]:
        """
        Run distributed training across multiple workers.
        
        Args:
            train_func: Training function to distribute
            config: Training configuration
            scaling_config: Resource scaling configuration
            
        Returns:
            Training results
        """
        if scaling_config is None:
            scaling_config = ScalingConfig(
                num_workers=self.num_workers,
                use_gpu=self.use_gpu,
                resources_per_worker={
                    "CPU": self.cpus_per_worker,
                    "GPU": self.gpus_per_worker
                }
            )
        
        # Create trainer
        trainer = TorchTrainer(
            train_loop_per_worker=train_func,
            train_loop_config=config,
            scaling_config=scaling_config,
            run_config=RunConfig(
                name="uber_ml_training",
                local_dir="./ray_results",
                checkpoint_config={"num_to_keep": 3}
            )
        )
        
        logger.info("Starting distributed training")
        
        # Execute training
        result = trainer.fit()
        
        logger.info(
            "Distributed training completed",
            extra={
                "extra_fields": {
                    "metrics": result.metrics if result else None
                }
            }
        )
        
        return result
    
    def hyperparameter_search(
        self,
        train_func: Callable,
        param_space: Dict[str, Any],
        metric: str = "loss",
        mode: str = "min",
        num_samples: int = 10,
        max_concurrent_trials: int = 4
    ) -> tune.ResultGrid:
        """
        Perform hyperparameter optimization using Ray Tune.
        
        Args:
            train_func: Training function to optimize
            param_space: Hyperparameter search space
            metric: Metric to optimize
            mode: 'min' or 'max' for optimization
            num_samples: Number of trials to run
            max_concurrent_trials: Maximum concurrent trials
            
        Returns:
            Ray Tune results
        """
        # Configure search algorithm
        search_alg = HyperOptSearch(
            metric=metric,
            mode=mode
        )
        
        # Configure scheduler for early stopping
        scheduler = ASHAScheduler(
            metric=metric,
            mode=mode,
            max_t=100,  # Maximum training iterations
            grace_period=10,  # Minimum iterations before stopping
            reduction_factor=2
        )
        
        # Configure resources per trial
        resources_per_trial = {
            "cpu": self.cpus_per_worker,
            "gpu": self.gpus_per_worker
        }
        
        logger.info(
            f"Starting hyperparameter search with {num_samples} trials",
            extra={
                "extra_fields": {
                    "metric": metric,
                    "mode": mode,
                    "param_space": list(param_space.keys())
                }
            }
        )
        
        # Run tuning
        tuner = tune.Tuner(
            tune.with_resources(
                train_func,
                resources=resources_per_trial
            ),
            param_space=param_space,
            tune_config=tune.TuneConfig(
                search_alg=search_alg,
                scheduler=scheduler,
                num_samples=num_samples,
                max_concurrent_trials=max_concurrent_trials
            ),
            run_config=RunConfig(
                name="hpo_search",
                local_dir="./ray_results"
            )
        )
        
        results = tuner.fit()
        
        # Get best result
        best_result = results.get_best_result(metric=metric, mode=mode)
        
        logger.info(
            "Hyperparameter search completed",
            extra={
                "extra_fields": {
                    "best_config": best_result.config,
                    f"best_{metric}": best_result.metrics[metric]
                }
            }
        )
        
        return results
    
    def shutdown(self):
        """Shutdown Ray cluster."""
        if ray.is_initialized():
            ray.shutdown()
            logger.info("Ray cluster shutdown")


# Example training functions for different models
@ray.remote
def train_eta_model_worker(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker function for distributed ETA prediction model training.
    
    This would be called by Ray across multiple workers for distributed training.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    
    # Extract hyperparameters
    learning_rate = config.get("learning_rate", 0.001)
    batch_size = config.get("batch_size", 32)
    num_epochs = config.get("num_epochs", 10)
    hidden_size = config.get("hidden_size", 128)
    
    # Initialize MLflow tracking
    mlflow_tracker = MLflowTracker(experiment_name="eta-prediction")
    run_id = mlflow_tracker.start_run(run_name="distributed_training")
    
    # Log hyperparameters
    mlflow_tracker.log_params({
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "num_epochs": num_epochs
    })
    
    # Build model (simplified for example)
    class ETAModel(nn.Module):
        def __init__(self, input_size, hidden_size):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, 1)
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x
    
    model = ETAModel(input_size=20, hidden_size=hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Training loop (simplified)
    for epoch in range(num_epochs):
        # In production: iterate over actual DataLoader
        # For now, simulate training
        fake_loss = np.random.rand() * 10.0 / (epoch + 1)
        
        # Log metrics
        mlflow_tracker.log_metric("train_loss", fake_loss, step=epoch)
        mlflow_tracker.log_metric("epoch", epoch, step=epoch)
        
        # Report to Ray Tune for HPO
        session.report({"loss": fake_loss, "epoch": epoch})
    
    # Final results
    final_metrics = {
        "final_loss": fake_loss,
        "run_id": run_id
    }
    
    mlflow_tracker.end_run()
    
    return final_metrics


def train_surge_pricing_model(config: Dict[str, Any]) -> Dict[str, Any]:
    """Training function for surge pricing model."""
    # Similar structure to ETA model
    # Would include specific logic for surge pricing prediction
    pass


def train_fraud_detection_model(config: Dict[str, Any]) -> Dict[str, Any]:
    """Training function for fraud detection model."""
    # Similar structure with classification-specific logic
    pass


# Example usage function
def run_distributed_eta_training():
    """
    Example: Run distributed training for ETA prediction model.
    """
    orchestrator = RayTrainingOrchestrator(
        num_workers=4,
        use_gpu=True,
        gpus_per_worker=1.0
    )
    
    # Define hyperparameter search space
    param_space = {
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([16, 32, 64, 128]),
        "hidden_size": tune.choice([64, 128, 256, 512]),
        "num_epochs": tune.choice([10, 20, 30])
    }
    
    # Run hyperparameter optimization
    results = orchestrator.hyperparameter_search(
        train_func=train_eta_model_worker,
        param_space=param_space,
        metric="loss",
        mode="min",
        num_samples=20,
        max_concurrent_trials=4
    )
    
    # Get best configuration
    best_result = results.get_best_result()
    print(f"Best config: {best_result.config}")
    print(f"Best loss: {best_result.metrics['loss']}")
    
    orchestrator.shutdown()


__all__ = [
    "RayTrainingOrchestrator",
    "train_eta_model_worker",
    "run_distributed_eta_training",
]
