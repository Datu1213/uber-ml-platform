"""
ETA (Estimated Time of Arrival) prediction model training.

Trains a model to predict trip duration based on:
- Geographic features (pickup/dropoff locations, distance)
- Temporal features (time of day, day of week)
- Traffic and weather conditions
- Historical driver/rider patterns
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from typing import Dict, Any, Tuple

from common.config import settings
from common.logging import get_logger
from training.mlflow_tracker import MLflowTracker
from feature_store.feature_client import get_feature_store

logger = get_logger(__name__)


class ETAPredictor:
    """
    ETA prediction model for Uber trips.
    
    Predicts trip duration in minutes based on trip, rider, driver,
    and environmental features.
    """
    
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        """
        Initialize ETA predictor.
        
        Args:
            model_params: XGBoost model parameters
        """
        self.model_params = model_params or {
            'objective': 'reg:squarederror',
            'max_depth': 8,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        self.model = None
        self.feature_names = None
    
    def prepare_features(
        self,
        entity_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features from feature store.
        
        Args:
            entity_df: DataFrame with entity IDs and timestamps
            
        Returns:
            Tuple of (features_df, target_series)
        """
        logger.info("Preparing features from feature store")
        
        # Get feature store client
        fs = get_feature_store()
        
        # Retrieve historical features with point-in-time correctness
        features_df = fs.get_historical_features(
            entity_df=entity_df,
            feature_service_name="eta_prediction"
        )
        
        # Define feature columns
        feature_cols = [
            # Trip features
            'trip_distance_km',
            'traffic_level',
            'weather_condition',
            
            # Driver features
            'driver_avg_trip_duration_7d',
            'vehicle_type',
            
            # Temporal features
            'hour_of_day',
            'day_of_week',
            'is_peak_hour',
        ]
        
        # Handle categorical variables
        features_df['traffic_level'] = features_df['traffic_level'].map({
            'light': 0, 'moderate': 1, 'heavy': 2
        }).fillna(1)
        
        features_df['weather_condition'] = features_df['weather_condition'].map({
            'clear': 0, 'rain': 1, 'snow': 2, 'fog': 3
        }).fillna(0)
        
        features_df['vehicle_type'] = features_df['vehicle_type'].map({
            'sedan': 0, 'suv': 1, 'luxury': 2
        }).fillna(0)
        
        # Extract features and target
        X = features_df[feature_cols].fillna(0)
        y = features_df['actual_duration_minutes']  # Assuming this exists in training data
        
        self.feature_names = feature_cols
        
        logger.info(f"Prepared {len(X)} samples with {len(feature_cols)} features")
        
        return X, y
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict[str, float]:
        """
        Train the ETA prediction model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Training ETA prediction model")
        
        # Initialize model
        self.model = xgb.XGBRegressor(**self.model_params)
        
        # Train with early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        # Evaluate on validation set
        metrics = self.evaluate(X_val, y_val)
        
        logger.info(
            "Training completed",
            extra={"extra_fields": metrics}
        )
        
        return metrics
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Features
            y: True targets
            
        Returns:
            Dictionary of metrics
        """
        predictions = self.model.predict(X)
        
        metrics = {
            'mae': mean_absolute_error(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'r2': r2_score(y, predictions),
            'mape': np.mean(np.abs((y - predictions) / y)) * 100  # Mean Absolute Percentage Error
        }
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features
            
        Returns:
            Predicted ETAs in minutes
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df


def train_eta_model(
    experiment_name: str = "eta-prediction",
    run_name: Optional[str] = None
) -> str:
    """
    Main training pipeline for ETA prediction model.
    
    Args:
        experiment_name: MLflow experiment name
        run_name: Optional run name
        
    Returns:
        MLflow run ID
    """
    # Initialize MLflow tracker
    mlflow_tracker = MLflowTracker(experiment_name=experiment_name)
    run_id = mlflow_tracker.start_run(run_name=run_name or "eta_training")
    
    try:
        # Generate or load training data
        # In production, this would query actual historical trips
        entity_df = pd.DataFrame({
            'trip_id': [f'trip_{i}' for i in range(10000)],
            'rider_id': [f'rider_{i%1000}' for i in range(10000)],
            'driver_id': [f'driver_{i%500}' for i in range(10000)],
            'event_timestamp': pd.date_range('2024-01-01', periods=10000, freq='1H'),
            'actual_duration_minutes': np.random.uniform(5, 60, 10000)  # Target variable
        })
        
        # Initialize predictor
        predictor = ETAPredictor(
            model_params={
                'objective': 'reg:squarederror',
                'max_depth': 8,
                'learning_rate': 0.1,
                'n_estimators': 200,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
        )
        
        # Log model parameters
        mlflow_tracker.log_params(predictor.model_params)
        
        # Prepare features
        X, y = predictor.prepare_features(entity_df)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Validation set: {len(X_val)} samples")
        
        # Train model
        metrics = predictor.train(X_train, y_train, X_val, y_val)
        
        # Log metrics to MLflow
        mlflow_tracker.log_metrics(metrics)
        
        # Log feature importance
        importance_df = predictor.get_feature_importance()
        logger.info(f"\nTop 5 Important Features:\n{importance_df.head()}")
        
        # Save feature importance as artifact
        importance_path = "/tmp/feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        mlflow_tracker.log_artifact(importance_path)
        
        # Log model to MLflow
        model_uri = mlflow_tracker.log_model(
            predictor.model,
            artifact_path="model",
            registered_model_name="eta-prediction-model"
        )
        
        logger.info(f"Model logged: {model_uri}")
        
        # Transition to staging if performance is good
        if metrics['mae'] < 5.0:  # MAE < 5 minutes
            version = mlflow_tracker.register_model(
                model_uri=model_uri,
                model_name="eta-prediction-model"
            )
            
            mlflow_tracker.transition_model_stage(
                model_name="eta-prediction-model",
                version=version,
                stage="Staging"
            )
            
            logger.info(f"Model promoted to Staging: version {version}")
        
        mlflow_tracker.end_run()
        
        logger.info("ETA model training completed successfully")
        return run_id
    
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        mlflow_tracker.end_run()
        raise


if __name__ == "__main__":
    # Run training
    run_id = train_eta_model()
    print(f"Training completed. Run ID: {run_id}")
