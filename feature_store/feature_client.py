"""
Feature Store Client for online and offline feature retrieval.

Provides a simplified interface to Feast for:
- Online feature serving (low latency)
- Offline feature retrieval (training)
- Feature materialization
"""

from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
from feast import FeatureStore
from feast.infra.online_stores.redis import RedisOnlineStoreConfig
from feast.repo_config import RepoConfig

from common.config import settings
from common.logging import get_logger

logger = get_logger(__name__)


class UberFeatureStore:
    """
    Wrapper around Feast FeatureStore for Uber ML Platform.
    
    Provides simplified methods for feature retrieval with built-in
    error handling, caching, and logging.
    """
    
    def __init__(self, repo_path: Optional[str] = None):
        """
        Initialize feature store client.
        
        Args:
            repo_path: Path to Feast repository configuration
        """
        self.repo_path = repo_path or settings.feast.registry_path
        
        # Initialize Feast store
        self.store = FeatureStore(repo_path=self.repo_path)
        
        logger.info(f"Feature store initialized from: {self.repo_path}")
    
    def get_online_features(
        self,
        feature_service_name: str,
        entity_rows: List[Dict[str, Any]],
        full_feature_names: bool = False
    ) -> pd.DataFrame:
        """
        Retrieve online features for real-time inference.
        
        Args:
            feature_service_name: Name of the feature service
            entity_rows: List of entity dictionaries (e.g., [{"rider_id": "123"}])
            full_feature_names: Whether to include feature view name in column names
            
        Returns:
            DataFrame with features for each entity
            
        Example:
            >>> fs = UberFeatureStore()
            >>> entities = [{"rider_id": "rider_123", "driver_id": "driver_456"}]
            >>> features = fs.get_online_features("eta_prediction", entities)
        """
        try:
            # Get feature service
            feature_service = self.store.get_feature_service(feature_service_name)
            
            # Retrieve features
            feature_vector = self.store.get_online_features(
                features=feature_service,
                entity_rows=entity_rows,
                full_feature_names=full_feature_names
            )
            
            # Convert to DataFrame
            df = feature_vector.to_df()
            
            logger.debug(
                f"Retrieved online features",
                extra={
                    "extra_fields": {
                        "feature_service": feature_service_name,
                        "num_entities": len(entity_rows),
                        "num_features": len(df.columns)
                    }
                }
            )
            
            return df
        
        except Exception as e:
            logger.error(
                f"Failed to retrieve online features: {e}",
                extra={
                    "extra_fields": {
                        "feature_service": feature_service_name,
                        "entity_count": len(entity_rows)
                    }
                },
                exc_info=True
            )
            raise
    
    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        feature_service_name: str,
        full_feature_names: bool = False
    ) -> pd.DataFrame:
        """
        Retrieve historical features for model training.
        
        Point-in-time correct join of features with entity timestamps.
        
        Args:
            entity_df: DataFrame with entity IDs and event_timestamp
            feature_service_name: Name of the feature service
            full_feature_names: Whether to include feature view name
            
        Returns:
            DataFrame with historical features joined to entities
            
        Example:
            >>> entities = pd.DataFrame({
            ...     "rider_id": ["rider_123", "rider_456"],
            ...     "event_timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2)]
            ... })
            >>> features = fs.get_historical_features(entities, "eta_prediction")
        """
        try:
            # Get feature service
            feature_service = self.store.get_feature_service(feature_service_name)
            
            # Get historical features with point-in-time correctness
            training_df = self.store.get_historical_features(
                entity_df=entity_df,
                features=feature_service,
                full_feature_names=full_feature_names
            ).to_df()
            
            logger.info(
                f"Retrieved historical features",
                extra={
                    "extra_fields": {
                        "feature_service": feature_service_name,
                        "num_rows": len(training_df),
                        "num_features": len(training_df.columns)
                    }
                }
            )
            
            return training_df
        
        except Exception as e:
            logger.error(
                f"Failed to retrieve historical features: {e}",
                extra={"extra_fields": {"feature_service": feature_service_name}},
                exc_info=True
            )
            raise
    
    def materialize(
        self,
        start_date: datetime,
        end_date: datetime,
        feature_views: Optional[List[str]] = None
    ):
        """
        Materialize features to online store.
        
        Loads feature data from offline store to online store (Redis)
        for low-latency serving.
        
        Args:
            start_date: Start of time range to materialize
            end_date: End of time range to materialize
            feature_views: Specific feature views to materialize (None = all)
        """
        try:
            self.store.materialize(
                start_date=start_date,
                end_date=end_date,
                feature_views=feature_views
            )
            
            logger.info(
                f"Features materialized to online store",
                extra={
                    "extra_fields": {
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "feature_views": feature_views or "all"
                    }
                }
            )
        
        except Exception as e:
            logger.error(f"Materialization failed: {e}", exc_info=True)
            raise
    
    def materialize_incremental(
        self,
        end_date: datetime,
        feature_views: Optional[List[str]] = None
    ):
        """
        Incrementally materialize features since last materialization.
        
        Args:
            end_date: End timestamp for incremental materialization
            feature_views: Specific feature views to materialize
        """
        try:
            self.store.materialize_incremental(
                end_date=end_date,
                feature_views=feature_views
            )
            
            logger.info(
                f"Incremental materialization completed",
                extra={
                    "extra_fields": {
                        "end_date": end_date.isoformat(),
                        "feature_views": feature_views or "all"
                    }
                }
            )
        
        except Exception as e:
            logger.error(f"Incremental materialization failed: {e}", exc_info=True)
            raise
    
    def get_eta_prediction_features(
        self,
        rider_id: str,
        driver_id: str,
        trip_id: str
    ) -> Dict[str, Any]:
        """
        Convenience method to get ETA prediction features.
        
        Args:
            rider_id: Rider identifier
            driver_id: Driver identifier
            trip_id: Trip identifier
            
        Returns:
            Dictionary of features
        """
        entity_rows = [{
            "rider_id": rider_id,
            "driver_id": driver_id,
            "trip_id": trip_id
        }]
        
        features_df = self.get_online_features("eta_prediction", entity_rows)
        return features_df.iloc[0].to_dict()
    
    def get_surge_pricing_features(
        self,
        rider_id: str,
        pickup_lat: float,
        pickup_lng: float
    ) -> Dict[str, Any]:
        """
        Convenience method to get surge pricing features.
        
        Args:
            rider_id: Rider identifier
            pickup_lat: Pickup latitude
            pickup_lng: Pickup longitude
            
        Returns:
            Dictionary of features
        """
        entity_rows = [{"rider_id": rider_id}]
        
        features_df = self.get_online_features("surge_pricing", entity_rows)
        features = features_df.iloc[0].to_dict()
        
        # Add location context
        features['pickup_lat'] = pickup_lat
        features['pickup_lng'] = pickup_lng
        
        return features
    
    def get_fraud_detection_features(
        self,
        rider_id: str,
        driver_id: str,
        trip_id: str
    ) -> Dict[str, Any]:
        """
        Convenience method to get fraud detection features.
        
        Args:
            rider_id: Rider identifier  
            driver_id: Driver identifier
            trip_id: Trip identifier
            
        Returns:
            Dictionary of features
        """
        entity_rows = [{
            "rider_id": rider_id,
            "driver_id": driver_id,
            "trip_id": trip_id
        }]
        
        features_df = self.get_online_features("fraud_detection", entity_rows)
        return features_df.iloc[0].to_dict()
    
    def list_feature_services(self) -> List[str]:
        """List all registered feature services."""
        services = [fs.name for fs in self.store.list_feature_services()]
        return services
    
    def list_feature_views(self) -> List[str]:
        """List all registered feature views."""
        views = [fv.name for fv in self.store.list_feature_views()]
        return views


# Singleton instance
_feature_store: Optional[UberFeatureStore] = None


def get_feature_store() -> UberFeatureStore:
    """Get or create singleton feature store instance."""
    global _feature_store
    
    if _feature_store is None:
        _feature_store = UberFeatureStore()
    
    return _feature_store


__all__ = [
    "UberFeatureStore",
    "get_feature_store",
]
