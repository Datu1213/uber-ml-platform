"""
Common configuration management for the Uber ML Platform.

This module provides centralized configuration using Pydantic for type safety
and environment variable management.
"""

from typing import Optional, List
from pydantic import BaseModel, Field, PostgresDsn, RedisDsn
from pydantic_settings import BaseSettings, SettingsConfigDict
import os


class DatabaseConfig(BaseModel):
    """Database configuration for PostgreSQL."""
    host: str = "localhost"
    port: int = 5432
    user: str = "uber_ml"
    password: str = "uber_ml_pass"
    database: str = "ml_platform"
    
    @property
    def connection_string(self) -> str:
        """Generate SQLAlchemy connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class RedisConfig(BaseModel):
    """Redis configuration for caching and feature serving."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    
    @property
    def connection_string(self) -> str:
        """Generate Redis connection string."""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


class KafkaConfig(BaseModel):
    """Kafka configuration for event streaming."""
    bootstrap_servers: List[str] = ["localhost:9094"]
    
    # Topic names for different event types
    rider_events_topic: str = "rider.events"
    driver_events_topic: str = "driver.events"
    courier_events_topic: str = "courier.events"
    trip_events_topic: str = "trip.events"
    
    # Consumer group IDs
    feature_consumer_group: str = "feature-engineering"
    monitoring_consumer_group: str = "model-monitoring"


class MLflowConfig(BaseModel):
    """MLflow configuration for experiment tracking."""
    tracking_uri: str = "http://localhost:5000"
    artifact_location: str = "./mlruns"
    registry_uri: Optional[str] = None
    
    # Experiment names
    eta_prediction_experiment: str = "eta-prediction"
    surge_pricing_experiment: str = "surge-pricing"
    fraud_detection_experiment: str = "fraud-detection"


class FeastConfig(BaseModel):
    """Feast feature store configuration."""
    registry_path: str = "feature_store/feature_repo/feature_store.yaml"
    online_store_type: str = "redis"
    offline_store_type: str = "postgresql"


class RayConfig(BaseModel):
    """Ray configuration for distributed training."""
    address: Optional[str] = None  # None means local mode
    num_cpus: Optional[int] = None
    num_gpus: Optional[int] = None
    dashboard_port: int = 8265


class ServingConfig(BaseModel):
    """Configuration for model serving layer."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False
    
    # Model serving settings
    model_cache_size: int = 10
    prediction_timeout_seconds: float = 5.0
    batch_size: int = 32


class MonitoringConfig(BaseModel):
    """Configuration for monitoring and alerting."""
    prometheus_port: int = 9090
    grafana_port: int = 3000
    
    # Drift detection thresholds
    feature_drift_threshold: float = 0.1
    prediction_drift_threshold: float = 0.15
    
    # Alert settings
    alert_email: Optional[str] = None
    slack_webhook: Optional[str] = None


class GovernanceConfig(BaseModel):
    """Configuration for model governance and compliance."""
    enable_audit_logging: bool = True
    require_approval: bool = True
    
    # Compliance settings
    data_retention_days: int = 90
    enable_encryption: bool = True
    
    # Model registry stages
    staging_approval_required: bool = True
    production_approval_required: bool = True


class Settings(BaseSettings):
    """Main settings class aggregating all configurations."""
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Component configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    kafka: KafkaConfig = Field(default_factory=KafkaConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    feast: FeastConfig = Field(default_factory=FeastConfig)
    ray: RayConfig = Field(default_factory=RayConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    governance: GovernanceConfig = Field(default_factory=GovernanceConfig)
    
    # Cloud provider settings
    aws_region: str = Field(default="us-west-2", env="AWS_REGION")
    aws_access_key_id: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")
    
    # Security
    secret_key: str = Field(default="change-me-in-production", env="SECRET_KEY")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_nested_delimiter="__"
    )
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == "development"


# Global settings instance
settings = Settings()


# Export commonly used configurations
__all__ = [
    "settings",
    "DatabaseConfig",
    "RedisConfig",
    "KafkaConfig",
    "MLflowConfig",
    "FeastConfig",
    "RayConfig",
    "ServingConfig",
    "MonitoringConfig",
    "GovernanceConfig",
]
