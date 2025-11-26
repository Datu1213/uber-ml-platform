"""
Feast Feature Store Configuration.

Defines feature views, entities, and data sources for the Uber ML Platform.
This enables consistent online/offline feature serving.
"""

from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Float64, Int64, String
from feast import RedisOnlineStore


# =============================================================================
# ENTITIES - Define the core business objects
# =============================================================================

rider_entity = Entity(
    name="rider_id",
    description="Unique identifier for a rider",
    value_type=ValueType.STRING
)

driver_entity = Entity(
    name="driver_id", 
    description="Unique identifier for a driver",
    value_type=ValueType.STRING
)

trip_entity = Entity(
    name="trip_id",
    description="Unique identifier for a trip",
    value_type=ValueType.STRING
)


# =============================================================================
# DATA SOURCES - Define where feature data comes from
# =============================================================================

# For MVP, using file sources. In production, these would be:
# - BigQuerySource for batch features
# - KafkaSource/KinesisSource for streaming features

rider_features_source = FileSource(
    path="data/rider_features.parquet",
    timestamp_field="event_timestamp",
)

driver_features_source = FileSource(
    path="data/driver_features.parquet", 
    timestamp_field="event_timestamp",
)

trip_features_source = FileSource(
    path="data/trip_features.parquet",
    timestamp_field="event_timestamp",
)


# =============================================================================
# FEATURE VIEWS - Define feature schemas and transformations
# =============================================================================

rider_features = FeatureView(
    name="rider_features",
    entities=[rider_entity],
    ttl=timedelta(days=1),
    schema=[
        Field(name="rider_trips_7d", dtype=Int64),
        Field(name="rider_trips_30d", dtype=Int64),
        Field(name="rider_avg_rating", dtype=Float32),
        Field(name="rider_cancellation_rate_7d", dtype=Float32),
        Field(name="rider_avg_fare_30d", dtype=Float32),
        Field(name="rider_preferred_vehicle_type", dtype=String),
        Field(name="hour_of_day", dtype=Int64),
        Field(name="day_of_week", dtype=Int64),
        Field(name="is_peak_hour", dtype=Int64),
        Field(name="is_weekend", dtype=Int64),
    ],
    online=True,
    source=rider_features_source,
    tags={"team": "marketplace", "domain": "riders"},
)

driver_features = FeatureView(
    name="driver_features",
    entities=[driver_entity],
    ttl=timedelta(hours=12),
    schema=[
        Field(name="driver_trips_today", dtype=Int64),
        Field(name="driver_trips_7d", dtype=Int64),
        Field(name="driver_trips_30d", dtype=Int64),
        Field(name="driver_avg_rating", dtype=Float32),
        Field(name="driver_acceptance_rate_7d", dtype=Float32),
        Field(name="driver_cancellation_rate_7d", dtype=Float32),
        Field(name="driver_avg_trip_distance_7d", dtype=Float32),
        Field(name="driver_avg_trip_duration_7d", dtype=Float32),
        Field(name="vehicle_type", dtype=String),
        Field(name="driver_status", dtype=String),
        Field(name="hours_online_today", dtype=Float32),
    ],
    online=True,
    source=driver_features_source,
    tags={"team": "marketplace", "domain": "drivers"},
)

trip_features = FeatureView(
    name="trip_features",
    entities=[trip_entity],
    ttl=timedelta(hours=24),
    schema=[
        Field(name="pickup_lat", dtype=Float64),
        Field(name="pickup_lng", dtype=Float64),
        Field(name="dropoff_lat", dtype=Float64),
        Field(name="dropoff_lng", dtype=Float64),
        Field(name="trip_distance_km", dtype=Float32),
        Field(name="estimated_duration_minutes", dtype=Float32),
        Field(name="estimated_fare", dtype=Float32),
        Field(name="surge_multiplier", dtype=Float32),
        Field(name="weather_condition", dtype=String),
        Field(name="traffic_level", dtype=String),
    ],
    online=True,
    source=trip_features_source,
    tags={"team": "marketplace", "domain": "trips"},
)


# =============================================================================
# FEATURE SERVICES - Group features for specific ML use cases
# =============================================================================

from feast import FeatureService

# ETA Prediction Feature Service
eta_prediction_service = FeatureService(
    name="eta_prediction",
    features=[
        trip_features[["trip_distance_km", "traffic_level", "weather_condition"]],
        driver_features[["driver_avg_trip_duration_7d", "vehicle_type"]],
        rider_features[["hour_of_day", "day_of_week", "is_peak_hour"]],
    ],
    tags={"model": "eta_prediction", "version": "v2"},
)

# Surge Pricing Feature Service
surge_pricing_service = FeatureService(
    name="surge_pricing",
    features=[
        rider_features[["rider_trips_7d", "hour_of_day", "is_peak_hour"]],
        driver_features[["driver_trips_today", "driver_status"]],
        trip_features[["pickup_lat", "pickup_lng", "traffic_level"]],
    ],
    tags={"model": "surge_pricing", "version": "v1"},
)

# Fraud Detection Feature Service
fraud_detection_service = FeatureService(
    name="fraud_detection",
    features=[
        rider_features[[
            "rider_trips_7d",
            "rider_cancellation_rate_7d",
            "rider_avg_fare_30d"
        ]],
        driver_features[[
            "driver_acceptance_rate_7d",
            "driver_cancellation_rate_7d",
            "driver_avg_rating"
        ]],
        trip_features[["trip_distance_km", "estimated_fare"]],
    ],
    tags={"model": "fraud_detection", "version": "v3"},
)
