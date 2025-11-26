"""
Spark job for batch feature transformation.

Processes raw event data into ML features at scale using PySpark.
Implements complex aggregations, window functions, and feature engineering logic.
"""

import sys
from datetime import datetime, timedelta
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, FloatType, TimestampType


def create_spark_session(app_name: str) -> SparkSession:
    """Create and configure Spark session."""
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.shuffle.partitions", "200") \
        .getOrCreate()


def compute_rider_features(spark: SparkSession, execution_date: str):
    """
    Compute rider features from historical data.
    
    Features include:
    - Trip counts over various time windows
    - Average ratings and fares
    - Cancellation rates
    - Behavioral patterns (preferred times, locations)
    """
    print(f"Computing rider features for {execution_date}")
    
    # Read raw rider data
    rider_data = spark.read.parquet(f"/tmp/data/riders/{execution_date}")
    
    # Define time windows for aggregations
    window_7d = Window.partitionBy("rider_id").orderBy("timestamp").rangeBetween(-7*86400, 0)
    window_30d = Window.partitionBy("rider_id").orderBy("timestamp").rangeBetween(-30*86400, 0)
    
    # Compute aggregated features
    rider_features = rider_data \
        .withColumn("event_timestamp", F.col("timestamp").cast(TimestampType())) \
        .withColumn("hour_of_day", F.hour("event_timestamp")) \
        .withColumn("day_of_week", F.dayofweek("event_timestamp")) \
        .withColumn("is_weekend", F.when(F.dayofweek("event_timestamp").isin([1, 7]), 1).otherwise(0)) \
        .withColumn("is_peak_hour", 
                   F.when(F.hour("event_timestamp").isin([7, 8, 9, 17, 18, 19]), 1).otherwise(0)) \
        .withColumn("rider_trips_7d", F.count("*").over(window_7d)) \
        .withColumn("rider_trips_30d", F.count("*").over(window_30d)) \
        .withColumn("rider_avg_rating", F.avg("avg_rating").over(window_30d)) \
        .withColumn("rider_cancellation_rate_7d", F.avg("cancellation_rate").over(window_7d)) \
        .withColumn("rider_avg_fare_30d", F.lit(25.50))  # Placeholder
    
    # Calculate trip distance
    rider_features = rider_features.withColumn(
        "trip_distance_km",
        F.when(
            F.col("destination_lat").isNotNull() & F.col("destination_lng").isNotNull(),
            haversine_distance_udf(
                F.col("latitude"), F.col("longitude"),
                F.col("destination_lat"), F.col("destination_lng")
            )
        ).otherwise(0.0)
    )
    
    # Determine preferred vehicle type (most frequently used)
    rider_features = rider_features.withColumn(
        "rider_preferred_vehicle_type",
        F.lit("sedan")  # Placeholder - would be computed from historical trip data
    )
    
    # Select final feature set
    final_features = rider_features.select(
        "rider_id",
        "event_timestamp",
        "rider_trips_7d",
        "rider_trips_30d",
        "rider_avg_rating",
        "rider_cancellation_rate_7d",
        "rider_avg_fare_30d",
        "rider_preferred_vehicle_type",
        "hour_of_day",
        "day_of_week",
        "is_peak_hour",
        "is_weekend"
    ).distinct()
    
    # Write to feature store offline storage
    output_path = f"/tmp/features/riders/{execution_date}"
    final_features.write.mode("overwrite").parquet(output_path)
    
    print(f"Rider features written to {output_path}")
    return final_features.count()


def compute_driver_features(spark: SparkSession, execution_date: str):
    """
    Compute driver features from historical data.
    
    Features include:
    - Trip counts and completion rates
    - Acceptance and cancellation rates
    - Average trip duration and distance
    - Online hours and activity patterns
    """
    print(f"Computing driver features for {execution_date}")
    
    # Read raw driver data
    driver_data = spark.read.parquet(f"/tmp/data/drivers/{execution_date}")
    
    # Define time windows
    window_today = Window.partitionBy("driver_id", F.to_date("timestamp"))
    window_7d = Window.partitionBy("driver_id").orderBy("timestamp").rangeBetween(-7*86400, 0)
    window_30d = Window.partitionBy("driver_id").orderBy("timestamp").rangeBetween(-30*86400, 0)
    
    # Compute features
    driver_features = driver_data \
        .withColumn("event_timestamp", F.col("timestamp").cast(TimestampType())) \
        .withColumn("driver_trips_today", F.count("*").over(window_today)) \
        .withColumn("driver_trips_7d", F.count("*").over(window_7d)) \
        .withColumn("driver_trips_30d", F.count("*").over(window_30d)) \
        .withColumn("driver_avg_rating", F.col("avg_rating")) \
        .withColumn("driver_acceptance_rate_7d", F.col("acceptance_rate")) \
        .withColumn("driver_cancellation_rate_7d", F.lit(0.05))  # Placeholder \
        .withColumn("driver_avg_trip_distance_7d", F.lit(8.5))  # Placeholder \
        .withColumn("driver_avg_trip_duration_7d", F.lit(18.3))  # Placeholder \
        .withColumn("hours_online_today", F.lit(6.5))  # Placeholder
    
    # Select final feature set
    final_features = driver_features.select(
        "driver_id",
        "event_timestamp",
        "driver_trips_today",
        "driver_trips_7d",
        "driver_trips_30d",
        "driver_avg_rating",
        "driver_acceptance_rate_7d",
        "driver_cancellation_rate_7d",
        "driver_avg_trip_distance_7d",
        "driver_avg_trip_duration_7d",
        "vehicle_type",
        "status",
        "hours_online_today"
    ).withColumnRenamed("status", "driver_status").distinct()
    
    # Write features
    output_path = f"/tmp/features/drivers/{execution_date}"
    final_features.write.mode("overwrite").parquet(output_path)
    
    print(f"Driver features written to {output_path}")
    return final_features.count()


def compute_trip_features(spark: SparkSession, execution_date: str):
    """
    Compute trip-level features.
    
    Features include:
    - Geographic features (distance, pickup/dropoff locations)
    - Temporal features (time of day, day of week)
    - Market features (surge, demand level)
    - External features (weather, traffic)
    """
    print(f"Computing trip features for {execution_date}")
    
    # In production, would join rider and driver data with trip events
    # For MVP, create synthetic trip features
    
    schema = StructType([
        StructField("trip_id", StringType(), False),
        StructField("event_timestamp", TimestampType(), False),
        StructField("pickup_lat", FloatType(), False),
        StructField("pickup_lng", FloatType(), False),
        StructField("dropoff_lat", FloatType(), False),
        StructField("dropoff_lng", FloatType(), False),
        StructField("trip_distance_km", FloatType(), False),
        StructField("estimated_duration_minutes", FloatType(), False),
        StructField("estimated_fare", FloatType(), False),
        StructField("surge_multiplier", FloatType(), False),
        StructField("weather_condition", StringType(), True),
        StructField("traffic_level", StringType(), True),
    ])
    
    # Create sample data
    data = [
        (f"trip_{i}", datetime.fromisoformat(execution_date), 
         37.7749 + i*0.01, -122.4194 + i*0.01,
         37.8044 + i*0.01, -122.2712 + i*0.01,
         8.5, 18.3, 25.50, 1.2, "clear", "moderate")
        for i in range(10)
    ]
    
    trip_features = spark.createDataFrame(data, schema)
    
    # Write features
    output_path = f"/tmp/features/trips/{execution_date}"
    trip_features.write.mode("overwrite").parquet(output_path)
    
    print(f"Trip features written to {output_path}")
    return trip_features.count()


# UDF for haversine distance calculation
@F.udf(returnType=FloatType())
def haversine_distance_udf(lat1, lng1, lat2, lng2):
    """Calculate haversine distance between two points."""
    from math import radians, sin, cos, sqrt, atan2
    
    if None in [lat1, lng1, lat2, lng2]:
        return None
    
    R = 6371  # Earth radius in km
    
    dlat = radians(lat2 - lat1)
    dlng = radians(lng2 - lng1)
    
    a = (sin(dlat / 2) ** 2 +
         cos(radians(lat1)) * cos(radians(lat2)) * sin(dlng / 2) ** 2)
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    return R * c


def main():
    """Main execution function."""
    if len(sys.argv) < 2:
        print("Usage: feature_transformation.py <execution_date>")
        sys.exit(1)
    
    execution_date = sys.argv[1]
    
    # Create Spark session
    spark = create_spark_session("FeatureTransformation")
    
    try:
        # Compute features for all entities
        rider_count = compute_rider_features(spark, execution_date)
        driver_count = compute_driver_features(spark, execution_date)
        trip_count = compute_trip_features(spark, execution_date)
        
        print(f"\nFeature transformation completed:")
        print(f"  - Rider features: {rider_count} rows")
        print(f"  - Driver features: {driver_count} rows")
        print(f"  - Trip features: {trip_count} rows")
        
    except Exception as e:
        print(f"Error during feature transformation: {e}")
        raise
    
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
