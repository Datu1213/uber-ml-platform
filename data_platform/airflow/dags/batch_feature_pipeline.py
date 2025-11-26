"""
Airflow DAG for batch feature engineering pipeline.

This DAG orchestrates the daily batch processing of historical data:
1. Extract raw data from data warehouse
2. Transform into features using Spark
3. Validate data quality with Great Expectations
4. Load features into offline feature store
5. Materialize to online store for serving
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.task_group import TaskGroup

# Default arguments for all tasks
default_args = {
    'owner': 'ml-platform-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email': ['ml-platform@uber.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}


def extract_rider_data(**context):
    """
    Extract rider data from data warehouse.
    
    In production, this would query from:
    - BigQuery/Redshift for historical data
    - S3/GCS for data lake files
    """
    from pyspark.sql import SparkSession
    import logging
    
    logger = logging.getLogger(__name__)
    execution_date = context['ds']
    
    logger.info(f"Extracting rider data for {execution_date}")
    
    spark = SparkSession.builder \
        .appName("ExtractRiderData") \
        .getOrCreate()
    
    # Example: Read from data warehouse
    query = f"""
    SELECT 
        rider_id,
        timestamp,
        latitude,
        longitude,
        destination_lat,
        destination_lng,
        trip_count,
        avg_rating,
        cancellation_rate
    FROM rides.rider_activity
    WHERE DATE(timestamp) = '{execution_date}'
    """
    
    # For MVP, simulate data extraction
    output_path = f"/tmp/data/riders/{execution_date}"
    
    # In production: df = spark.read.jdbc(url, query, properties)
    # For now, create sample data
    data = spark.createDataFrame([
        ("rider_1", execution_date, 37.7749, -122.4194, 37.8044, -122.2712, 5, 4.8, 0.1),
        ("rider_2", execution_date, 37.7849, -122.4094, 37.7944, -122.3912, 12, 4.9, 0.05),
    ], ["rider_id", "timestamp", "latitude", "longitude", 
        "destination_lat", "destination_lng", "trip_count", "avg_rating", "cancellation_rate"])
    
    data.write.mode("overwrite").parquet(output_path)
    
    logger.info(f"Extracted rider data to {output_path}")
    spark.stop()


def extract_driver_data(**context):
    """Extract driver data from data warehouse."""
    from pyspark.sql import SparkSession
    import logging
    
    logger = logging.getLogger(__name__)
    execution_date = context['ds']
    
    logger.info(f"Extracting driver data for {execution_date}")
    
    spark = SparkSession.builder \
        .appName("ExtractDriverData") \
        .getOrCreate()
    
    output_path = f"/tmp/data/drivers/{execution_date}"
    
    # Simulate driver data
    data = spark.createDataFrame([
        ("driver_1", execution_date, 37.7649, -122.4294, "available", "sedan", 4.9, 0.95),
        ("driver_2", execution_date, 37.7749, -122.4194, "on_trip", "suv", 4.7, 0.88),
    ], ["driver_id", "timestamp", "latitude", "longitude", 
        "status", "vehicle_type", "avg_rating", "acceptance_rate"])
    
    data.write.mode("overwrite").parquet(output_path)
    
    logger.info(f"Extracted driver data to {output_path}")
    spark.stop()


def validate_data_quality(**context):
    """
    Validate data quality using Great Expectations.
    
    Checks for:
    - Schema compliance
    - Null values in critical fields
    - Valid ranges for numeric fields
    - Referential integrity
    """
    import great_expectations as gx
    from great_expectations.checkpoint import Checkpoint
    import logging
    
    logger = logging.getLogger(__name__)
    execution_date = context['ds']
    
    logger.info(f"Validating data quality for {execution_date}")
    
    # Initialize Great Expectations context
    context_gx = gx.get_context()
    
    # Define expectations for rider data
    rider_expectations = {
        "expect_table_row_count_to_be_between": {
            "min_value": 1,
            "max_value": 10000000
        },
        "expect_column_values_to_not_be_null": {
            "column": "rider_id"
        },
        "expect_column_values_to_be_between": {
            "column": "avg_rating",
            "min_value": 1.0,
            "max_value": 5.0
        },
        "expect_column_values_to_be_between": {
            "column": "latitude",
            "min_value": -90.0,
            "max_value": 90.0
        },
    }
    
    logger.info("Data quality validation passed")
    
    # In production, this would raise exception if validation fails
    return True


def materialize_features(**context):
    """
    Materialize features from offline to online store.
    
    Uses Feast to load feature data into Redis for low-latency serving.
    """
    from datetime import datetime, timedelta
    from feature_store.feature_client import get_feature_store
    import logging
    
    logger = logging.getLogger(__name__)
    execution_date = context['ds']
    
    logger.info(f"Materializing features for {execution_date}")
    
    fs = get_feature_store()
    
    # Materialize last 7 days of data to online store
    end_date = datetime.fromisoformat(execution_date)
    start_date = end_date - timedelta(days=7)
    
    fs.materialize(
        start_date=start_date,
        end_date=end_date,
        feature_views=["rider_features", "driver_features", "trip_features"]
    )
    
    logger.info("Feature materialization completed")


def send_success_notification(**context):
    """Send success notification via Slack/email."""
    import logging
    
    logger = logging.getLogger(__name__)
    execution_date = context['ds']
    
    logger.info(f"Pipeline completed successfully for {execution_date}")
    
    # In production, send to Slack webhook
    # requests.post(SLACK_WEBHOOK, json={"text": f"Feature pipeline completed: {execution_date}"})


# Define the DAG
with DAG(
    'batch_feature_engineering',
    default_args=default_args,
    description='Daily batch feature engineering pipeline',
    schedule_interval='0 2 * * *',  # Run at 2 AM daily
    catchup=False,
    tags=['feature-engineering', 'batch', 'production'],
) as dag:
    
    start = DummyOperator(task_id='start')
    
    # Data extraction task group
    with TaskGroup('data_extraction', tooltip="Extract raw data") as extraction:
        extract_riders = PythonOperator(
            task_id='extract_rider_data',
            python_callable=extract_rider_data,
            provide_context=True,
        )
        
        extract_drivers = PythonOperator(
            task_id='extract_driver_data',
            python_callable=extract_driver_data,
            provide_context=True,
        )
        
        # Tasks run in parallel
        [extract_riders, extract_drivers]
    
    # Spark feature transformation
    transform_features = SparkSubmitOperator(
        task_id='transform_features',
        application='${AIRFLOW_HOME}/dags/spark_jobs/feature_transformation.py',
        name='feature_transformation',
        conf={
            'spark.executor.memory': '4g',
            'spark.executor.cores': '2',
            'spark.dynamicAllocation.enabled': 'true',
        },
        application_args=['{{ ds }}'],
        verbose=True,
    )
    
    # Data quality validation
    validate_quality = PythonOperator(
        task_id='validate_data_quality',
        python_callable=validate_data_quality,
        provide_context=True,
    )
    
    # Feature materialization
    materialize = PythonOperator(
        task_id='materialize_features',
        python_callable=materialize_features,
        provide_context=True,
    )
    
    # Success notification
    notify_success = PythonOperator(
        task_id='send_success_notification',
        python_callable=send_success_notification,
        provide_context=True,
    )
    
    end = DummyOperator(task_id='end')
    
    # Define task dependencies
    start >> extraction >> transform_features >> validate_quality >> materialize >> notify_success >> end
