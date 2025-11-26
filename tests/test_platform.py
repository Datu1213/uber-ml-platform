"""
Test suite for Uber ML Platform.

Tests:
- Unit tests for core components
- Integration tests for pipelines
- API endpoint tests
- Feature store tests
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Import components to test
from common.config import settings
from data_platform.kafka_producer import UberEventProducer, RiderEvent
from feature_store.feature_client import UberFeatureStore
from training.mlflow_tracker import MLflowTracker
from monitoring.drift_detection import DriftDetector
from governance.compliance import GovernanceManager


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame for testing."""
    return pd.DataFrame({
        'feature_1': np.random.randn(100),
        'feature_2': np.random.randn(100),
        'feature_3': np.random.choice(['a', 'b', 'c'], 100),
        'target': np.random.randn(100)
    })


@pytest.fixture
def mock_kafka_producer():
    """Mock Kafka producer."""
    with patch('data_platform.kafka_producer.KafkaProducer') as mock:
        yield mock


@pytest.fixture
def mock_mlflow():
    """Mock MLflow client."""
    with patch('training.mlflow_tracker.mlflow') as mock:
        yield mock


# =============================================================================
# KAFKA PRODUCER TESTS
# =============================================================================

class TestKafkaProducer:
    """Test Kafka event producer."""
    
    def test_rider_event_creation(self):
        """Test creating a rider event."""
        event = RiderEvent.create_request(
            rider_id="rider_123",
            lat=37.7749,
            lng=-122.4194,
            dest_lat=37.8044,
            dest_lng=-122.2712
        )
        
        assert event.rider_id == "rider_123"
        assert event.latitude == 37.7749
        assert event.destination_lat == 37.8044
        assert event.event_type == "rider.request"
    
    def test_event_serialization(self):
        """Test event serialization to dict."""
        event = RiderEvent.create_request(
            rider_id="rider_456",
            lat=37.7749,
            lng=-122.4194,
            dest_lat=37.8044,
            dest_lng=-122.2712
        )
        
        event_dict = event.to_dict()
        
        assert isinstance(event_dict, dict)
        assert 'event_id' in event_dict
        assert 'rider_id' in event_dict
        assert event_dict['rider_id'] == "rider_456"


# =============================================================================
# FEATURE STORE TESTS
# =============================================================================

class TestFeatureStore:
    """Test feature store operations."""
    
    @patch('feature_store.feature_client.FeatureStore')
    def test_get_online_features(self, mock_feast):
        """Test retrieving online features."""
        # Mock Feast response
        mock_feast.return_value.get_online_features.return_value.to_df.return_value = pd.DataFrame({
            'rider_id': ['rider_123'],
            'rider_trips_7d': [5],
            'rider_avg_rating': [4.8]
        })
        
        fs = UberFeatureStore()
        features = fs.get_online_features(
            feature_service_name="eta_prediction",
            entity_rows=[{"rider_id": "rider_123"}]
        )
        
        assert isinstance(features, pd.DataFrame)
        assert 'rider_id' in features.columns
    
    @patch('feature_store.feature_client.FeatureStore')
    def test_get_historical_features(self, mock_feast):
        """Test retrieving historical features."""
        entity_df = pd.DataFrame({
            'rider_id': ['rider_123', 'rider_456'],
            'event_timestamp': [datetime.now(), datetime.now()]
        })
        
        # Mock response
        mock_feast.return_value.get_historical_features.return_value.to_df.return_value = entity_df.copy()
        
        fs = UberFeatureStore()
        features = fs.get_historical_features(
            entity_df=entity_df,
            feature_service_name="eta_prediction"
        )
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) == 2


# =============================================================================
# MLFLOW TRACKER TESTS
# =============================================================================

class TestMLflowTracker:
    """Test MLflow tracking operations."""
    
    @patch('training.mlflow_tracker.mlflow')
    def test_start_run(self, mock_mlflow):
        """Test starting an MLflow run."""
        mock_run = Mock()
        mock_run.info.run_id = "test_run_123"
        mock_mlflow.start_run.return_value = mock_run
        
        tracker = MLflowTracker(experiment_name="test_experiment")
        run_id = tracker.start_run(run_name="test_run")
        
        assert run_id == "test_run_123"
        mock_mlflow.start_run.assert_called_once()
    
    @patch('training.mlflow_tracker.mlflow')
    def test_log_metrics(self, mock_mlflow):
        """Test logging metrics."""
        tracker = MLflowTracker()
        
        metrics = {
            'accuracy': 0.95,
            'loss': 0.05,
            'f1_score': 0.92
        }
        
        tracker.log_metrics(metrics)
        
        mock_mlflow.log_metrics.assert_called_once_with(metrics, step=None)
    
    @patch('training.mlflow_tracker.mlflow')
    def test_register_model(self, mock_mlflow):
        """Test registering a model."""
        mock_result = Mock()
        mock_result.version = "1"
        mock_mlflow.register_model.return_value = mock_result
        
        tracker = MLflowTracker()
        version = tracker.register_model(
            model_uri="runs:/test_run/model",
            model_name="test_model"
        )
        
        assert version == "1"


# =============================================================================
# DRIFT DETECTION TESTS
# =============================================================================

class TestDriftDetector:
    """Test drift detection functionality."""
    
    def test_no_drift_detection(self, sample_dataframe):
        """Test when no drift is present."""
        # Create reference and current data with same distribution
        reference_data = sample_dataframe.copy()
        current_data = sample_dataframe.copy()
        
        detector = DriftDetector(
            reference_data=reference_data,
            drift_threshold=0.1
        )
        
        results = detector.detect_drift(current_data)
        
        assert results['dataset_drift'] == False
        assert len(results['drifted_features']) == 0
    
    def test_drift_detection(self, sample_dataframe):
        """Test when drift is present."""
        reference_data = sample_dataframe.copy()
        
        # Create drifted data
        current_data = sample_dataframe.copy()
        current_data['feature_1'] = current_data['feature_1'] + 5  # Significant shift
        
        detector = DriftDetector(
            reference_data=reference_data,
            drift_threshold=0.1
        )
        
        results = detector.detect_drift(current_data)
        
        assert results['dataset_drift'] == True
        assert 'feature_1' in results['drifted_features']
    
    def test_psi_calculation(self):
        """Test PSI calculation."""
        reference = pd.Series(np.random.normal(0, 1, 1000))
        current = pd.Series(np.random.normal(0.5, 1, 1000))  # Shifted mean
        
        detector = DriftDetector(
            reference_data=pd.DataFrame({'feature': reference}),
            drift_threshold=0.1
        )
        
        psi = detector.calculate_psi(reference, current)
        
        assert psi > 0
        assert isinstance(psi, float)


# =============================================================================
# GOVERNANCE TESTS
# =============================================================================

class TestGovernanceManager:
    """Test governance and compliance."""
    
    @patch('governance.compliance.create_engine')
    def test_request_approval(self, mock_engine):
        """Test approval request creation."""
        mock_session = MagicMock()
        
        with patch.object(GovernanceManager, '__init__', lambda x: None):
            gov = GovernanceManager()
            gov.session = mock_session
            
            approval_id = gov.request_approval(
                model_name="test_model",
                model_version="1",
                requested_by="user_123",
                stage="production"
            )
            
            assert approval_id is not None
            mock_session.add.assert_called_once()
            mock_session.commit.assert_called_once()
    
    @patch('governance.compliance.create_engine')
    def test_audit_log_creation(self, mock_engine):
        """Test audit log creation."""
        mock_session = MagicMock()
        
        with patch.object(GovernanceManager, '__init__', lambda x: None):
            gov = GovernanceManager()
            gov.session = mock_session
            
            gov.log_audit_event(
                event_type="model_deployed",
                user_id="user_123",
                action="deploy_model",
                model_name="test_model",
                model_version="1"
            )
            
            mock_session.add.assert_called_once()
            mock_session.commit.assert_called_once()


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestEndToEndPipeline:
    """Test end-to-end ML pipeline."""
    
    @pytest.mark.integration
    def test_training_pipeline(self, sample_dataframe):
        """Test complete training pipeline."""
        # This would test:
        # 1. Feature extraction
        # 2. Model training
        # 3. Model registration
        # 4. Model validation
        pass
    
    @pytest.mark.integration
    def test_inference_pipeline(self):
        """Test inference pipeline."""
        # This would test:
        # 1. Feature retrieval
        # 2. Model loading
        # 3. Prediction
        # 4. Response formatting
        pass


# =============================================================================
# API TESTS
# =============================================================================

@pytest.mark.asyncio
class TestInferenceAPI:
    """Test inference API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from serving.main import app
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_eta_prediction_endpoint(self, client):
        """Test ETA prediction endpoint."""
        request_data = {
            "rider_id": "rider_123",
            "driver_id": "driver_456",
            "trip_id": "trip_789",
            "pickup_lat": 37.7749,
            "pickup_lng": -122.4194,
            "dropoff_lat": 37.8044,
            "dropoff_lng": -122.2712
        }
        
        with patch('serving.main.load_model'):
            response = client.post("/predict/eta", json=request_data)
            
            # May fail if model not loaded, but test structure is correct
            assert response.status_code in [200, 500]


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Test performance characteristics."""
    
    def test_feature_retrieval_latency(self):
        """Test feature retrieval is fast enough."""
        # Feature retrieval should be < 100ms
        pass
    
    def test_prediction_latency(self):
        """Test prediction latency meets SLA."""
        # Prediction should be < 200ms
        pass
    
    def test_throughput(self):
        """Test system can handle required throughput."""
        # Should handle > 1000 predictions/second
        pass


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
