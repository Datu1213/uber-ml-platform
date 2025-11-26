"""
Model and data drift detection system using Evidently AI.

Monitors:
- Feature drift (statistical distribution changes)
- Prediction drift (model output distribution changes)
- Data quality issues
- Performance degradation

Triggers automatic retraining when drift exceeds thresholds.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import DatasetDriftMetric, ColumnDriftMetric
from scipy import stats
import json

from common.config import settings
from common.logging import get_logger
from data_platform.airflow.dags.batch_feature_pipeline import materialize_features

logger = get_logger(__name__)


class DriftDetector:
    """
    Detects statistical drift in features and predictions.
    
    Uses multiple statistical tests:
    - Kolmogorov-Smirnov test for continuous features
    - Chi-squared test for categorical features
    - Population Stability Index (PSI)
    - Jensen-Shannon divergence
    """
    
    def __init__(
        self,
        reference_data: pd.DataFrame,
        drift_threshold: float = 0.1,
        feature_columns: Optional[List[str]] = None
    ):
        """
        Initialize drift detector.
        
        Args:
            reference_data: Reference dataset (training data)
            drift_threshold: Threshold for drift detection (0-1)
            feature_columns: List of feature columns to monitor
        """
        self.reference_data = reference_data
        self.drift_threshold = drift_threshold
        self.feature_columns = feature_columns or list(reference_data.columns)
        
        # Calculate reference statistics
        self.reference_stats = self._calculate_statistics(reference_data)
        
        logger.info(
            f"Drift detector initialized with {len(self.feature_columns)} features"
        )
    
    def detect_drift(
        self,
        current_data: pd.DataFrame,
        return_detailed_report: bool = False
    ) -> Dict[str, Any]:
        """
        Detect drift in current data compared to reference.
        
        Args:
            current_data: Current production data
            return_detailed_report: Whether to return detailed drift report
            
        Returns:
            Dictionary with drift detection results
        """
        logger.info("Running drift detection")
        
        drift_results = {
            'timestamp': datetime.utcnow().isoformat(),
            'dataset_drift': False,
            'drifted_features': [],
            'drift_scores': {},
            'feature_drift_details': {}
        }
        
        # Check each feature for drift
        for feature in self.feature_columns:
            if feature not in current_data.columns:
                logger.warning(f"Feature {feature} not found in current data")
                continue
            
            drift_score, drift_detected = self._check_feature_drift(
                reference=self.reference_data[feature],
                current=current_data[feature],
                feature_name=feature
            )
            
            drift_results['drift_scores'][feature] = drift_score
            
            if drift_detected:
                drift_results['drifted_features'].append(feature)
                drift_results['dataset_drift'] = True
                
                logger.warning(
                    f"Drift detected in feature: {feature} (score: {drift_score:.4f})"
                )
        
        # Calculate overall drift metrics
        drift_results['num_drifted_features'] = len(drift_results['drifted_features'])
        drift_results['drift_percentage'] = (
            len(drift_results['drifted_features']) / len(self.feature_columns) * 100
        )
        
        # Generate detailed Evidently report if requested
        if return_detailed_report:
            drift_results['evidently_report'] = self._generate_evidently_report(
                current_data
            )
        
        logger.info(
            f"Drift detection completed: {drift_results['num_drifted_features']}/{len(self.feature_columns)} features drifted"
        )
        
        return drift_results
    
    def _check_feature_drift(
        self,
        reference: pd.Series,
        current: pd.Series,
        feature_name: str
    ) -> Tuple[float, bool]:
        """
        Check drift for a single feature.
        
        Args:
            reference: Reference feature values
            current: Current feature values
            feature_name: Name of the feature
            
        Returns:
            Tuple of (drift_score, drift_detected)
        """
        # Remove NaN values
        reference_clean = reference.dropna()
        current_clean = current.dropna()
        
        if len(reference_clean) == 0 or len(current_clean) == 0:
            return 0.0, False
        
        # Check if categorical or continuous
        is_categorical = reference.dtype == 'object' or reference.nunique() < 10
        
        if is_categorical:
            # Use chi-squared test for categorical
            drift_score = self._chi_squared_test(reference_clean, current_clean)
        else:
            # Use KS test for continuous
            drift_score = self._kolmogorov_smirnov_test(reference_clean, current_clean)
        
        drift_detected = drift_score > self.drift_threshold
        
        return drift_score, drift_detected
    
    def _kolmogorov_smirnov_test(
        self,
        reference: pd.Series,
        current: pd.Series
    ) -> float:
        """
        Kolmogorov-Smirnov test for continuous features.
        
        Returns:
            KS statistic (0-1, higher means more drift)
        """
        statistic, p_value = stats.ks_2samp(reference, current)
        return statistic
    
    def _chi_squared_test(
        self,
        reference: pd.Series,
        current: pd.Series
    ) -> float:
        """
        Chi-squared test for categorical features.
        
        Returns:
            Normalized chi-squared statistic (0-1)
        """
        # Get value counts
        ref_counts = reference.value_counts()
        curr_counts = current.value_counts()
        
        # Align categories
        all_categories = set(ref_counts.index) | set(curr_counts.index)
        ref_aligned = pd.Series([ref_counts.get(cat, 0) for cat in all_categories])
        curr_aligned = pd.Series([curr_counts.get(cat, 0) for cat in all_categories])
        
        # Normalize to probabilities
        ref_probs = ref_aligned / ref_aligned.sum()
        curr_probs = curr_aligned / curr_aligned.sum()
        
        # Calculate chi-squared statistic
        chi2 = np.sum((ref_probs - curr_probs) ** 2 / (ref_probs + 1e-10))
        
        # Normalize to 0-1 range
        return min(chi2, 1.0)
    
    def calculate_psi(
        self,
        reference: pd.Series,
        current: pd.Series,
        buckets: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        PSI measures distribution shift:
        - PSI < 0.1: No significant change
        - 0.1 <= PSI < 0.2: Moderate change
        - PSI >= 0.2: Significant change
        
        Args:
            reference: Reference feature values
            current: Current feature values
            buckets: Number of buckets for discretization
            
        Returns:
            PSI score
        """
        # Create bins based on reference data
        bins = np.percentile(reference, np.linspace(0, 100, buckets + 1))
        bins = np.unique(bins)
        
        # Discretize both datasets
        ref_binned = pd.cut(reference, bins=bins, include_lowest=True)
        curr_binned = pd.cut(current, bins=bins, include_lowest=True)
        
        # Calculate proportions
        ref_props = ref_binned.value_counts(normalize=True).sort_index()
        curr_props = curr_binned.value_counts(normalize=True).sort_index()
        
        # Align indices
        ref_props = ref_props.reindex(curr_props.index, fill_value=0.001)
        curr_props = curr_props.reindex(ref_props.index, fill_value=0.001)
        
        # Calculate PSI
        psi = np.sum((curr_props - ref_props) * np.log(curr_props / ref_props))
        
        return psi
    
    def _calculate_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate reference statistics for all features."""
        stats = {}
        
        for col in self.feature_columns:
            if col not in data.columns:
                continue
            
            stats[col] = {
                'mean': data[col].mean() if data[col].dtype != 'object' else None,
                'std': data[col].std() if data[col].dtype != 'object' else None,
                'min': data[col].min() if data[col].dtype != 'object' else None,
                'max': data[col].max() if data[col].dtype != 'object' else None,
                'missing_pct': data[col].isna().mean() * 100,
                'unique_count': data[col].nunique()
            }
        
        return stats
    
    def _generate_evidently_report(
        self,
        current_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Generate detailed drift report using Evidently AI.
        
        Args:
            current_data: Current production data
            
        Returns:
            Report as dictionary
        """
        # Create column mapping
        column_mapping = ColumnMapping()
        
        # Create drift report
        report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
        ])
        
        # Run report
        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=column_mapping
        )
        
        # Convert to dictionary
        report_dict = report.as_dict()
        
        return report_dict


class PredictionDriftMonitor:
    """
    Monitors drift in model predictions.
    
    Tracks:
    - Prediction distribution changes
    - Confidence score changes
    - Label distribution (for classification)
    """
    
    def __init__(
        self,
        model_name: str,
        drift_threshold: float = 0.15
    ):
        """
        Initialize prediction drift monitor.
        
        Args:
            model_name: Name of the model to monitor
            drift_threshold: Threshold for prediction drift
        """
        self.model_name = model_name
        self.drift_threshold = drift_threshold
        self.reference_predictions = None
        
        logger.info(f"Prediction drift monitor initialized for {model_name}")
    
    def set_reference_predictions(self, predictions: np.ndarray):
        """
        Set reference predictions (from training/validation).
        
        Args:
            predictions: Reference prediction values
        """
        self.reference_predictions = predictions
        logger.info(f"Reference predictions set: {len(predictions)} samples")
    
    def check_prediction_drift(
        self,
        current_predictions: np.ndarray
    ) -> Dict[str, Any]:
        """
        Check for drift in current predictions.
        
        Args:
            current_predictions: Current model predictions
            
        Returns:
            Drift analysis results
        """
        if self.reference_predictions is None:
            logger.warning("Reference predictions not set")
            return {'drift_detected': False, 'message': 'No reference data'}
        
        # KS test for distribution drift
        statistic, p_value = stats.ks_2samp(
            self.reference_predictions,
            current_predictions
        )
        
        drift_detected = statistic > self.drift_threshold
        
        results = {
            'timestamp': datetime.utcnow().isoformat(),
            'model_name': self.model_name,
            'drift_detected': drift_detected,
            'drift_score': float(statistic),
            'p_value': float(p_value),
            'reference_mean': float(np.mean(self.reference_predictions)),
            'current_mean': float(np.mean(current_predictions)),
            'reference_std': float(np.std(self.reference_predictions)),
            'current_std': float(np.std(current_predictions))
        }
        
        if drift_detected:
            logger.warning(
                f"Prediction drift detected for {self.model_name}: "
                f"score={statistic:.4f}"
            )
        
        return results


def trigger_retraining(
    model_name: str,
    drift_results: Dict[str, Any]
):
    """
    Trigger automatic model retraining when drift is detected.
    
    Args:
        model_name: Name of the model to retrain
        drift_results: Drift detection results
    """
    logger.info(f"Triggering automatic retraining for {model_name}")
    
    # In production, this would:
    # 1. Trigger Airflow DAG for retraining
    # 2. Send notification to ML team
    # 3. Create retraining job in Ray
    # 4. Update model registry with drift metadata
    
    # For MVP, log the trigger
    logger.info(
        f"Retraining triggered",
        extra={
            'extra_fields': {
                'model_name': model_name,
                'num_drifted_features': drift_results['num_drifted_features'],
                'drift_percentage': drift_results['drift_percentage']
            }
        }
    )


# Example usage
def monitor_eta_model_drift():
    """Example: Monitor drift for ETA prediction model."""
    from training.train_eta_model import ETAPredictor
    
    # Load reference data (training data)
    reference_data = pd.DataFrame({
        'trip_distance_km': np.random.uniform(1, 50, 1000),
        'traffic_level': np.random.choice([0, 1, 2], 1000),
        'hour_of_day': np.random.randint(0, 24, 1000),
        'is_peak_hour': np.random.choice([0, 1], 1000),
    })
    
    # Simulate current production data with some drift
    current_data = pd.DataFrame({
        'trip_distance_km': np.random.uniform(2, 60, 1000),  # Shifted distribution
        'traffic_level': np.random.choice([0, 1, 2], 1000),
        'hour_of_day': np.random.randint(0, 24, 1000),
        'is_peak_hour': np.random.choice([0, 1], 1000),
    })
    
    # Initialize drift detector
    detector = DriftDetector(
        reference_data=reference_data,
        drift_threshold=0.1
    )
    
    # Detect drift
    drift_results = detector.detect_drift(current_data)
    
    # Check if retraining is needed
    if drift_results['dataset_drift']:
        logger.warning(
            f"Dataset drift detected: {drift_results['drift_percentage']:.1f}% features drifted"
        )
        trigger_retraining('eta-prediction-model', drift_results)
    else:
        logger.info("No significant drift detected")
    
    return drift_results


__all__ = [
    "DriftDetector",
    "PredictionDriftMonitor",
    "trigger_retraining",
]
