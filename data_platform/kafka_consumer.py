"""
Kafka consumer for real-time feature engineering.

Consumes events from Kafka, processes them for feature extraction,
and stores features in the online feature store (Redis).
"""

from typing import Dict, Any, List, Optional, Callable
import json
from datetime import datetime, timedelta
from kafka import KafkaConsumer
from kafka.errors import KafkaError
import redis
from collections import defaultdict

from common.config import settings
from common.logging import get_logger
from data_platform.kafka_producer import EventType

logger = get_logger(__name__)


class FeatureProcessor:
    """
    Processes raw events into ML features.
    
    Implements real-time feature engineering including:
    - Aggregations (count, sum, avg over time windows)
    - Geographic features (distance, location density)
    - Temporal features (hour, day_of_week, is_peak_hour)
    - Historical features (user behavior patterns)
    """
    
    def __init__(self, redis_client: redis.Redis):
        """
        Initialize feature processor.
        
        Args:
            redis_client: Redis client for online feature storage
        """
        self.redis = redis_client
        self.feature_ttl = 86400  # 24 hours
    
    def process_rider_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features from rider event.
        
        Args:
            event: Rider event dictionary
            
        Returns:
            Dictionary of computed features
        """
        rider_id = event['rider_id']
        timestamp = datetime.fromisoformat(event['timestamp'])
        
        features = {
            # Temporal features
            'hour_of_day': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'is_peak_hour': int(timestamp.hour in [7, 8, 9, 17, 18, 19]),
            'is_weekend': int(timestamp.weekday() >= 5),
            
            # Location features
            'pickup_lat': event['latitude'],
            'pickup_lng': event['longitude'],
        }
        
        # Calculate trip distance if destination provided
        if event.get('destination_lat') and event.get('destination_lng'):
            features['trip_distance_km'] = self._calculate_distance(
                event['latitude'], event['longitude'],
                event['destination_lat'], event['destination_lng']
            )
        
        # Historical aggregations (last 7 days, 30 days)
        features['rider_trips_7d'] = self._get_aggregation(
            rider_id, 'trips', days=7
        )
        features['rider_trips_30d'] = self._get_aggregation(
            rider_id, 'trips', days=30
        )
        features['rider_avg_rating'] = self._get_avg_rating(rider_id)
        
        # Update aggregation counters
        self._increment_counter(rider_id, 'trips', timestamp)
        
        return features
    
    def process_driver_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features from driver event.
        
        Args:
            event: Driver event dictionary
            
        Returns:
            Dictionary of computed features
        """
        driver_id = event['driver_id']
        timestamp = datetime.fromisoformat(event['timestamp'])
        
        features = {
            # Driver attributes
            'driver_status': event['status'],
            'vehicle_type': event['vehicle_type'],
            'current_lat': event['latitude'],
            'current_lng': event['longitude'],
            
            # Temporal features
            'hour_of_day': timestamp.hour,
            'day_of_week': timestamp.weekday(),
        }
        
        # Historical metrics
        features['driver_trips_today'] = self._get_aggregation(
            driver_id, 'trips', days=1
        )
        features['driver_trips_7d'] = self._get_aggregation(
            driver_id, 'trips', days=7
        )
        features['driver_acceptance_rate_7d'] = self._get_acceptance_rate(
            driver_id, days=7
        )
        features['driver_avg_rating'] = self._get_avg_rating(driver_id)
        
        return features
    
    def process_trip_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features from trip event.
        
        Args:
            event: Trip event dictionary
            
        Returns:
            Dictionary of computed features
        """
        features = {
            'trip_id': event['trip_id'],
            'pickup_lat': event['pickup_lat'],
            'pickup_lng': event['pickup_lng'],
            'dropoff_lat': event['dropoff_lat'],
            'dropoff_lng': event['dropoff_lng'],
        }
        
        # Calculate trip distance
        features['trip_distance_km'] = self._calculate_distance(
            event['pickup_lat'], event['pickup_lng'],
            event['dropoff_lat'], event['dropoff_lng']
        )
        
        if event['event_type'] == EventType.TRIP_END.value:
            features['actual_duration'] = event.get('duration_minutes')
            features['actual_fare'] = event.get('actual_fare')
            
            # Update historical counters
            self._increment_counter(event['rider_id'], 'trips', 
                                   datetime.fromisoformat(event['timestamp']))
            self._increment_counter(event['driver_id'], 'trips',
                                   datetime.fromisoformat(event['timestamp']))
        
        return features
    
    def _calculate_distance(
        self,
        lat1: float, lng1: float,
        lat2: float, lng2: float
    ) -> float:
        """
        Calculate haversine distance between two coordinates.
        
        Args:
            lat1, lng1: First coordinate
            lat2, lng2: Second coordinate
            
        Returns:
            Distance in kilometers
        """
        from math import radians, sin, cos, sqrt, atan2
        
        R = 6371  # Earth radius in km
        
        dlat = radians(lat2 - lat1)
        dlng = radians(lng2 - lng1)
        
        a = (sin(dlat / 2) ** 2 +
             cos(radians(lat1)) * cos(radians(lat2)) * sin(dlng / 2) ** 2)
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        
        return R * c
    
    def _get_aggregation(
        self,
        entity_id: str,
        metric: str,
        days: int
    ) -> int:
        """Get aggregated count over time window."""
        key = f"agg:{entity_id}:{metric}:{days}d"
        value = self.redis.get(key)
        return int(value) if value else 0
    
    def _increment_counter(
        self,
        entity_id: str,
        metric: str,
        timestamp: datetime
    ):
        """Increment time-windowed counters."""
        # Increment various time windows
        for days in [1, 7, 30]:
            key = f"agg:{entity_id}:{metric}:{days}d"
            self.redis.incr(key)
            self.redis.expire(key, days * 86400)
    
    def _get_avg_rating(self, entity_id: str) -> float:
        """Get average rating for entity."""
        key = f"rating:{entity_id}"
        value = self.redis.get(key)
        return float(value) if value else 4.5  # Default rating
    
    def _get_acceptance_rate(self, driver_id: str, days: int) -> float:
        """Calculate driver acceptance rate over time window."""
        accepted = self._get_aggregation(driver_id, 'accepted', days)
        total = self._get_aggregation(driver_id, 'requests', days)
        return (accepted / total) if total > 0 else 0.0
    
    def store_features(
        self,
        entity_id: str,
        entity_type: str,
        features: Dict[str, Any]
    ):
        """
        Store computed features in Redis.
        
        Args:
            entity_id: ID of the entity (rider, driver, trip)
            entity_type: Type of entity
            features: Dictionary of features
        """
        key = f"features:{entity_type}:{entity_id}"
        
        # Store as JSON with TTL
        self.redis.setex(
            key,
            self.feature_ttl,
            json.dumps(features)
        )
        
        logger.debug(
            f"Features stored for {entity_type}:{entity_id}",
            extra={"extra_fields": {"feature_count": len(features)}}
        )


class RealTimeFeatureConsumer:
    """
    Kafka consumer for real-time feature engineering pipeline.
    
    Consumes events, processes features, and stores in online feature store.
    """
    
    def __init__(
        self,
        topics: List[str],
        group_id: str = "feature-engineering",
        bootstrap_servers: Optional[List[str]] = None
    ):
        """
        Initialize feature consumer.
        
        Args:
            topics: List of Kafka topics to consume
            group_id: Consumer group ID
            bootstrap_servers: Kafka broker addresses
        """
        servers = bootstrap_servers or settings.kafka.bootstrap_servers
        
        self.consumer = KafkaConsumer(
            *topics,
            bootstrap_servers=servers,
            group_id=group_id,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='latest',
            enable_auto_commit=True,
            max_poll_records=100
        )
        
        # Initialize Redis for feature storage
        redis_client = redis.Redis(
            host=settings.redis.host,
            port=settings.redis.port,
            db=settings.redis.db,
            decode_responses=True
        )
        
        self.processor = FeatureProcessor(redis_client)
        self.metrics = defaultdict(int)
        
        logger.info(f"Feature consumer initialized for topics: {topics}")
    
    def start(self):
        """Start consuming and processing events."""
        logger.info("Starting feature consumer...")
        
        try:
            for message in self.consumer:
                self._process_message(message)
                
                # Log metrics periodically
                self.metrics['total_processed'] += 1
                if self.metrics['total_processed'] % 1000 == 0:
                    logger.info(
                        "Processing metrics",
                        extra={"extra_fields": dict(self.metrics)}
                    )
        
        except KeyboardInterrupt:
            logger.info("Consumer interrupted by user")
        
        except Exception as e:
            logger.error(f"Consumer error: {e}", exc_info=True)
            raise
        
        finally:
            self.close()
    
    def _process_message(self, message):
        """Process a single Kafka message."""
        try:
            event = message.value
            event_type = event.get('event_type', '')
            
            # Route to appropriate processor
            if event_type.startswith('rider'):
                features = self.processor.process_rider_event(event)
                entity_id = event['rider_id']
                entity_type = 'rider'
            
            elif event_type.startswith('driver'):
                features = self.processor.process_driver_event(event)
                entity_id = event['driver_id']
                entity_type = 'driver'
            
            elif event_type.startswith('trip'):
                features = self.processor.process_trip_event(event)
                entity_id = event['trip_id']
                entity_type = 'trip'
            
            else:
                logger.warning(f"Unknown event type: {event_type}")
                return
            
            # Store features in online store
            self.processor.store_features(entity_id, entity_type, features)
            
            self.metrics[f'{entity_type}_events'] += 1
        
        except Exception as e:
            logger.error(
                f"Failed to process message: {e}",
                extra={
                    "extra_fields": {
                        "topic": message.topic,
                        "partition": message.partition,
                        "offset": message.offset
                    }
                },
                exc_info=True
            )
            self.metrics['errors'] += 1
    
    def close(self):
        """Close consumer and release resources."""
        self.consumer.close()
        logger.info("Feature consumer closed")


__all__ = [
    "RealTimeFeatureConsumer",
    "FeatureProcessor",
]
