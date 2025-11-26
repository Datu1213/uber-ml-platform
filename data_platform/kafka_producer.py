"""
Kafka event producer for real-time data ingestion.

Handles production of rider, driver, courier, and trip events to Kafka topics
with schema validation and error handling.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import json
import uuid
from kafka import KafkaProducer
from kafka.errors import KafkaError

from common.config import settings
from common.logging import get_logger

logger = get_logger(__name__)


class EventType(Enum):
    """Types of events in the Uber platform."""
    RIDER_REQUEST = "rider.request"
    RIDER_LOCATION = "rider.location"
    DRIVER_LOCATION = "driver.location"
    DRIVER_STATUS = "driver.status"
    TRIP_START = "trip.start"
    TRIP_END = "trip.end"
    COURIER_PICKUP = "courier.pickup"
    COURIER_DELIVERY = "courier.delivery"


@dataclass
class BaseEvent:
    """Base class for all events."""
    event_id: str
    event_type: str
    timestamp: str
    version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return asdict(self)


@dataclass
class RiderEvent(BaseEvent):
    """Event representing rider activity."""
    rider_id: str
    latitude: float
    longitude: float
    destination_lat: Optional[float] = None
    destination_lng: Optional[float] = None
    rider_rating: Optional[float] = None
    
    @staticmethod
    def create_request(
        rider_id: str,
        lat: float,
        lng: float,
        dest_lat: float,
        dest_lng: float
    ) -> "RiderEvent":
        """Create a rider request event."""
        return RiderEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.RIDER_REQUEST.value,
            timestamp=datetime.utcnow().isoformat(),
            rider_id=rider_id,
            latitude=lat,
            longitude=lng,
            destination_lat=dest_lat,
            destination_lng=dest_lng
        )


@dataclass
class DriverEvent(BaseEvent):
    """Event representing driver activity."""
    driver_id: str
    latitude: float
    longitude: float
    status: str  # available, on_trip, offline
    vehicle_type: str
    driver_rating: Optional[float] = None
    
    @staticmethod
    def create_location_update(
        driver_id: str,
        lat: float,
        lng: float,
        status: str,
        vehicle_type: str
    ) -> "DriverEvent":
        """Create a driver location update event."""
        return DriverEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.DRIVER_LOCATION.value,
            timestamp=datetime.utcnow().isoformat(),
            driver_id=driver_id,
            latitude=lat,
            longitude=lng,
            status=status,
            vehicle_type=vehicle_type
        )


@dataclass
class TripEvent(BaseEvent):
    """Event representing trip lifecycle."""
    trip_id: str
    rider_id: str
    driver_id: str
    pickup_lat: float
    pickup_lng: float
    dropoff_lat: float
    dropoff_lng: float
    estimated_fare: Optional[float] = None
    actual_fare: Optional[float] = None
    distance_miles: Optional[float] = None
    duration_minutes: Optional[float] = None
    
    @staticmethod
    def create_trip_start(
        trip_id: str,
        rider_id: str,
        driver_id: str,
        pickup_lat: float,
        pickup_lng: float,
        dropoff_lat: float,
        dropoff_lng: float,
        estimated_fare: float
    ) -> "TripEvent":
        """Create a trip start event."""
        return TripEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.TRIP_START.value,
            timestamp=datetime.utcnow().isoformat(),
            trip_id=trip_id,
            rider_id=rider_id,
            driver_id=driver_id,
            pickup_lat=pickup_lat,
            pickup_lng=pickup_lng,
            dropoff_lat=dropoff_lat,
            dropoff_lng=dropoff_lng,
            estimated_fare=estimated_fare
        )


class UberEventProducer:
    """
    Kafka producer for Uber platform events.
    
    Handles serialization, partitioning, and delivery guarantees for
    all event types in the platform.
    """
    
    def __init__(self, bootstrap_servers: Optional[List[str]] = None):
        """
        Initialize Kafka producer.
        
        Args:
            bootstrap_servers: List of Kafka broker addresses
        """
        servers = bootstrap_servers or settings.kafka.bootstrap_servers
        
        self.producer = KafkaProducer(
            bootstrap_servers=servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            acks='all',  # Wait for all replicas
            retries=3,
            max_in_flight_requests_per_connection=1,  # Maintain ordering
            compression_type='gzip'
        )
        
        logger.info(f"Kafka producer initialized with servers: {servers}")
    
    def _get_topic_for_event(self, event_type: str) -> str:
        """
        Route event to appropriate topic based on event type.
        
        Args:
            event_type: Type of event
            
        Returns:
            Kafka topic name
        """
        if event_type.startswith("rider"):
            return settings.kafka.rider_events_topic
        elif event_type.startswith("driver"):
            return settings.kafka.driver_events_topic
        elif event_type.startswith("trip"):
            return settings.kafka.trip_events_topic
        elif event_type.startswith("courier"):
            return settings.kafka.courier_events_topic
        else:
            raise ValueError(f"Unknown event type: {event_type}")
    
    def produce_event(
        self,
        event: BaseEvent,
        partition_key: Optional[str] = None
    ) -> bool:
        """
        Produce event to Kafka.
        
        Args:
            event: Event object to produce
            partition_key: Optional key for partitioning (e.g., rider_id, driver_id)
            
        Returns:
            True if successfully produced, False otherwise
        """
        try:
            topic = self._get_topic_for_event(event.event_type)
            event_dict = event.to_dict()
            
            # Use partition key for consistent routing
            key = partition_key or event.event_id
            
            # Produce message asynchronously
            future = self.producer.send(
                topic=topic,
                key=key,
                value=event_dict
            )
            
            # Block for confirmation (with timeout)
            record_metadata = future.get(timeout=10)
            
            logger.info(
                f"Event produced successfully",
                extra={
                    "extra_fields": {
                        "event_id": event.event_id,
                        "event_type": event.event_type,
                        "topic": topic,
                        "partition": record_metadata.partition,
                        "offset": record_metadata.offset
                    }
                }
            )
            return True
            
        except KafkaError as e:
            logger.error(
                f"Failed to produce event: {e}",
                extra={
                    "extra_fields": {
                        "event_id": event.event_id,
                        "event_type": event.event_type,
                        "error": str(e)
                    }
                }
            )
            return False
    
    def produce_batch(
        self,
        events: List[BaseEvent],
        partition_key_fn: Optional[callable] = None
    ) -> Dict[str, int]:
        """
        Produce multiple events in batch.
        
        Args:
            events: List of events to produce
            partition_key_fn: Function to extract partition key from event
            
        Returns:
            Dictionary with success/failure counts
        """
        results = {"success": 0, "failed": 0}
        
        for event in events:
            partition_key = partition_key_fn(event) if partition_key_fn else None
            success = self.produce_event(event, partition_key)
            
            if success:
                results["success"] += 1
            else:
                results["failed"] += 1
        
        self.flush()
        
        logger.info(
            f"Batch production completed",
            extra={
                "extra_fields": {
                    "total_events": len(events),
                    "successful": results["success"],
                    "failed": results["failed"]
                }
            }
        )
        
        return results
    
    def flush(self, timeout: Optional[float] = None):
        """
        Flush pending messages.
        
        Args:
            timeout: Maximum time to wait in seconds
        """
        self.producer.flush(timeout)
    
    def close(self):
        """Close the producer and release resources."""
        self.producer.close()
        logger.info("Kafka producer closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Example usage functions for testing
def simulate_rider_request(producer: UberEventProducer):
    """Simulate a rider requesting a trip."""
    event = RiderEvent.create_request(
        rider_id="rider_123",
        lat=37.7749,
        lng=-122.4194,
        dest_lat=37.8044,
        dest_lng=-122.2712
    )
    producer.produce_event(event, partition_key="rider_123")


def simulate_driver_location(producer: UberEventProducer):
    """Simulate driver location update."""
    event = DriverEvent.create_location_update(
        driver_id="driver_456",
        lat=37.7849,
        lng=-122.4094,
        status="available",
        vehicle_type="sedan"
    )
    producer.produce_event(event, partition_key="driver_456")


__all__ = [
    "UberEventProducer",
    "RiderEvent",
    "DriverEvent",
    "TripEvent",
    "EventType",
]
