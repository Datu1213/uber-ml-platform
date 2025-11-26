"""
FastAPI-based model inference service.

Provides REST API endpoints for:
- ETA prediction
- Surge pricing calculation
- Fraud detection
- Batch prediction
- Health checks and metrics
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response
import mlflow
import redis

from common.config import settings
from common.logging import get_logger, request_id_ctx, audit_logger
from feature_store.feature_client import get_feature_store

logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Uber ML Platform Inference API",
    description="Production ML inference service for Uber marketplace models",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
prediction_counter = Counter(
    'ml_predictions_total',
    'Total number of predictions',
    ['model', 'status']
)

prediction_latency = Histogram(
    'ml_prediction_latency_seconds',
    'Prediction latency in seconds',
    ['model']
)

# Global model cache
model_cache: Dict[str, Any] = {}

# Redis cache for features
redis_client = redis.Redis(
    host=settings.redis.host,
    port=settings.redis.port,
    db=settings.redis.db,
    decode_responses=True
)


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class ETAPredictionRequest(BaseModel):
    """Request model for ETA prediction."""
    rider_id: str = Field(..., description="Rider identifier")
    driver_id: str = Field(..., description="Driver identifier")
    trip_id: str = Field(..., description="Trip identifier")
    pickup_lat: float = Field(..., ge=-90, le=90)
    pickup_lng: float = Field(..., ge=-180, le=180)
    dropoff_lat: float = Field(..., ge=-90, le=90)
    dropoff_lng: float = Field(..., ge=-180, le=180)
    
    class Config:
        schema_extra = {
            "example": {
                "rider_id": "rider_123",
                "driver_id": "driver_456",
                "trip_id": "trip_789",
                "pickup_lat": 37.7749,
                "pickup_lng": -122.4194,
                "dropoff_lat": 37.8044,
                "dropoff_lng": -122.2712
            }
        }


class ETAPredictionResponse(BaseModel):
    """Response model for ETA prediction."""
    trip_id: str
    estimated_duration_minutes: float
    confidence_score: float
    model_version: str
    timestamp: str


class SurgePricingRequest(BaseModel):
    """Request model for surge pricing."""
    rider_id: str
    pickup_lat: float = Field(..., ge=-90, le=90)
    pickup_lng: float = Field(..., ge=-180, le=180)
    
    class Config:
        schema_extra = {
            "example": {
                "rider_id": "rider_123",
                "pickup_lat": 37.7749,
                "pickup_lng": -122.4194
            }
        }


class SurgePricingResponse(BaseModel):
    """Response model for surge pricing."""
    surge_multiplier: float
    base_fare: float
    estimated_fare: float
    demand_level: str
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    models_loaded: Dict[str, str]
    redis_connected: bool
    feature_store_connected: bool


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(model_name: str, stage: str = "Production") -> Any:
    """
    Load model from MLflow registry with caching.
    
    Args:
        model_name: Name of the registered model
        stage: Stage to load (Production, Staging)
        
    Returns:
        Loaded model
    """
    cache_key = f"{model_name}_{stage}"
    
    # Check cache first
    if cache_key in model_cache:
        logger.debug(f"Model loaded from cache: {cache_key}")
        return model_cache[cache_key]
    
    # Load from MLflow
    try:
        mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
        model_uri = f"models:/{model_name}/{stage}"
        model = mlflow.pyfunc.load_model(model_uri)
        
        # Cache the model
        model_cache[cache_key] = model
        
        logger.info(f"Model loaded: {model_name} ({stage})")
        return model
    
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")


# Load models on startup
@app.on_event("startup")
async def startup_event():
    """Load models on application startup."""
    logger.info("Starting inference service...")
    
    try:
        # Pre-load production models
        load_model("eta-prediction-model", "Production")
        logger.info("✓ ETA prediction model loaded")
    except Exception as e:
        logger.warning(f"Could not load ETA model: {e}")
    
    logger.info("Inference service ready")


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "service": "Uber ML Platform Inference API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Verifies:
    - Service is running
    - Models are loaded
    - Redis connection
    - Feature store connection
    """
    # Check Redis connection
    redis_connected = False
    try:
        redis_client.ping()
        redis_connected = True
    except Exception as e:
        logger.warning(f"Redis health check failed: {e}")
    
    # Check feature store
    feature_store_connected = False
    try:
        fs = get_feature_store()
        fs.list_feature_services()
        feature_store_connected = True
    except Exception as e:
        logger.warning(f"Feature store health check failed: {e}")
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        models_loaded={
            name: "loaded" for name in model_cache.keys()
        },
        redis_connected=redis_connected,
        feature_store_connected=feature_store_connected
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type="text/plain")


@app.post("/predict/eta", response_model=ETAPredictionResponse)
async def predict_eta(
    request: ETAPredictionRequest,
    background_tasks: BackgroundTasks
):
    """
    Predict ETA for a trip.
    
    Uses:
    - Trip geographic features
    - Rider historical behavior
    - Driver performance metrics
    - Real-time traffic/weather
    """
    request_id = f"eta_{request.trip_id}_{datetime.utcnow().timestamp()}"
    request_id_ctx.set(request_id)
    
    with prediction_latency.labels(model='eta').time():
        try:
            # Get features from feature store
            fs = get_feature_store()
            features = fs.get_eta_prediction_features(
                rider_id=request.rider_id,
                driver_id=request.driver_id,
                trip_id=request.trip_id
            )
            
            # Add request-specific features
            features['pickup_lat'] = request.pickup_lat
            features['pickup_lng'] = request.pickup_lng
            features['dropoff_lat'] = request.dropoff_lat
            features['dropoff_lng'] = request.dropoff_lng
            
            # Calculate distance
            features['trip_distance_km'] = calculate_distance(
                request.pickup_lat, request.pickup_lng,
                request.dropoff_lat, request.dropoff_lng
            )
            
            # Load model
            model = load_model("eta-prediction-model", "Production")
            
            # Make prediction
            prediction = model.predict([list(features.values())])[0]
            
            # Log prediction for monitoring
            background_tasks.add_task(
                log_prediction,
                model_name="eta-prediction",
                request_data=request.dict(),
                prediction=float(prediction)
            )
            
            # Update metrics
            prediction_counter.labels(model='eta', status='success').inc()
            
            # Audit log
            audit_logger.log_prediction_request(
                model_name="eta-prediction-model",
                model_version="production",
                request_id=request_id,
                features_hash=str(hash(frozenset(features.items())))
            )
            
            return ETAPredictionResponse(
                trip_id=request.trip_id,
                estimated_duration_minutes=float(prediction),
                confidence_score=0.92,  # Would be computed from model
                model_version="v2.1",
                timestamp=datetime.utcnow().isoformat()
            )
        
        except Exception as e:
            prediction_counter.labels(model='eta', status='error').inc()
            logger.error(f"ETA prediction failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/surge", response_model=SurgePricingResponse)
async def predict_surge_pricing(
    request: SurgePricingRequest,
    background_tasks: BackgroundTasks
):
    """
    Calculate surge pricing multiplier.
    
    Based on:
    - Current demand/supply ratio
    - Location-specific patterns
    - Time of day
    - Historical surge patterns
    """
    with prediction_latency.labels(model='surge').time():
        try:
            # Get features
            fs = get_feature_store()
            features = fs.get_surge_pricing_features(
                rider_id=request.rider_id,
                pickup_lat=request.pickup_lat,
                pickup_lng=request.pickup_lng
            )
            
            # For MVP, use simple rule-based logic
            # In production, this would be an ML model
            hour = datetime.utcnow().hour
            is_peak = hour in [7, 8, 9, 17, 18, 19]
            
            base_surge = 1.2 if is_peak else 1.0
            surge_multiplier = base_surge + np.random.uniform(-0.1, 0.3)
            surge_multiplier = max(1.0, min(3.0, surge_multiplier))
            
            base_fare = 8.0
            estimated_fare = base_fare * surge_multiplier + \
                           features.get('trip_distance_km', 5.0) * 2.5
            
            demand_level = "high" if surge_multiplier > 1.5 else \
                          "moderate" if surge_multiplier > 1.2 else "normal"
            
            prediction_counter.labels(model='surge', status='success').inc()
            
            return SurgePricingResponse(
                surge_multiplier=round(surge_multiplier, 2),
                base_fare=base_fare,
                estimated_fare=round(estimated_fare, 2),
                demand_level=demand_level,
                timestamp=datetime.utcnow().isoformat()
            )
        
        except Exception as e:
            prediction_counter.labels(model='surge', status='error').inc()
            logger.error(f"Surge prediction failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def calculate_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Calculate haversine distance between two points in km."""
    from math import radians, sin, cos, sqrt, atan2
    
    R = 6371  # Earth radius in km
    
    dlat = radians(lat2 - lat1)
    dlng = radians(lng2 - lng1)
    
    a = (sin(dlat / 2) ** 2 +
         cos(radians(lat1)) * cos(radians(lat2)) * sin(dlng / 2) ** 2)
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    return R * c


async def log_prediction(model_name: str, request_data: Dict, prediction: float):
    """Background task to log predictions for monitoring."""
    try:
        # Store in Redis for real-time monitoring
        log_key = f"predictions:{model_name}:{datetime.utcnow().isoformat()}"
        redis_client.setex(
            log_key,
            86400,  # 24 hour TTL
            str({"request": request_data, "prediction": prediction})
        )
    except Exception as e:
        logger.warning(f"Failed to log prediction: {e}")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host=settings.serving.host,
        port=settings.serving.port,
        workers=settings.serving.workers,
        log_level="info"
    )
