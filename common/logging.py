"""
Centralized logging configuration for the Uber ML Platform.

Provides structured logging with context propagation for distributed tracing
and audit trail generation.
"""

import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict, Optional
from contextvars import ContextVar
from functools import wraps
import traceback


# Context variables for request tracing
request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
user_id_ctx: ContextVar[Optional[str]] = ContextVar("user_id", default=None)
model_id_ctx: ContextVar[Optional[str]] = ContextVar("model_id", default=None)


class StructuredFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    
    Outputs logs in JSON format with consistent fields for easy parsing
    by log aggregation systems like ELK, Splunk, or CloudWatch.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add context variables if set
        if request_id := request_id_ctx.get():
            log_data["request_id"] = request_id
        
        if user_id := user_id_ctx.get():
            log_data["user_id"] = user_id
        
        if model_id := model_id_ctx.get():
            log_data["model_id"] = model_id
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields from record
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)
        
        return json.dumps(log_data)


class AuditLogger:
    """
    Specialized logger for audit trail and compliance tracking.
    
    Records all model operations, data access, and governance actions
    for regulatory compliance (GDPR, SOC2, etc.).
    """
    
    def __init__(self, name: str = "audit"):
        self.logger = logging.getLogger(f"audit.{name}")
    
    def log_model_training_start(
        self,
        model_name: str,
        model_version: str,
        dataset_id: str,
        user_id: str,
        **kwargs
    ):
        """Log the start of model training."""
        self.logger.info(
            "Model training started",
            extra={
                "extra_fields": {
                    "event_type": "model_training_start",
                    "model_name": model_name,
                    "model_version": model_version,
                    "dataset_id": dataset_id,
                    "user_id": user_id,
                    **kwargs
                }
            }
        )
    
    def log_model_deployment(
        self,
        model_name: str,
        model_version: str,
        environment: str,
        deployed_by: str,
        approval_id: Optional[str] = None,
        **kwargs
    ):
        """Log model deployment to production or staging."""
        self.logger.info(
            "Model deployed",
            extra={
                "extra_fields": {
                    "event_type": "model_deployment",
                    "model_name": model_name,
                    "model_version": model_version,
                    "environment": environment,
                    "deployed_by": deployed_by,
                    "approval_id": approval_id,
                    **kwargs
                }
            }
        )
    
    def log_data_access(
        self,
        dataset_id: str,
        user_id: str,
        access_type: str,
        purpose: str,
        **kwargs
    ):
        """Log data access for privacy compliance."""
        self.logger.info(
            "Data accessed",
            extra={
                "extra_fields": {
                    "event_type": "data_access",
                    "dataset_id": dataset_id,
                    "user_id": user_id,
                    "access_type": access_type,
                    "purpose": purpose,
                    **kwargs
                }
            }
        )
    
    def log_prediction_request(
        self,
        model_name: str,
        model_version: str,
        request_id: str,
        features_hash: str,
        **kwargs
    ):
        """Log inference request for audit trail."""
        self.logger.info(
            "Prediction request",
            extra={
                "extra_fields": {
                    "event_type": "prediction_request",
                    "model_name": model_name,
                    "model_version": model_version,
                    "request_id": request_id,
                    "features_hash": features_hash,
                    **kwargs
                }
            }
        )


def setup_logging(
    level: str = "INFO",
    format_json: bool = True,
    log_file: Optional[str] = None
) -> None:
    """
    Configure application-wide logging.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_json: Whether to use JSON structured logging
        log_file: Optional file path for log output
    """
    log_level = getattr(logging, level.upper())
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    if format_json:
        console_handler.setFormatter(StructuredFormatter())
    else:
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
    
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(file_handler)
    
    # Suppress noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name."""
    return logging.getLogger(name)


def log_execution_time(logger: Optional[logging.Logger] = None):
    """
    Decorator to log function execution time.
    
    Usage:
        @log_execution_time()
        def my_function():
            pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.utcnow()
            log = logger or get_logger(func.__module__)
            
            try:
                result = func(*args, **kwargs)
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                
                log.info(
                    f"{func.__name__} completed",
                    extra={
                        "extra_fields": {
                            "function": func.__name__,
                            "execution_time_seconds": execution_time,
                            "status": "success"
                        }
                    }
                )
                return result
            
            except Exception as e:
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                
                log.error(
                    f"{func.__name__} failed",
                    extra={
                        "extra_fields": {
                            "function": func.__name__,
                            "execution_time_seconds": execution_time,
                            "status": "error",
                            "error": str(e)
                        }
                    },
                    exc_info=True
                )
                raise
        
        return wrapper
    return decorator


# Global audit logger instance
audit_logger = AuditLogger()


__all__ = [
    "setup_logging",
    "get_logger",
    "log_execution_time",
    "audit_logger",
    "request_id_ctx",
    "user_id_ctx",
    "model_id_ctx",
    "AuditLogger",
]
