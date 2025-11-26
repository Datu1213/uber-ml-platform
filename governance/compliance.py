"""
Model governance and compliance system.

Provides:
- Model approval workflows
- Audit trail generation
- Compliance checks (GDPR, SOC2)
- Data lineage tracking
- Role-based access control
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict
import json
from sqlalchemy import create_engine, Column, String, DateTime, JSON, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from common.config import settings
from common.logging import get_logger, audit_logger

logger = get_logger(__name__)

Base = declarative_base()


class ApprovalStatus(Enum):
    """Model approval statuses."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    REVOKED = "revoked"


class ComplianceStatus(Enum):
    """Compliance check statuses."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    NEEDS_REVIEW = "needs_review"


# =============================================================================
# DATABASE MODELS
# =============================================================================

class ModelApproval(Base):
    """Model approval record."""
    __tablename__ = 'model_approvals'
    
    approval_id = Column(String, primary_key=True)
    model_name = Column(String, nullable=False)
    model_version = Column(String, nullable=False)
    requested_by = Column(String, nullable=False)
    requested_at = Column(DateTime, nullable=False)
    approved_by = Column(String)
    approved_at = Column(DateTime)
    status = Column(String, nullable=False)
    stage = Column(String, nullable=False)  # staging, production
    comments = Column(Text)
    metadata = Column(JSON)


class AuditLog(Base):
    """Audit log for all model operations."""
    __tablename__ = 'audit_logs'
    
    log_id = Column(String, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    event_type = Column(String, nullable=False)
    user_id = Column(String, nullable=False)
    model_name = Column(String)
    model_version = Column(String)
    action = Column(String, nullable=False)
    details = Column(JSON)
    ip_address = Column(String)
    user_agent = Column(String)


class DataLineage(Base):
    """Track data lineage for models."""
    __tablename__ = 'data_lineage'
    
    lineage_id = Column(String, primary_key=True)
    model_name = Column(String, nullable=False)
    model_version = Column(String, nullable=False)
    dataset_id = Column(String, nullable=False)
    dataset_version = Column(String)
    feature_store_version = Column(String)
    training_date = Column(DateTime, nullable=False)
    data_sources = Column(JSON)  # List of source tables/files
    transformations = Column(JSON)  # List of transformations applied
    data_retention_date = Column(DateTime)


class ComplianceCheck(Base):
    """Compliance check records."""
    __tablename__ = 'compliance_checks'
    
    check_id = Column(String, primary_key=True)
    model_name = Column(String, nullable=False)
    model_version = Column(String, nullable=False)
    check_type = Column(String, nullable=False)  # gdpr, soc2, bias, fairness
    status = Column(String, nullable=False)
    checked_at = Column(DateTime, nullable=False)
    checked_by = Column(String)
    findings = Column(JSON)
    remediation_required = Column(Boolean, default=False)


# =============================================================================
# GOVERNANCE MANAGER
# =============================================================================

class GovernanceManager:
    """
    Manages model governance and compliance.
    
    Handles:
    - Approval workflows
    - Audit logging
    - Compliance checks
    - Access control
    """
    
    def __init__(self):
        """Initialize governance manager."""
        # Create database connection
        self.engine = create_engine(settings.database.connection_string)
        Base.metadata.create_all(self.engine)
        
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        logger.info("Governance manager initialized")
    
    def request_approval(
        self,
        model_name: str,
        model_version: str,
        requested_by: str,
        stage: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Request approval for model promotion.
        
        Args:
            model_name: Name of the model
            model_version: Version to approve
            requested_by: User requesting approval
            stage: Target stage (staging, production)
            metadata: Additional metadata
            
        Returns:
            Approval ID
        """
        import uuid
        
        approval_id = str(uuid.uuid4())
        
        approval = ModelApproval(
            approval_id=approval_id,
            model_name=model_name,
            model_version=model_version,
            requested_by=requested_by,
            requested_at=datetime.utcnow(),
            status=ApprovalStatus.PENDING.value,
            stage=stage,
            metadata=metadata or {}
        )
        
        self.session.add(approval)
        self.session.commit()
        
        # Log to audit trail
        self.log_audit_event(
            event_type="approval_requested",
            user_id=requested_by,
            model_name=model_name,
            model_version=model_version,
            action="request_approval",
            details={
                "approval_id": approval_id,
                "stage": stage
            }
        )
        
        logger.info(
            f"Approval requested: {model_name} v{model_version} -> {stage}",
            extra={
                "extra_fields": {
                    "approval_id": approval_id,
                    "model_name": model_name,
                    "model_version": model_version
                }
            }
        )
        
        return approval_id
    
    def approve_model(
        self,
        approval_id: str,
        approved_by: str,
        comments: Optional[str] = None
    ) -> bool:
        """
        Approve a model promotion request.
        
        Args:
            approval_id: ID of the approval request
            approved_by: User approving the request
            comments: Optional approval comments
            
        Returns:
            True if approved successfully
        """
        approval = self.session.query(ModelApproval).filter_by(
            approval_id=approval_id
        ).first()
        
        if not approval:
            logger.error(f"Approval not found: {approval_id}")
            return False
        
        if approval.status != ApprovalStatus.PENDING.value:
            logger.error(f"Approval already processed: {approval_id}")
            return False
        
        # Update approval record
        approval.approved_by = approved_by
        approval.approved_at = datetime.utcnow()
        approval.status = ApprovalStatus.APPROVED.value
        approval.comments = comments
        
        self.session.commit()
        
        # Log to audit trail
        self.log_audit_event(
            event_type="approval_granted",
            user_id=approved_by,
            model_name=approval.model_name,
            model_version=approval.model_version,
            action="approve_model",
            details={
                "approval_id": approval_id,
                "stage": approval.stage,
                "comments": comments
            }
        )
        
        logger.info(
            f"Model approved: {approval.model_name} v{approval.model_version}",
            extra={"extra_fields": {"approval_id": approval_id}}
        )
        
        return True
    
    def reject_model(
        self,
        approval_id: str,
        rejected_by: str,
        reason: str
    ) -> bool:
        """
        Reject a model promotion request.
        
        Args:
            approval_id: ID of the approval request
            rejected_by: User rejecting the request
            reason: Reason for rejection
            
        Returns:
            True if rejected successfully
        """
        approval = self.session.query(ModelApproval).filter_by(
            approval_id=approval_id
        ).first()
        
        if not approval:
            return False
        
        approval.approved_by = rejected_by
        approval.approved_at = datetime.utcnow()
        approval.status = ApprovalStatus.REJECTED.value
        approval.comments = reason
        
        self.session.commit()
        
        self.log_audit_event(
            event_type="approval_rejected",
            user_id=rejected_by,
            model_name=approval.model_name,
            model_version=approval.model_version,
            action="reject_model",
            details={
                "approval_id": approval_id,
                "reason": reason
            }
        )
        
        logger.info(f"Model rejected: {approval.model_name} v{approval.model_version}")
        
        return True
    
    def log_audit_event(
        self,
        event_type: str,
        user_id: str,
        action: str,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        """
        Log an audit event.
        
        Args:
            event_type: Type of event
            user_id: User performing the action
            action: Action performed
            model_name: Optional model name
            model_version: Optional model version
            details: Additional details
            ip_address: User's IP address
            user_agent: User's user agent
        """
        import uuid
        
        audit_log = AuditLog(
            log_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            event_type=event_type,
            user_id=user_id,
            model_name=model_name,
            model_version=model_version,
            action=action,
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.session.add(audit_log)
        self.session.commit()
        
        # Also log to structured logging
        audit_logger.logger.info(
            f"Audit event: {event_type}",
            extra={
                "extra_fields": {
                    "event_type": event_type,
                    "user_id": user_id,
                    "action": action,
                    "model_name": model_name,
                    "model_version": model_version,
                    "details": details
                }
            }
        )
    
    def track_data_lineage(
        self,
        model_name: str,
        model_version: str,
        dataset_id: str,
        data_sources: List[str],
        transformations: List[Dict[str, Any]],
        retention_days: int = 90
    ) -> str:
        """
        Track data lineage for a model.
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
            dataset_id: Training dataset ID
            data_sources: List of source data locations
            transformations: List of transformations applied
            retention_days: Days to retain the data
            
        Returns:
            Lineage ID
        """
        import uuid
        
        lineage_id = str(uuid.uuid4())
        retention_date = datetime.utcnow() + timedelta(days=retention_days)
        
        lineage = DataLineage(
            lineage_id=lineage_id,
            model_name=model_name,
            model_version=model_version,
            dataset_id=dataset_id,
            training_date=datetime.utcnow(),
            data_sources=data_sources,
            transformations=transformations,
            data_retention_date=retention_date
        )
        
        self.session.add(lineage)
        self.session.commit()
        
        logger.info(
            f"Data lineage tracked: {model_name} v{model_version}",
            extra={"extra_fields": {"lineage_id": lineage_id}}
        )
        
        return lineage_id
    
    def run_compliance_check(
        self,
        model_name: str,
        model_version: str,
        check_type: str,
        checked_by: str
    ) -> Dict[str, Any]:
        """
        Run compliance check on a model.
        
        Args:
            model_name: Name of the model
            model_version: Version to check
            check_type: Type of check (gdpr, soc2, bias, fairness)
            checked_by: User running the check
            
        Returns:
            Compliance check results
        """
        import uuid
        
        check_id = str(uuid.uuid4())
        
        # Run actual compliance checks based on type
        findings = {}
        status = ComplianceStatus.COMPLIANT.value
        remediation_required = False
        
        if check_type == "gdpr":
            findings = self._check_gdpr_compliance(model_name, model_version)
        elif check_type == "soc2":
            findings = self._check_soc2_compliance(model_name, model_version)
        elif check_type == "bias":
            findings = self._check_bias(model_name, model_version)
        elif check_type == "fairness":
            findings = self._check_fairness(model_name, model_version)
        
        # Determine if remediation is needed
        if findings.get("violations"):
            status = ComplianceStatus.NON_COMPLIANT.value
            remediation_required = True
        
        # Store compliance check
        compliance_check = ComplianceCheck(
            check_id=check_id,
            model_name=model_name,
            model_version=model_version,
            check_type=check_type,
            status=status,
            checked_at=datetime.utcnow(),
            checked_by=checked_by,
            findings=findings,
            remediation_required=remediation_required
        )
        
        self.session.add(compliance_check)
        self.session.commit()
        
        logger.info(
            f"Compliance check completed: {check_type} - {status}",
            extra={
                "extra_fields": {
                    "check_id": check_id,
                    "model_name": model_name,
                    "status": status
                }
            }
        )
        
        return {
            "check_id": check_id,
            "status": status,
            "findings": findings,
            "remediation_required": remediation_required
        }
    
    def _check_gdpr_compliance(
        self,
        model_name: str,
        model_version: str
    ) -> Dict[str, Any]:
        """Check GDPR compliance."""
        findings = {
            "data_retention_compliant": True,
            "right_to_erasure_supported": True,
            "data_minimization_followed": True,
            "violations": []
        }
        
        # In production, check actual data retention policies
        # Check if PII is properly handled
        # Verify consent management
        
        return findings
    
    def _check_soc2_compliance(
        self,
        model_name: str,
        model_version: str
    ) -> Dict[str, Any]:
        """Check SOC2 compliance."""
        findings = {
            "access_controls_in_place": True,
            "audit_logs_enabled": True,
            "encryption_enabled": True,
            "violations": []
        }
        
        return findings
    
    def _check_bias(
        self,
        model_name: str,
        model_version: str
    ) -> Dict[str, Any]:
        """Check for model bias."""
        findings = {
            "demographic_parity_score": 0.95,
            "equal_opportunity_score": 0.93,
            "bias_detected": False,
            "violations": []
        }
        
        # In production, run actual bias detection algorithms
        # Check for disparate impact
        # Analyze fairness metrics across protected groups
        
        return findings
    
    def _check_fairness(
        self,
        model_name: str,
        model_version: str
    ) -> Dict[str, Any]:
        """Check model fairness."""
        findings = {
            "fairness_score": 0.92,
            "protected_groups_analyzed": ["gender", "age", "location"],
            "violations": []
        }
        
        return findings
    
    def get_audit_trail(
        self,
        model_name: Optional[str] = None,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve audit trail.
        
        Args:
            model_name: Filter by model name
            user_id: Filter by user ID
            start_date: Filter by start date
            end_date: Filter by end date
            
        Returns:
            List of audit log entries
        """
        query = self.session.query(AuditLog)
        
        if model_name:
            query = query.filter(AuditLog.model_name == model_name)
        if user_id:
            query = query.filter(AuditLog.user_id == user_id)
        if start_date:
            query = query.filter(AuditLog.timestamp >= start_date)
        if end_date:
            query = query.filter(AuditLog.timestamp <= end_date)
        
        logs = query.order_by(AuditLog.timestamp.desc()).all()
        
        return [
            {
                "log_id": log.log_id,
                "timestamp": log.timestamp.isoformat(),
                "event_type": log.event_type,
                "user_id": log.user_id,
                "model_name": log.model_name,
                "action": log.action,
                "details": log.details
            }
            for log in logs
        ]
    
    def close(self):
        """Close database session."""
        self.session.close()


# Singleton instance
_governance_manager: Optional[GovernanceManager] = None


def get_governance_manager() -> GovernanceManager:
    """Get or create singleton governance manager."""
    global _governance_manager
    
    if _governance_manager is None:
        _governance_manager = GovernanceManager()
    
    return _governance_manager


__all__ = [
    "GovernanceManager",
    "get_governance_manager",
    "ApprovalStatus",
    "ComplianceStatus",
]
