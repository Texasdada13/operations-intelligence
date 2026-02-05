"""
Database Models - Operations Intelligence

SQLAlchemy models for operational data management.
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from typing import Dict, Any, Optional, List
import uuid
import json

db = SQLAlchemy()


def generate_uuid() -> str:
    return str(uuid.uuid4())


class Organization(db.Model):
    """Organization/Company for multi-tenant support."""
    __tablename__ = 'organization'

    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    name = db.Column(db.String(200), nullable=False)
    industry = db.Column(db.String(100))
    size_category = db.Column(db.String(50))  # Small, Medium, Large, Enterprise
    employee_count = db.Column(db.Integer)

    # Operational profile
    operation_type = db.Column(db.String(50))  # Manufacturing, Service, Hybrid
    shift_pattern = db.Column(db.String(50))  # Single, Double, Triple, 24/7
    working_days_per_week = db.Column(db.Integer, default=5)
    hours_per_shift = db.Column(db.Float, default=8)

    # Calculated metrics
    overall_oee = db.Column(db.Float)
    efficiency_score = db.Column(db.Float)
    risk_level = db.Column(db.String(20))

    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    departments = db.relationship('Department', backref='organization', lazy='dynamic', cascade='all, delete-orphan')
    processes = db.relationship('Process', backref='organization', lazy='dynamic', cascade='all, delete-orphan')
    resources = db.relationship('Resource', backref='organization', lazy='dynamic', cascade='all, delete-orphan')
    kpi_entries = db.relationship('KPIEntry', backref='organization', lazy='dynamic', cascade='all, delete-orphan')
    chat_sessions = db.relationship('ChatSession', backref='organization', lazy='dynamic', cascade='all, delete-orphan')

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'industry': self.industry,
            'size_category': self.size_category,
            'employee_count': self.employee_count,
            'operation_type': self.operation_type,
            'shift_pattern': self.shift_pattern,
            'overall_oee': self.overall_oee,
            'efficiency_score': self.efficiency_score,
            'risk_level': self.risk_level,
            'department_count': self.departments.count() if self.departments else 0,
            'process_count': self.processes.count() if self.processes else 0
        }


class Department(db.Model):
    """Operational department within organization."""
    __tablename__ = 'department'

    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    organization_id = db.Column(db.String(36), db.ForeignKey('organization.id'), nullable=False)
    name = db.Column(db.String(200), nullable=False)
    department_type = db.Column(db.String(50))  # Production, Quality, Maintenance, Logistics
    manager_name = db.Column(db.String(200))
    headcount = db.Column(db.Integer)
    budget = db.Column(db.Float)

    # Performance metrics
    efficiency_score = db.Column(db.Float)
    quality_score = db.Column(db.Float)
    safety_score = db.Column(db.Float)
    cost_variance = db.Column(db.Float)

    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    processes = db.relationship('Process', backref='department', lazy='dynamic')
    resources = db.relationship('Resource', backref='department', lazy='dynamic')

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'organization_id': self.organization_id,
            'name': self.name,
            'department_type': self.department_type,
            'manager_name': self.manager_name,
            'headcount': self.headcount,
            'budget': self.budget,
            'efficiency_score': self.efficiency_score,
            'quality_score': self.quality_score,
            'safety_score': self.safety_score,
            'process_count': self.processes.count() if self.processes else 0
        }


class Process(db.Model):
    """Operational process to be analyzed."""
    __tablename__ = 'process'

    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    organization_id = db.Column(db.String(36), db.ForeignKey('organization.id'), nullable=False)
    department_id = db.Column(db.String(36), db.ForeignKey('department.id'))
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    process_type = db.Column(db.String(50))  # Core, Support, Management

    # Process metrics
    target_throughput = db.Column(db.Float)  # units per hour
    actual_throughput = db.Column(db.Float)
    cycle_time = db.Column(db.Float)  # minutes
    lead_time = db.Column(db.Float)  # minutes
    setup_time = db.Column(db.Float)  # minutes

    # Quality metrics
    first_pass_yield = db.Column(db.Float)
    defect_rate = db.Column(db.Float)
    rework_rate = db.Column(db.Float)

    # Efficiency metrics
    oee = db.Column(db.Float)  # Overall Equipment Effectiveness
    availability = db.Column(db.Float)
    performance_rate = db.Column(db.Float)
    quality_rate = db.Column(db.Float)

    # Status
    status = db.Column(db.String(20))  # Optimal, Healthy, Attention Needed, At Risk, Critical
    risk_level = db.Column(db.String(20))

    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    steps = db.relationship('ProcessStep', backref='process', lazy='dynamic', cascade='all, delete-orphan')

    def calculate_oee(self):
        """Calculate OEE from components."""
        if all([self.availability, self.performance_rate, self.quality_rate]):
            self.oee = (self.availability / 100) * (self.performance_rate / 100) * (self.quality_rate / 100) * 100
        return self.oee

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'organization_id': self.organization_id,
            'department_id': self.department_id,
            'name': self.name,
            'description': self.description,
            'process_type': self.process_type,
            'target_throughput': self.target_throughput,
            'actual_throughput': self.actual_throughput,
            'cycle_time': self.cycle_time,
            'lead_time': self.lead_time,
            'first_pass_yield': self.first_pass_yield,
            'defect_rate': self.defect_rate,
            'oee': self.oee,
            'status': self.status,
            'risk_level': self.risk_level,
            'step_count': self.steps.count() if self.steps else 0
        }


class ProcessStep(db.Model):
    """Individual step within a process."""
    __tablename__ = 'process_step'

    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    process_id = db.Column(db.String(36), db.ForeignKey('process.id'), nullable=False)
    name = db.Column(db.String(200), nullable=False)
    sequence_order = db.Column(db.Integer, nullable=False)

    # Timing
    cycle_time = db.Column(db.Float)  # minutes
    wait_time = db.Column(db.Float, default=0)  # minutes
    setup_time = db.Column(db.Float, default=0)  # minutes

    # Capacity
    capacity = db.Column(db.Float)  # units per hour
    utilization = db.Column(db.Float)  # percentage

    # Quality
    defect_rate = db.Column(db.Float, default=0)  # percentage

    # Resources
    resources_required = db.Column(db.Integer, default=1)
    equipment_id = db.Column(db.String(100))

    # Status
    is_bottleneck = db.Column(db.Boolean, default=False)
    bottleneck_severity = db.Column(db.String(20))

    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'process_id': self.process_id,
            'name': self.name,
            'sequence_order': self.sequence_order,
            'cycle_time': self.cycle_time,
            'wait_time': self.wait_time,
            'capacity': self.capacity,
            'utilization': self.utilization,
            'defect_rate': self.defect_rate,
            'is_bottleneck': self.is_bottleneck,
            'bottleneck_severity': self.bottleneck_severity
        }


class Resource(db.Model):
    """Operational resource (labor, equipment, space)."""
    __tablename__ = 'resource'

    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    organization_id = db.Column(db.String(36), db.ForeignKey('organization.id'), nullable=False)
    department_id = db.Column(db.String(36), db.ForeignKey('department.id'))
    name = db.Column(db.String(200), nullable=False)
    resource_type = db.Column(db.String(50), nullable=False)  # Labor, Equipment, Space, Material

    # Capacity
    available_capacity = db.Column(db.Float)  # units per period
    allocated_capacity = db.Column(db.Float)
    utilization = db.Column(db.Float)  # percentage
    efficiency = db.Column(db.Float, default=100)  # percentage

    # Cost
    cost_per_unit = db.Column(db.Float)
    hourly_rate = db.Column(db.Float)
    fixed_cost = db.Column(db.Float)

    # Availability
    availability_hours = db.Column(db.Float)
    downtime_hours = db.Column(db.Float)
    maintenance_schedule = db.Column(db.String(100))

    # Status
    status = db.Column(db.String(20))  # Available, In Use, Maintenance, Offline

    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def calculate_utilization(self):
        """Calculate utilization percentage."""
        if self.available_capacity and self.available_capacity > 0:
            self.utilization = (self.allocated_capacity or 0) / self.available_capacity * 100
        return self.utilization

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'organization_id': self.organization_id,
            'department_id': self.department_id,
            'name': self.name,
            'resource_type': self.resource_type,
            'available_capacity': self.available_capacity,
            'allocated_capacity': self.allocated_capacity,
            'utilization': self.utilization,
            'efficiency': self.efficiency,
            'cost_per_unit': self.cost_per_unit,
            'status': self.status
        }


class KPIEntry(db.Model):
    """KPI measurement entry for tracking operational performance."""
    __tablename__ = 'kpi_entry'

    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    organization_id = db.Column(db.String(36), db.ForeignKey('organization.id'), nullable=False)
    department_id = db.Column(db.String(36), db.ForeignKey('department.id'))
    process_id = db.Column(db.String(36), db.ForeignKey('process.id'))

    # Period
    period_date = db.Column(db.Date, nullable=False)
    period_label = db.Column(db.String(50))  # e.g., "Jan 2024", "Week 1"
    period_type = db.Column(db.String(20))  # Daily, Weekly, Monthly

    # Efficiency KPIs
    oee = db.Column(db.Float)
    labor_productivity = db.Column(db.Float)
    capacity_utilization = db.Column(db.Float)
    throughput_rate = db.Column(db.Float)

    # Quality KPIs
    first_pass_yield = db.Column(db.Float)
    defect_rate = db.Column(db.Float)
    rework_rate = db.Column(db.Float)
    scrap_rate = db.Column(db.Float)

    # Delivery KPIs
    on_time_delivery = db.Column(db.Float)
    order_fulfillment = db.Column(db.Float)
    lead_time = db.Column(db.Float)

    # Cost KPIs
    cost_per_unit = db.Column(db.Float)
    waste_rate = db.Column(db.Float)
    overtime_percentage = db.Column(db.Float)

    # Safety KPIs
    incident_rate = db.Column(db.Float)
    near_miss_count = db.Column(db.Integer)
    safety_compliance = db.Column(db.Float)

    # Customer KPIs (for service operations)
    response_time = db.Column(db.Float)
    resolution_rate = db.Column(db.Float)
    customer_satisfaction = db.Column(db.Float)
    sla_compliance = db.Column(db.Float)

    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'organization_id': self.organization_id,
            'period_date': self.period_date.isoformat() if self.period_date else None,
            'period_label': self.period_label,
            'oee': self.oee,
            'labor_productivity': self.labor_productivity,
            'capacity_utilization': self.capacity_utilization,
            'first_pass_yield': self.first_pass_yield,
            'defect_rate': self.defect_rate,
            'on_time_delivery': self.on_time_delivery,
            'lead_time': self.lead_time,
            'cost_per_unit': self.cost_per_unit,
            'incident_rate': self.incident_rate,
            'customer_satisfaction': self.customer_satisfaction
        }

    def get_kpi_values(self) -> Dict[str, float]:
        """Get all KPI values as a dictionary for analysis."""
        return {
            'oee': self.oee,
            'labor_productivity': self.labor_productivity,
            'capacity_utilization': self.capacity_utilization,
            'first_pass_yield': self.first_pass_yield,
            'defect_rate': self.defect_rate,
            'rework_rate': self.rework_rate,
            'on_time_delivery': self.on_time_delivery,
            'order_fulfillment': self.order_fulfillment,
            'lead_time': self.lead_time,
            'cost_per_unit': self.cost_per_unit,
            'waste_rate': self.waste_rate,
            'incident_rate': self.incident_rate,
            'near_miss_reporting': self.safety_compliance,
            'response_time': self.response_time,
            'resolution_rate': self.resolution_rate,
            'customer_satisfaction': self.customer_satisfaction,
            'sla_compliance': self.sla_compliance
        }


class CapacityPlan(db.Model):
    """Saved capacity plan for an organization."""
    __tablename__ = 'capacity_plan'

    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    organization_id = db.Column(db.String(36), db.ForeignKey('organization.id'), nullable=False)
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)

    # Planning period
    start_date = db.Column(db.Date)
    end_date = db.Column(db.Date)
    planning_horizon = db.Column(db.Integer)  # periods

    # Summary metrics
    total_demand = db.Column(db.Float)
    total_capacity = db.Column(db.Float)
    average_utilization = db.Column(db.Float)
    peak_utilization = db.Column(db.Float)
    bottleneck_count = db.Column(db.Integer)

    # Risk assessment
    risk_level = db.Column(db.String(20))
    risk_assessment = db.Column(db.Text)

    # Recommendations (stored as JSON)
    recommendations = db.Column(db.Text)  # JSON array
    cost_summary = db.Column(db.Text)  # JSON object

    # Full plan data (stored as JSON)
    plan_data = db.Column(db.Text)  # Full plan as JSON

    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_recommendations(self, recs: List[str]):
        self.recommendations = json.dumps(recs)

    def get_recommendations(self) -> List[str]:
        return json.loads(self.recommendations) if self.recommendations else []

    def set_cost_summary(self, costs: Dict[str, float]):
        self.cost_summary = json.dumps(costs)

    def get_cost_summary(self) -> Dict[str, float]:
        return json.loads(self.cost_summary) if self.cost_summary else {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'organization_id': self.organization_id,
            'name': self.name,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'total_demand': self.total_demand,
            'total_capacity': self.total_capacity,
            'average_utilization': self.average_utilization,
            'peak_utilization': self.peak_utilization,
            'risk_level': self.risk_level,
            'recommendations': self.get_recommendations(),
            'cost_summary': self.get_cost_summary()
        }


class BenchmarkResult(db.Model):
    """Benchmark analysis result."""
    __tablename__ = 'benchmark_result'

    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    organization_id = db.Column(db.String(36), db.ForeignKey('organization.id'), nullable=False)
    benchmark_type = db.Column(db.String(50))  # operations, manufacturing, service

    # Analysis period
    analysis_date = db.Column(db.DateTime, default=datetime.utcnow)

    # Overall scores
    overall_score = db.Column(db.Float)
    overall_rating = db.Column(db.String(20))
    grade = db.Column(db.String(2))
    percentile = db.Column(db.Float)

    # Category scores (stored as JSON)
    category_scores = db.Column(db.Text)  # JSON object

    # Detailed results (stored as JSON)
    kpi_scores = db.Column(db.Text)  # JSON array
    strengths = db.Column(db.Text)  # JSON array
    improvements = db.Column(db.Text)  # JSON array
    recommendations = db.Column(db.Text)  # JSON array

    def set_category_scores(self, scores: Dict[str, Any]):
        self.category_scores = json.dumps(scores)

    def get_category_scores(self) -> Dict[str, Any]:
        return json.loads(self.category_scores) if self.category_scores else {}

    def set_strengths(self, items: List[str]):
        self.strengths = json.dumps(items)

    def get_strengths(self) -> List[str]:
        return json.loads(self.strengths) if self.strengths else []

    def set_improvements(self, items: List[str]):
        self.improvements = json.dumps(items)

    def get_improvements(self) -> List[str]:
        return json.loads(self.improvements) if self.improvements else []

    def set_recommendations(self, recs: List[str]):
        self.recommendations = json.dumps(recs)

    def get_recommendations(self) -> List[str]:
        return json.loads(self.recommendations) if self.recommendations else []

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'organization_id': self.organization_id,
            'benchmark_type': self.benchmark_type,
            'analysis_date': self.analysis_date.isoformat() if self.analysis_date else None,
            'overall_score': self.overall_score,
            'overall_rating': self.overall_rating,
            'grade': self.grade,
            'percentile': self.percentile,
            'category_scores': self.get_category_scores(),
            'strengths': self.get_strengths(),
            'improvements': self.get_improvements(),
            'recommendations': self.get_recommendations()
        }


class ChatSession(db.Model):
    """AI chat session for COO consultation."""
    __tablename__ = 'chat_session'

    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    organization_id = db.Column(db.String(36), db.ForeignKey('organization.id'))

    # Session info
    conversation_mode = db.Column(db.String(50), default='general')
    context_summary = db.Column(db.Text)

    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    messages = db.relationship('ChatMessage', backref='session', lazy='dynamic', cascade='all, delete-orphan')

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'organization_id': self.organization_id,
            'conversation_mode': self.conversation_mode,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'message_count': self.messages.count() if self.messages else 0
        }


class ChatMessage(db.Model):
    """Individual message in a chat session."""
    __tablename__ = 'chat_message'

    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    session_id = db.Column(db.String(36), db.ForeignKey('chat_session.id'), nullable=False)

    # Message content
    role = db.Column(db.String(20), nullable=False)  # user, assistant
    content = db.Column(db.Text, nullable=False)

    # Context
    context_data = db.Column(db.Text)  # JSON object with operational context

    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_context(self, context: Dict[str, Any]):
        self.context_data = json.dumps(context)

    def get_context(self) -> Dict[str, Any]:
        return json.loads(self.context_data) if self.context_data else {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'session_id': self.session_id,
            'role': self.role,
            'content': self.content,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
