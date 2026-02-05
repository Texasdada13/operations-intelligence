"""
Repository Pattern - Operations Intelligence

Data access layer for operational data management.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, date
from sqlalchemy import desc, and_
import logging

from .models import (
    db,
    Organization,
    Department,
    Process,
    ProcessStep,
    Resource,
    KPIEntry,
    CapacityPlan,
    BenchmarkResult,
    ChatSession,
    ChatMessage
)

logger = logging.getLogger(__name__)


class OperationsRepository:
    """Repository for operational data access."""

    # ==================== Organization ====================

    def create_organization(self, data: Dict[str, Any]) -> Organization:
        """Create a new organization."""
        org = Organization(
            name=data['name'],
            industry=data.get('industry'),
            size_category=data.get('size_category'),
            employee_count=data.get('employee_count'),
            operation_type=data.get('operation_type'),
            shift_pattern=data.get('shift_pattern'),
            working_days_per_week=data.get('working_days_per_week', 5),
            hours_per_shift=data.get('hours_per_shift', 8)
        )
        db.session.add(org)
        db.session.commit()
        return org

    def get_organization(self, org_id: str) -> Optional[Organization]:
        """Get organization by ID."""
        return Organization.query.get(org_id)

    def get_all_organizations(self) -> List[Organization]:
        """Get all organizations."""
        return Organization.query.order_by(desc(Organization.created_at)).all()

    def update_organization(self, org_id: str, data: Dict[str, Any]) -> Optional[Organization]:
        """Update organization."""
        org = Organization.query.get(org_id)
        if not org:
            return None

        for key, value in data.items():
            if hasattr(org, key) and key != 'id':
                setattr(org, key, value)

        db.session.commit()
        return org

    def delete_organization(self, org_id: str) -> bool:
        """Delete organization and all related data."""
        org = Organization.query.get(org_id)
        if not org:
            return False

        db.session.delete(org)
        db.session.commit()
        return True

    # ==================== Department ====================

    def create_department(self, org_id: str, data: Dict[str, Any]) -> Department:
        """Create a new department."""
        dept = Department(
            organization_id=org_id,
            name=data['name'],
            department_type=data.get('department_type'),
            manager_name=data.get('manager_name'),
            headcount=data.get('headcount'),
            budget=data.get('budget')
        )
        db.session.add(dept)
        db.session.commit()
        return dept

    def get_departments(self, org_id: str) -> List[Department]:
        """Get all departments for an organization."""
        return Department.query.filter_by(organization_id=org_id).order_by(Department.name).all()

    def get_department(self, dept_id: str) -> Optional[Department]:
        """Get department by ID."""
        return Department.query.get(dept_id)

    def update_department(self, dept_id: str, data: Dict[str, Any]) -> Optional[Department]:
        """Update department."""
        dept = Department.query.get(dept_id)
        if not dept:
            return None

        for key, value in data.items():
            if hasattr(dept, key) and key not in ['id', 'organization_id']:
                setattr(dept, key, value)

        db.session.commit()
        return dept

    # ==================== Process ====================

    def create_process(self, org_id: str, data: Dict[str, Any]) -> Process:
        """Create a new process."""
        process = Process(
            organization_id=org_id,
            department_id=data.get('department_id'),
            name=data['name'],
            description=data.get('description'),
            process_type=data.get('process_type'),
            target_throughput=data.get('target_throughput'),
            actual_throughput=data.get('actual_throughput'),
            cycle_time=data.get('cycle_time'),
            lead_time=data.get('lead_time'),
            first_pass_yield=data.get('first_pass_yield'),
            defect_rate=data.get('defect_rate')
        )
        db.session.add(process)
        db.session.commit()
        return process

    def get_processes(self, org_id: str, department_id: Optional[str] = None) -> List[Process]:
        """Get all processes for an organization, optionally filtered by department."""
        query = Process.query.filter_by(organization_id=org_id)
        if department_id:
            query = query.filter_by(department_id=department_id)
        return query.order_by(Process.name).all()

    def get_process(self, process_id: str) -> Optional[Process]:
        """Get process by ID."""
        return Process.query.get(process_id)

    def update_process(self, process_id: str, data: Dict[str, Any]) -> Optional[Process]:
        """Update process."""
        process = Process.query.get(process_id)
        if not process:
            return None

        for key, value in data.items():
            if hasattr(process, key) and key not in ['id', 'organization_id']:
                setattr(process, key, value)

        # Recalculate OEE if components are updated
        process.calculate_oee()

        db.session.commit()
        return process

    def delete_process(self, process_id: str) -> bool:
        """Delete process."""
        process = Process.query.get(process_id)
        if not process:
            return False

        db.session.delete(process)
        db.session.commit()
        return True

    # ==================== Process Steps ====================

    def create_process_step(self, process_id: str, data: Dict[str, Any]) -> ProcessStep:
        """Create a process step."""
        # Get max sequence order
        max_order = db.session.query(db.func.max(ProcessStep.sequence_order)).filter_by(
            process_id=process_id
        ).scalar() or 0

        step = ProcessStep(
            process_id=process_id,
            name=data['name'],
            sequence_order=data.get('sequence_order', max_order + 1),
            cycle_time=data.get('cycle_time', 0),
            wait_time=data.get('wait_time', 0),
            setup_time=data.get('setup_time', 0),
            capacity=data.get('capacity', 100),
            utilization=data.get('utilization', 0),
            defect_rate=data.get('defect_rate', 0),
            resources_required=data.get('resources_required', 1),
            equipment_id=data.get('equipment_id')
        )
        db.session.add(step)
        db.session.commit()
        return step

    def get_process_steps(self, process_id: str) -> List[ProcessStep]:
        """Get all steps for a process, ordered by sequence."""
        return ProcessStep.query.filter_by(process_id=process_id).order_by(ProcessStep.sequence_order).all()

    def update_process_step(self, step_id: str, data: Dict[str, Any]) -> Optional[ProcessStep]:
        """Update a process step."""
        step = ProcessStep.query.get(step_id)
        if not step:
            return None

        for key, value in data.items():
            if hasattr(step, key) and key not in ['id', 'process_id']:
                setattr(step, key, value)

        db.session.commit()
        return step

    def delete_process_step(self, step_id: str) -> bool:
        """Delete a process step."""
        step = ProcessStep.query.get(step_id)
        if not step:
            return False

        db.session.delete(step)
        db.session.commit()
        return True

    # ==================== Resource ====================

    def create_resource(self, org_id: str, data: Dict[str, Any]) -> Resource:
        """Create a resource."""
        resource = Resource(
            organization_id=org_id,
            department_id=data.get('department_id'),
            name=data['name'],
            resource_type=data['resource_type'],
            available_capacity=data.get('available_capacity', 0),
            allocated_capacity=data.get('allocated_capacity', 0),
            efficiency=data.get('efficiency', 100),
            cost_per_unit=data.get('cost_per_unit', 0),
            hourly_rate=data.get('hourly_rate'),
            fixed_cost=data.get('fixed_cost'),
            availability_hours=data.get('availability_hours'),
            status=data.get('status', 'Available')
        )
        resource.calculate_utilization()
        db.session.add(resource)
        db.session.commit()
        return resource

    def get_resources(
        self,
        org_id: str,
        department_id: Optional[str] = None,
        resource_type: Optional[str] = None
    ) -> List[Resource]:
        """Get resources with optional filters."""
        query = Resource.query.filter_by(organization_id=org_id)
        if department_id:
            query = query.filter_by(department_id=department_id)
        if resource_type:
            query = query.filter_by(resource_type=resource_type)
        return query.order_by(Resource.name).all()

    def get_resource(self, resource_id: str) -> Optional[Resource]:
        """Get resource by ID."""
        return Resource.query.get(resource_id)

    def update_resource(self, resource_id: str, data: Dict[str, Any]) -> Optional[Resource]:
        """Update resource."""
        resource = Resource.query.get(resource_id)
        if not resource:
            return None

        for key, value in data.items():
            if hasattr(resource, key) and key not in ['id', 'organization_id']:
                setattr(resource, key, value)

        resource.calculate_utilization()
        db.session.commit()
        return resource

    # ==================== KPI Entries ====================

    def create_kpi_entry(self, org_id: str, data: Dict[str, Any]) -> KPIEntry:
        """Create a KPI entry."""
        entry = KPIEntry(
            organization_id=org_id,
            department_id=data.get('department_id'),
            process_id=data.get('process_id'),
            period_date=data['period_date'],
            period_label=data.get('period_label'),
            period_type=data.get('period_type', 'Monthly'),
            oee=data.get('oee'),
            labor_productivity=data.get('labor_productivity'),
            capacity_utilization=data.get('capacity_utilization'),
            throughput_rate=data.get('throughput_rate'),
            first_pass_yield=data.get('first_pass_yield'),
            defect_rate=data.get('defect_rate'),
            rework_rate=data.get('rework_rate'),
            scrap_rate=data.get('scrap_rate'),
            on_time_delivery=data.get('on_time_delivery'),
            order_fulfillment=data.get('order_fulfillment'),
            lead_time=data.get('lead_time'),
            cost_per_unit=data.get('cost_per_unit'),
            waste_rate=data.get('waste_rate'),
            overtime_percentage=data.get('overtime_percentage'),
            incident_rate=data.get('incident_rate'),
            near_miss_count=data.get('near_miss_count'),
            safety_compliance=data.get('safety_compliance'),
            response_time=data.get('response_time'),
            resolution_rate=data.get('resolution_rate'),
            customer_satisfaction=data.get('customer_satisfaction'),
            sla_compliance=data.get('sla_compliance')
        )
        db.session.add(entry)
        db.session.commit()
        return entry

    def get_kpi_entries(
        self,
        org_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        department_id: Optional[str] = None,
        process_id: Optional[str] = None,
        limit: int = 100
    ) -> List[KPIEntry]:
        """Get KPI entries with optional filters."""
        query = KPIEntry.query.filter_by(organization_id=org_id)

        if start_date:
            query = query.filter(KPIEntry.period_date >= start_date)
        if end_date:
            query = query.filter(KPIEntry.period_date <= end_date)
        if department_id:
            query = query.filter_by(department_id=department_id)
        if process_id:
            query = query.filter_by(process_id=process_id)

        return query.order_by(desc(KPIEntry.period_date)).limit(limit).all()

    def get_latest_kpi_entry(self, org_id: str) -> Optional[KPIEntry]:
        """Get the most recent KPI entry."""
        return KPIEntry.query.filter_by(organization_id=org_id).order_by(
            desc(KPIEntry.period_date)
        ).first()

    # ==================== Capacity Plans ====================

    def save_capacity_plan(self, org_id: str, plan_data: Dict[str, Any]) -> CapacityPlan:
        """Save a capacity plan."""
        import json

        plan = CapacityPlan(
            organization_id=org_id,
            name=plan_data.get('plan_name', 'Capacity Plan'),
            description=plan_data.get('description'),
            start_date=plan_data.get('start_date'),
            end_date=plan_data.get('end_date'),
            planning_horizon=plan_data.get('planning_horizon'),
            total_demand=plan_data.get('total_demand'),
            total_capacity=plan_data.get('total_capacity'),
            average_utilization=plan_data.get('average_utilization'),
            peak_utilization=plan_data.get('peak_utilization'),
            bottleneck_count=len(plan_data.get('bottleneck_periods', [])),
            risk_level=plan_data.get('risk_level'),
            risk_assessment=plan_data.get('risk_assessment'),
            plan_data=json.dumps(plan_data)
        )
        plan.set_recommendations(plan_data.get('investment_recommendations', []))
        plan.set_cost_summary(plan_data.get('cost_summary', {}))

        db.session.add(plan)
        db.session.commit()
        return plan

    def get_capacity_plans(self, org_id: str, limit: int = 10) -> List[CapacityPlan]:
        """Get capacity plans for an organization."""
        return CapacityPlan.query.filter_by(organization_id=org_id).order_by(
            desc(CapacityPlan.created_at)
        ).limit(limit).all()

    def get_capacity_plan(self, plan_id: str) -> Optional[CapacityPlan]:
        """Get capacity plan by ID."""
        return CapacityPlan.query.get(plan_id)

    # ==================== Benchmark Results ====================

    def save_benchmark_result(self, org_id: str, result_data: Dict[str, Any]) -> BenchmarkResult:
        """Save a benchmark result."""
        result = BenchmarkResult(
            organization_id=org_id,
            benchmark_type=result_data.get('benchmark_type', 'operations'),
            overall_score=result_data.get('overall_score'),
            overall_rating=result_data.get('overall_rating'),
            grade=result_data.get('grade'),
            percentile=result_data.get('percentile')
        )
        result.set_category_scores(result_data.get('category_scores', {}))
        result.set_strengths(result_data.get('top_strengths', []))
        result.set_improvements(result_data.get('top_improvements', []))
        result.set_recommendations(result_data.get('recommendations', []))

        db.session.add(result)
        db.session.commit()
        return result

    def get_benchmark_results(self, org_id: str, limit: int = 10) -> List[BenchmarkResult]:
        """Get benchmark results for an organization."""
        return BenchmarkResult.query.filter_by(organization_id=org_id).order_by(
            desc(BenchmarkResult.analysis_date)
        ).limit(limit).all()

    def get_latest_benchmark(self, org_id: str) -> Optional[BenchmarkResult]:
        """Get the most recent benchmark result."""
        return BenchmarkResult.query.filter_by(organization_id=org_id).order_by(
            desc(BenchmarkResult.analysis_date)
        ).first()

    # ==================== Chat Sessions ====================

    def create_chat_session(self, org_id: Optional[str] = None, mode: str = 'general') -> ChatSession:
        """Create a new chat session."""
        session = ChatSession(
            organization_id=org_id,
            conversation_mode=mode
        )
        db.session.add(session)
        db.session.commit()
        return session

    def get_chat_session(self, session_id: str) -> Optional[ChatSession]:
        """Get chat session by ID."""
        return ChatSession.query.get(session_id)

    def add_chat_message(
        self,
        session_id: str,
        role: str,
        content: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ChatMessage:
        """Add a message to a chat session."""
        message = ChatMessage(
            session_id=session_id,
            role=role,
            content=content
        )
        if context:
            message.set_context(context)

        db.session.add(message)
        db.session.commit()
        return message

    def get_chat_messages(self, session_id: str, limit: int = 50) -> List[ChatMessage]:
        """Get messages for a chat session."""
        return ChatMessage.query.filter_by(session_id=session_id).order_by(
            ChatMessage.created_at
        ).limit(limit).all()

    def update_session_mode(self, session_id: str, mode: str) -> Optional[ChatSession]:
        """Update conversation mode for a session."""
        session = ChatSession.query.get(session_id)
        if session:
            session.conversation_mode = mode
            db.session.commit()
        return session

    # ==================== Analytics ====================

    def get_organization_summary(self, org_id: str) -> Dict[str, Any]:
        """Get operational summary for an organization."""
        org = self.get_organization(org_id)
        if not org:
            return {}

        latest_kpi = self.get_latest_kpi_entry(org_id)
        latest_benchmark = self.get_latest_benchmark(org_id)

        return {
            'organization': org.to_dict(),
            'department_count': org.departments.count(),
            'process_count': org.processes.count(),
            'resource_count': org.resources.count(),
            'latest_kpi': latest_kpi.to_dict() if latest_kpi else None,
            'latest_benchmark': latest_benchmark.to_dict() if latest_benchmark else None,
            'overall_oee': org.overall_oee,
            'efficiency_score': org.efficiency_score,
            'risk_level': org.risk_level
        }

    def calculate_organization_metrics(self, org_id: str) -> Dict[str, Any]:
        """Calculate aggregate metrics for an organization."""
        processes = self.get_processes(org_id)
        resources = self.get_resources(org_id)
        kpi_entries = self.get_kpi_entries(org_id, limit=12)

        # Average OEE across processes
        oee_values = [p.oee for p in processes if p.oee]
        avg_oee = sum(oee_values) / len(oee_values) if oee_values else None

        # Average resource utilization
        util_values = [r.utilization for r in resources if r.utilization]
        avg_utilization = sum(util_values) / len(util_values) if util_values else None

        # Trend analysis from KPI entries
        if len(kpi_entries) >= 2:
            recent_oee = kpi_entries[0].oee if kpi_entries[0].oee else 0
            older_oee = kpi_entries[-1].oee if kpi_entries[-1].oee else 0
            oee_trend = recent_oee - older_oee
        else:
            oee_trend = 0

        # Update organization with calculated metrics
        org = self.get_organization(org_id)
        if org:
            org.overall_oee = avg_oee
            db.session.commit()

        return {
            'average_oee': round(avg_oee, 1) if avg_oee else None,
            'average_utilization': round(avg_utilization, 1) if avg_utilization else None,
            'oee_trend': round(oee_trend, 1),
            'process_count': len(processes),
            'resource_count': len(resources),
            'kpi_entry_count': len(kpi_entries)
        }
