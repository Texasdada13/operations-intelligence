"""
Capacity Planner - Operations Intelligence

Capacity planning, resource allocation, and demand forecasting
for operational optimization.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from datetime import datetime, timedelta
import logging
import math

logger = logging.getLogger(__name__)


class CapacityStatus(Enum):
    EXCESS = "Excess Capacity"
    OPTIMAL = "Optimal"
    TIGHT = "Tight"
    CONSTRAINED = "Constrained"
    OVERLOADED = "Overloaded"


class ResourceType(Enum):
    LABOR = "Labor"
    EQUIPMENT = "Equipment"
    SPACE = "Space"
    MATERIAL = "Material"


class AllocationPriority(Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


@dataclass
class Resource:
    """A resource that can be allocated."""
    resource_id: str
    name: str
    resource_type: ResourceType
    available_capacity: float  # units per period
    cost_per_unit: float = 0
    utilization: float = 0
    efficiency: float = 100  # percentage
    constraints: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DemandForecast:
    """Demand forecast for a period."""
    period: str
    demand: float
    confidence: float = 0.8
    seasonality_factor: float = 1.0
    trend_factor: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CapacityForecast:
    """Capacity forecast and analysis."""
    period: str
    available_capacity: float
    required_capacity: float
    utilization: float
    status: CapacityStatus
    gap: float
    gap_percentage: float
    resource_breakdown: Dict[str, float]
    recommendations: List[str]
    risk_level: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "period": self.period,
            "available_capacity": round(self.available_capacity, 1),
            "required_capacity": round(self.required_capacity, 1),
            "utilization": round(self.utilization, 1),
            "status": self.status.value,
            "gap": round(self.gap, 1),
            "gap_percentage": round(self.gap_percentage, 1),
            "risk_level": self.risk_level,
            "recommendations": self.recommendations
        }


@dataclass
class ResourceAllocation:
    """Resource allocation result."""
    resource_id: str
    resource_name: str
    resource_type: str
    allocated_capacity: float
    total_capacity: float
    utilization: float
    allocation_efficiency: float
    cost: float
    priority: AllocationPriority
    tasks_assigned: List[str]
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resource_id": self.resource_id,
            "resource_name": self.resource_name,
            "resource_type": self.resource_type,
            "allocated_capacity": round(self.allocated_capacity, 1),
            "total_capacity": round(self.total_capacity, 1),
            "utilization": round(self.utilization, 1),
            "efficiency": round(self.allocation_efficiency, 1),
            "cost": round(self.cost, 2),
            "priority": self.priority.value,
            "tasks_assigned": self.tasks_assigned
        }


@dataclass
class CapacityPlan:
    """Complete capacity plan."""
    plan_id: str
    plan_name: str
    periods: List[CapacityForecast]
    total_demand: float
    total_capacity: float
    average_utilization: float
    peak_utilization: float
    resource_allocations: List[ResourceAllocation]
    bottleneck_periods: List[str]
    investment_recommendations: List[str]
    cost_summary: Dict[str, float]
    risk_assessment: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "plan_name": self.plan_name,
            "total_demand": round(self.total_demand, 1),
            "total_capacity": round(self.total_capacity, 1),
            "average_utilization": round(self.average_utilization, 1),
            "peak_utilization": round(self.peak_utilization, 1),
            "bottleneck_periods": self.bottleneck_periods,
            "investment_recommendations": self.investment_recommendations,
            "cost_summary": {k: round(v, 2) for k, v in self.cost_summary.items()},
            "risk_assessment": self.risk_assessment,
            "periods": [p.to_dict() for p in self.periods]
        }


class CapacityPlanner:
    """Plans capacity and allocates resources for operations."""

    UTILIZATION_THRESHOLDS = {
        95: CapacityStatus.OVERLOADED,
        85: CapacityStatus.CONSTRAINED,
        70: CapacityStatus.TIGHT,
        50: CapacityStatus.OPTIMAL,
        0: CapacityStatus.EXCESS
    }

    def __init__(
        self,
        target_utilization: float = 80,
        planning_buffer: float = 10  # percentage buffer for uncertainty
    ):
        self.target_utilization = target_utilization
        self.planning_buffer = planning_buffer

    def create_capacity_plan(
        self,
        plan_id: str,
        plan_name: str,
        resources: List[Resource],
        demand_forecasts: List[DemandForecast],
        tasks: Optional[List[Dict[str, Any]]] = None
    ) -> CapacityPlan:
        """Create a comprehensive capacity plan."""
        if not resources or not demand_forecasts:
            return self._empty_plan(plan_id, plan_name)

        # Calculate total available capacity
        total_capacity = sum(
            r.available_capacity * (r.efficiency / 100)
            for r in resources
        )

        # Analyze each period
        period_forecasts = []
        bottleneck_periods = []

        for demand in demand_forecasts:
            required = demand.demand * (1 + self.planning_buffer / 100)
            utilization = (required / total_capacity * 100) if total_capacity > 0 else 100
            gap = total_capacity - required
            gap_pct = (gap / total_capacity * 100) if total_capacity > 0 else -100

            status = self._determine_capacity_status(utilization)

            if status in [CapacityStatus.CONSTRAINED, CapacityStatus.OVERLOADED]:
                bottleneck_periods.append(demand.period)

            resource_breakdown = {
                r.name: r.available_capacity * (r.efficiency / 100)
                for r in resources
            }

            recommendations = self._generate_period_recommendations(
                utilization, gap, status, demand
            )

            risk_level = self._assess_period_risk(utilization, demand.confidence)

            period_forecasts.append(CapacityForecast(
                period=demand.period,
                available_capacity=total_capacity,
                required_capacity=required,
                utilization=utilization,
                status=status,
                gap=gap,
                gap_percentage=gap_pct,
                resource_breakdown=resource_breakdown,
                recommendations=recommendations,
                risk_level=risk_level
            ))

        # Calculate resource allocations
        allocations = self.allocate_resources(resources, tasks or [])

        # Calculate summary metrics
        total_demand = sum(d.demand for d in demand_forecasts)
        avg_utilization = sum(p.utilization for p in period_forecasts) / len(period_forecasts)
        peak_utilization = max(p.utilization for p in period_forecasts)

        # Cost summary
        cost_summary = self._calculate_cost_summary(resources, allocations)

        # Investment recommendations
        investment_recommendations = self._generate_investment_recommendations(
            period_forecasts, resources, peak_utilization
        )

        # Overall risk assessment
        risk_assessment = self._assess_overall_risk(
            avg_utilization, peak_utilization, len(bottleneck_periods), len(period_forecasts)
        )

        return CapacityPlan(
            plan_id=plan_id,
            plan_name=plan_name,
            periods=period_forecasts,
            total_demand=total_demand,
            total_capacity=total_capacity * len(period_forecasts),
            average_utilization=avg_utilization,
            peak_utilization=peak_utilization,
            resource_allocations=allocations,
            bottleneck_periods=bottleneck_periods,
            investment_recommendations=investment_recommendations,
            cost_summary=cost_summary,
            risk_assessment=risk_assessment
        )

    def allocate_resources(
        self,
        resources: List[Resource],
        tasks: List[Dict[str, Any]]
    ) -> List[ResourceAllocation]:
        """Allocate resources to tasks based on priority and capacity."""
        allocations = []

        # Sort tasks by priority
        sorted_tasks = sorted(
            tasks,
            key=lambda t: {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}.get(
                t.get("priority", "Medium"), 2
            )
        )

        # Track remaining capacity per resource
        remaining_capacity = {r.resource_id: r.available_capacity for r in resources}

        for resource in resources:
            allocated = 0
            assigned_tasks = []

            # Find matching tasks for this resource type
            matching_tasks = [
                t for t in sorted_tasks
                if t.get("resource_type") == resource.resource_type.value or
                t.get("resource_id") == resource.resource_id
            ]

            for task in matching_tasks:
                required = task.get("capacity_required", 0)
                available = remaining_capacity.get(resource.resource_id, 0)

                if available >= required:
                    allocated += required
                    remaining_capacity[resource.resource_id] -= required
                    assigned_tasks.append(task.get("name", task.get("task_id", "Unknown")))

            utilization = (allocated / resource.available_capacity * 100) if resource.available_capacity > 0 else 0
            efficiency = resource.efficiency * (utilization / 100) if utilization > 0 else 0
            cost = allocated * resource.cost_per_unit

            priority = self._determine_allocation_priority(utilization)
            recommendations = self._generate_allocation_recommendations(
                resource, utilization, allocated
            )

            allocations.append(ResourceAllocation(
                resource_id=resource.resource_id,
                resource_name=resource.name,
                resource_type=resource.resource_type.value,
                allocated_capacity=allocated,
                total_capacity=resource.available_capacity,
                utilization=utilization,
                allocation_efficiency=efficiency,
                cost=cost,
                priority=priority,
                tasks_assigned=assigned_tasks,
                recommendations=recommendations
            ))

        return allocations

    def forecast_demand(
        self,
        historical_demand: List[float],
        periods_ahead: int = 6,
        growth_rate: float = 0,
        seasonality: Optional[List[float]] = None
    ) -> List[DemandForecast]:
        """Forecast demand for future periods."""
        if not historical_demand:
            return []

        # Calculate base demand (average of recent periods)
        recent_periods = min(6, len(historical_demand))
        base_demand = sum(historical_demand[-recent_periods:]) / recent_periods

        # Calculate trend from historical data
        if len(historical_demand) >= 3:
            recent_avg = sum(historical_demand[-3:]) / 3
            older_avg = sum(historical_demand[:3]) / 3 if len(historical_demand) >= 6 else recent_avg
            calculated_growth = ((recent_avg / older_avg) - 1) if older_avg > 0 else 0
            growth_rate = growth_rate or calculated_growth

        # Default seasonality if not provided
        if seasonality is None:
            seasonality = [1.0] * 12  # No seasonality

        forecasts = []
        for i in range(periods_ahead):
            period_num = i + 1
            period_name = f"Period {period_num}"

            # Apply growth trend
            trend_factor = 1 + (growth_rate * period_num)

            # Apply seasonality (cycle through pattern)
            season_idx = i % len(seasonality)
            seasonality_factor = seasonality[season_idx]

            # Calculate forecast with decreasing confidence
            forecast_demand = base_demand * trend_factor * seasonality_factor
            confidence = max(0.5, 0.95 - (0.05 * i))  # Confidence decreases over time

            forecasts.append(DemandForecast(
                period=period_name,
                demand=round(forecast_demand, 1),
                confidence=confidence,
                seasonality_factor=seasonality_factor,
                trend_factor=trend_factor
            ))

        return forecasts

    def calculate_resource_requirements(
        self,
        demand: float,
        productivity_rate: float,
        availability_rate: float = 0.9,
        buffer: float = 0.1
    ) -> Dict[str, Any]:
        """Calculate resource requirements to meet demand."""
        if productivity_rate <= 0:
            return {"error": "Productivity rate must be positive"}

        # Base requirement
        base_requirement = demand / productivity_rate

        # Adjust for availability
        availability_adjusted = base_requirement / availability_rate

        # Add buffer for uncertainty
        total_requirement = availability_adjusted * (1 + buffer)

        return {
            "demand": round(demand, 1),
            "base_requirement": round(base_requirement, 1),
            "availability_adjusted": round(availability_adjusted, 1),
            "total_requirement": round(total_requirement, 1),
            "buffer_units": round(total_requirement - availability_adjusted, 1),
            "productivity_rate": productivity_rate,
            "availability_rate": availability_rate,
            "buffer_percentage": buffer * 100
        }

    def simulate_capacity_scenario(
        self,
        resources: List[Resource],
        demand_forecasts: List[DemandForecast],
        scenario: Dict[str, float]
    ) -> Tuple[CapacityPlan, Dict[str, Any]]:
        """
        Simulate capacity scenarios with modifications.

        scenario: Dict of modifications
            - "demand_change": percentage change in demand
            - "capacity_change": percentage change in capacity
            - "efficiency_change": percentage change in efficiency
        """
        demand_change = scenario.get("demand_change", 0) / 100
        capacity_change = scenario.get("capacity_change", 0) / 100
        efficiency_change = scenario.get("efficiency_change", 0) / 100

        # Modify demand forecasts
        modified_forecasts = [
            DemandForecast(
                period=d.period,
                demand=d.demand * (1 + demand_change),
                confidence=d.confidence,
                seasonality_factor=d.seasonality_factor,
                trend_factor=d.trend_factor
            )
            for d in demand_forecasts
        ]

        # Modify resources
        modified_resources = [
            Resource(
                resource_id=r.resource_id,
                name=r.name,
                resource_type=r.resource_type,
                available_capacity=r.available_capacity * (1 + capacity_change),
                cost_per_unit=r.cost_per_unit,
                utilization=r.utilization,
                efficiency=min(100, r.efficiency * (1 + efficiency_change)),
                constraints=r.constraints
            )
            for r in resources
        ]

        # Create scenario plan
        scenario_plan = self.create_capacity_plan(
            "scenario",
            "Scenario Analysis",
            modified_resources,
            modified_forecasts
        )

        # Create baseline for comparison
        baseline_plan = self.create_capacity_plan(
            "baseline",
            "Baseline",
            resources,
            demand_forecasts
        )

        # Calculate comparison metrics
        comparison = {
            "demand_change_pct": demand_change * 100,
            "capacity_change_pct": capacity_change * 100,
            "efficiency_change_pct": efficiency_change * 100,
            "utilization_change": scenario_plan.average_utilization - baseline_plan.average_utilization,
            "bottleneck_periods_change": len(scenario_plan.bottleneck_periods) - len(baseline_plan.bottleneck_periods),
            "cost_change": sum(scenario_plan.cost_summary.values()) - sum(baseline_plan.cost_summary.values()),
            "baseline_avg_utilization": baseline_plan.average_utilization,
            "scenario_avg_utilization": scenario_plan.average_utilization
        }

        return scenario_plan, comparison

    def _empty_plan(self, plan_id: str, plan_name: str) -> CapacityPlan:
        """Return empty plan when no data available."""
        return CapacityPlan(
            plan_id=plan_id,
            plan_name=plan_name,
            periods=[],
            total_demand=0,
            total_capacity=0,
            average_utilization=0,
            peak_utilization=0,
            resource_allocations=[],
            bottleneck_periods=[],
            investment_recommendations=["Add resources and demand forecasts to begin planning"],
            cost_summary={},
            risk_assessment="Unknown - insufficient data"
        )

    def _determine_capacity_status(self, utilization: float) -> CapacityStatus:
        """Determine capacity status based on utilization."""
        for threshold, status in sorted(self.UTILIZATION_THRESHOLDS.items(), reverse=True):
            if utilization >= threshold:
                return status
        return CapacityStatus.EXCESS

    def _determine_allocation_priority(self, utilization: float) -> AllocationPriority:
        """Determine allocation priority based on utilization."""
        if utilization >= 90:
            return AllocationPriority.CRITICAL
        elif utilization >= 75:
            return AllocationPriority.HIGH
        elif utilization >= 50:
            return AllocationPriority.MEDIUM
        return AllocationPriority.LOW

    def _assess_period_risk(self, utilization: float, confidence: float) -> str:
        """Assess risk level for a period."""
        uncertainty = 1 - confidence
        risk_score = utilization * (1 + uncertainty * 0.5)

        if risk_score >= 100:
            return "Critical"
        elif risk_score >= 85:
            return "High"
        elif risk_score >= 70:
            return "Medium"
        return "Low"

    def _assess_overall_risk(
        self,
        avg_utilization: float,
        peak_utilization: float,
        bottleneck_count: int,
        total_periods: int
    ) -> str:
        """Assess overall capacity plan risk."""
        bottleneck_ratio = bottleneck_count / total_periods if total_periods > 0 else 0

        if peak_utilization > 100 or bottleneck_ratio > 0.5:
            return "Critical - Capacity shortfall expected"
        elif peak_utilization > 90 or bottleneck_ratio > 0.3:
            return "High - Capacity constraints likely"
        elif avg_utilization > 80 or bottleneck_ratio > 0.15:
            return "Medium - Monitor closely"
        elif avg_utilization > 60:
            return "Low - Adequate capacity"
        return "Very Low - Excess capacity available"

    def _generate_period_recommendations(
        self,
        utilization: float,
        gap: float,
        status: CapacityStatus,
        demand: DemandForecast
    ) -> List[str]:
        """Generate recommendations for a specific period."""
        recommendations = []

        if status == CapacityStatus.OVERLOADED:
            recommendations.append(f"URGENT: Add {abs(gap):.0f} units of capacity")
            recommendations.append("Consider overtime or temporary resources")

        elif status == CapacityStatus.CONSTRAINED:
            recommendations.append("Prioritize critical workloads")
            recommendations.append("Evaluate capacity expansion options")

        elif status == CapacityStatus.TIGHT:
            recommendations.append("Monitor utilization closely")
            if demand.confidence < 0.8:
                recommendations.append("Prepare contingency capacity")

        elif status == CapacityStatus.EXCESS:
            recommendations.append("Consider reassigning resources")
            recommendations.append("Evaluate cost optimization opportunities")

        return recommendations[:3]

    def _generate_allocation_recommendations(
        self,
        resource: Resource,
        utilization: float,
        allocated: float
    ) -> List[str]:
        """Generate recommendations for resource allocation."""
        recommendations = []

        if utilization > 90:
            recommendations.append("Resource at capacity - consider adding backup")

        elif utilization < 50:
            recommendations.append("Underutilized - consider reassignment")

        if resource.efficiency < 80:
            recommendations.append("Improve resource efficiency through training or maintenance")

        return recommendations

    def _generate_investment_recommendations(
        self,
        periods: List[CapacityForecast],
        resources: List[Resource],
        peak_utilization: float
    ) -> List[str]:
        """Generate investment recommendations."""
        recommendations = []

        if peak_utilization > 95:
            gap = max(p.required_capacity - p.available_capacity for p in periods)
            recommendations.append(f"Invest in {gap:.0f} units of additional capacity")

        # Identify constrained resource types
        constrained_periods = [p for p in periods if p.status in [CapacityStatus.CONSTRAINED, CapacityStatus.OVERLOADED]]
        if len(constrained_periods) > len(periods) * 0.3:
            recommendations.append("Consider long-term capacity expansion")

        # Check for efficiency improvements
        low_efficiency_resources = [r for r in resources if r.efficiency < 80]
        if low_efficiency_resources:
            recommendations.append(
                f"Improve efficiency for: {', '.join(r.name for r in low_efficiency_resources[:3])}"
            )

        if not recommendations:
            recommendations.append("Current capacity adequate - focus on optimization")

        return recommendations[:5]

    def _calculate_cost_summary(
        self,
        resources: List[Resource],
        allocations: List[ResourceAllocation]
    ) -> Dict[str, float]:
        """Calculate cost summary for capacity plan."""
        total_resource_cost = sum(
            r.available_capacity * r.cost_per_unit for r in resources
        )
        allocated_cost = sum(a.cost for a in allocations)
        idle_cost = total_resource_cost - allocated_cost

        return {
            "total_resource_cost": total_resource_cost,
            "allocated_cost": allocated_cost,
            "idle_cost": max(0, idle_cost),
            "utilization_efficiency": (allocated_cost / total_resource_cost * 100) if total_resource_cost > 0 else 0
        }
