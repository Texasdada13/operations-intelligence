"""
Process Analyzer - Operations Intelligence

Analyzes operational processes for efficiency, bottlenecks,
and improvement opportunities.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class ProcessStatus(Enum):
    OPTIMAL = "Optimal"
    HEALTHY = "Healthy"
    ATTENTION_NEEDED = "Attention Needed"
    AT_RISK = "At Risk"
    CRITICAL = "Critical"


class BottleneckSeverity(Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


@dataclass
class ProcessStep:
    """Individual step in a process."""
    step_id: str
    name: str
    cycle_time: float  # minutes
    wait_time: float = 0  # minutes
    defect_rate: float = 0  # percentage
    capacity: float = 100  # units per hour
    utilization: float = 0  # percentage
    resources_required: int = 1
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessMetrics:
    """Calculated metrics for a process."""
    process_id: str
    process_name: str
    total_cycle_time: float
    total_wait_time: float
    total_lead_time: float
    process_efficiency: float  # cycle_time / lead_time
    first_pass_yield: float
    throughput: float
    bottleneck_step: Optional[str]
    status: ProcessStatus
    oee: float  # Overall Equipment Effectiveness
    takt_time: Optional[float]
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "process_id": self.process_id,
            "process_name": self.process_name,
            "total_cycle_time": self.total_cycle_time,
            "total_wait_time": self.total_wait_time,
            "total_lead_time": self.total_lead_time,
            "process_efficiency": round(self.process_efficiency, 1),
            "first_pass_yield": round(self.first_pass_yield, 1),
            "throughput": round(self.throughput, 2),
            "bottleneck_step": self.bottleneck_step,
            "status": self.status.value,
            "oee": round(self.oee, 1),
            "recommendations": self.recommendations
        }


@dataclass
class BottleneckAnalysis:
    """Analysis of process bottlenecks."""
    step_id: str
    step_name: str
    severity: BottleneckSeverity
    impact_score: float  # 0-100
    constraint_type: str  # capacity, resource, quality, time
    current_capacity: float
    required_capacity: float
    capacity_gap: float
    utilization: float
    downstream_impact: float  # percentage of process affected
    root_causes: List[str]
    recommendations: List[str]
    estimated_improvement: float  # percentage improvement if resolved
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "step_name": self.step_name,
            "severity": self.severity.value,
            "impact_score": round(self.impact_score, 1),
            "constraint_type": self.constraint_type,
            "capacity_gap": round(self.capacity_gap, 1),
            "utilization": round(self.utilization, 1),
            "root_causes": self.root_causes,
            "recommendations": self.recommendations,
            "estimated_improvement": round(self.estimated_improvement, 1)
        }


class ProcessAnalyzer:
    """Analyzes operational processes for efficiency and bottlenecks."""

    STATUS_THRESHOLDS = {
        90: ProcessStatus.OPTIMAL,
        75: ProcessStatus.HEALTHY,
        60: ProcessStatus.ATTENTION_NEEDED,
        40: ProcessStatus.AT_RISK,
        0: ProcessStatus.CRITICAL
    }

    BOTTLENECK_THRESHOLDS = {
        95: BottleneckSeverity.CRITICAL,
        85: BottleneckSeverity.HIGH,
        70: BottleneckSeverity.MEDIUM,
        0: BottleneckSeverity.LOW
    }

    def __init__(
        self,
        target_throughput: Optional[float] = None,
        available_time_per_day: float = 480  # 8 hours in minutes
    ):
        self.target_throughput = target_throughput
        self.available_time = available_time_per_day
        self.takt_time = None
        if target_throughput and target_throughput > 0:
            self.takt_time = available_time_per_day / target_throughput

    def analyze_process(
        self,
        process_id: str,
        process_name: str,
        steps: List[ProcessStep]
    ) -> ProcessMetrics:
        """Analyze a complete process and return metrics."""
        if not steps:
            return self._empty_metrics(process_id, process_name)

        # Calculate timing metrics
        total_cycle_time = sum(s.cycle_time for s in steps)
        total_wait_time = sum(s.wait_time for s in steps)
        total_lead_time = total_cycle_time + total_wait_time

        # Process efficiency (value-added time / total time)
        process_efficiency = (total_cycle_time / total_lead_time * 100) if total_lead_time > 0 else 0

        # First pass yield (cumulative)
        first_pass_yield = 100
        for step in steps:
            first_pass_yield *= (100 - step.defect_rate) / 100
        first_pass_yield *= 100

        # Find bottleneck (lowest capacity step)
        bottleneck_step = min(steps, key=lambda s: s.capacity)
        throughput = bottleneck_step.capacity

        # Calculate OEE components
        avg_availability = 100 - (total_wait_time / total_lead_time * 100) if total_lead_time > 0 else 100
        avg_performance = sum(s.utilization for s in steps) / len(steps) if steps else 0
        quality_rate = first_pass_yield

        oee = (avg_availability / 100) * (avg_performance / 100) * (quality_rate / 100) * 100

        # Determine status
        status = self._determine_status(oee)

        # Generate recommendations
        recommendations = self._generate_process_recommendations(
            steps, process_efficiency, first_pass_yield, oee, bottleneck_step
        )

        return ProcessMetrics(
            process_id=process_id,
            process_name=process_name,
            total_cycle_time=round(total_cycle_time, 1),
            total_wait_time=round(total_wait_time, 1),
            total_lead_time=round(total_lead_time, 1),
            process_efficiency=process_efficiency,
            first_pass_yield=first_pass_yield,
            throughput=throughput,
            bottleneck_step=bottleneck_step.name,
            status=status,
            oee=oee,
            takt_time=self.takt_time,
            recommendations=recommendations
        )

    def identify_bottlenecks(
        self,
        steps: List[ProcessStep],
        demand_rate: Optional[float] = None
    ) -> List[BottleneckAnalysis]:
        """Identify and analyze bottlenecks in process steps."""
        if not steps:
            return []

        demand = demand_rate or self.target_throughput or max(s.capacity for s in steps)
        bottlenecks = []

        for step in steps:
            utilization = (demand / step.capacity * 100) if step.capacity > 0 else 100

            if utilization >= 70:  # Step is constrained
                severity = self._determine_bottleneck_severity(utilization)
                capacity_gap = demand - step.capacity if demand > step.capacity else 0

                # Determine constraint type
                constraint_type = self._identify_constraint_type(step, utilization)

                # Calculate downstream impact
                step_idx = steps.index(step)
                downstream_steps = len(steps) - step_idx - 1
                downstream_impact = (downstream_steps / len(steps) * 100) if len(steps) > 1 else 0

                # Impact score based on utilization and downstream effect
                impact_score = (utilization * 0.6) + (downstream_impact * 0.4)

                # Root cause analysis
                root_causes = self._analyze_root_causes(step, constraint_type)

                # Recommendations
                recommendations = self._generate_bottleneck_recommendations(
                    step, constraint_type, capacity_gap, utilization
                )

                # Estimated improvement if resolved
                estimated_improvement = min(30, (utilization - 70) * 0.5) if utilization > 70 else 5

                bottlenecks.append(BottleneckAnalysis(
                    step_id=step.step_id,
                    step_name=step.name,
                    severity=severity,
                    impact_score=impact_score,
                    constraint_type=constraint_type,
                    current_capacity=step.capacity,
                    required_capacity=demand,
                    capacity_gap=capacity_gap,
                    utilization=utilization,
                    downstream_impact=downstream_impact,
                    root_causes=root_causes,
                    recommendations=recommendations,
                    estimated_improvement=estimated_improvement
                ))

        # Sort by impact score descending
        bottlenecks.sort(key=lambda b: b.impact_score, reverse=True)
        return bottlenecks

    def calculate_value_stream_metrics(
        self,
        steps: List[ProcessStep]
    ) -> Dict[str, Any]:
        """Calculate value stream mapping metrics."""
        if not steps:
            return {}

        value_added_time = sum(s.cycle_time for s in steps)
        non_value_added_time = sum(s.wait_time for s in steps)
        total_time = value_added_time + non_value_added_time

        # Calculate work-in-progress using Little's Law approximation
        avg_throughput = sum(s.capacity for s in steps) / len(steps)
        wip = (total_time / 60) * avg_throughput if avg_throughput > 0 else 0

        return {
            "value_added_time": round(value_added_time, 1),
            "non_value_added_time": round(non_value_added_time, 1),
            "total_lead_time": round(total_time, 1),
            "value_added_ratio": round(value_added_time / total_time * 100, 1) if total_time > 0 else 0,
            "estimated_wip": round(wip, 0),
            "step_count": len(steps),
            "avg_cycle_time": round(value_added_time / len(steps), 1),
            "avg_wait_time": round(non_value_added_time / len(steps), 1)
        }

    def simulate_improvement(
        self,
        steps: List[ProcessStep],
        improvement_scenario: Dict[str, Dict[str, float]]
    ) -> Tuple[ProcessMetrics, Dict[str, Any]]:
        """
        Simulate process improvement scenarios.

        improvement_scenario: {step_id: {metric: improvement_percentage}}
        Example: {"step_1": {"cycle_time": -10, "capacity": 20}}
        """
        # Create modified steps
        modified_steps = []
        for step in steps:
            improvements = improvement_scenario.get(step.step_id, {})

            new_cycle_time = step.cycle_time * (1 + improvements.get("cycle_time", 0) / 100)
            new_wait_time = step.wait_time * (1 + improvements.get("wait_time", 0) / 100)
            new_capacity = step.capacity * (1 + improvements.get("capacity", 0) / 100)
            new_defect_rate = max(0, step.defect_rate * (1 + improvements.get("defect_rate", 0) / 100))

            modified_steps.append(ProcessStep(
                step_id=step.step_id,
                name=step.name,
                cycle_time=max(0.1, new_cycle_time),
                wait_time=max(0, new_wait_time),
                capacity=new_capacity,
                defect_rate=new_defect_rate,
                utilization=step.utilization,
                resources_required=step.resources_required,
                dependencies=step.dependencies
            ))

        # Analyze improved process
        improved_metrics = self.analyze_process(
            "improved",
            "Improved Process",
            modified_steps
        )

        # Calculate baseline for comparison
        baseline_metrics = self.analyze_process(
            "baseline",
            "Baseline Process",
            steps
        )

        # Calculate improvements
        comparison = {
            "cycle_time_reduction": round(
                (baseline_metrics.total_cycle_time - improved_metrics.total_cycle_time) /
                baseline_metrics.total_cycle_time * 100, 1
            ) if baseline_metrics.total_cycle_time > 0 else 0,
            "lead_time_reduction": round(
                (baseline_metrics.total_lead_time - improved_metrics.total_lead_time) /
                baseline_metrics.total_lead_time * 100, 1
            ) if baseline_metrics.total_lead_time > 0 else 0,
            "efficiency_improvement": round(
                improved_metrics.process_efficiency - baseline_metrics.process_efficiency, 1
            ),
            "yield_improvement": round(
                improved_metrics.first_pass_yield - baseline_metrics.first_pass_yield, 1
            ),
            "throughput_improvement": round(
                (improved_metrics.throughput - baseline_metrics.throughput) /
                baseline_metrics.throughput * 100, 1
            ) if baseline_metrics.throughput > 0 else 0,
            "oee_improvement": round(
                improved_metrics.oee - baseline_metrics.oee, 1
            )
        }

        return improved_metrics, comparison

    def _empty_metrics(self, process_id: str, process_name: str) -> ProcessMetrics:
        """Return empty metrics for processes with no steps."""
        return ProcessMetrics(
            process_id=process_id,
            process_name=process_name,
            total_cycle_time=0,
            total_wait_time=0,
            total_lead_time=0,
            process_efficiency=0,
            first_pass_yield=0,
            throughput=0,
            bottleneck_step=None,
            status=ProcessStatus.CRITICAL,
            oee=0,
            takt_time=None,
            recommendations=["Add process steps to begin analysis"]
        )

    def _determine_status(self, oee: float) -> ProcessStatus:
        """Determine process status based on OEE."""
        for threshold, status in sorted(self.STATUS_THRESHOLDS.items(), reverse=True):
            if oee >= threshold:
                return status
        return ProcessStatus.CRITICAL

    def _determine_bottleneck_severity(self, utilization: float) -> BottleneckSeverity:
        """Determine bottleneck severity based on utilization."""
        for threshold, severity in sorted(self.BOTTLENECK_THRESHOLDS.items(), reverse=True):
            if utilization >= threshold:
                return severity
        return BottleneckSeverity.LOW

    def _identify_constraint_type(self, step: ProcessStep, utilization: float) -> str:
        """Identify the primary constraint type for a step."""
        if step.defect_rate > 5:
            return "quality"
        if step.resources_required > 1 and utilization > 90:
            return "resource"
        if step.wait_time > step.cycle_time:
            return "time"
        return "capacity"

    def _analyze_root_causes(self, step: ProcessStep, constraint_type: str) -> List[str]:
        """Analyze potential root causes for a bottleneck."""
        causes = []

        if constraint_type == "capacity":
            causes.append("Equipment capacity limit reached")
            if step.utilization < 85:
                causes.append("Potential setup time losses")

        elif constraint_type == "quality":
            causes.append(f"High defect rate ({step.defect_rate}%)")
            causes.append("Possible quality control gaps")

        elif constraint_type == "resource":
            causes.append(f"Resource constraint ({step.resources_required} resources required)")
            causes.append("Possible skill/training gaps")

        elif constraint_type == "time":
            causes.append(f"Excessive wait time ({step.wait_time} min)")
            causes.append("Upstream process variability")

        return causes

    def _generate_process_recommendations(
        self,
        steps: List[ProcessStep],
        efficiency: float,
        first_pass_yield: float,
        oee: float,
        bottleneck: ProcessStep
    ) -> List[str]:
        """Generate recommendations for process improvement."""
        recommendations = []

        if efficiency < 50:
            recommendations.append("Focus on reducing wait times between steps to improve flow")

        if first_pass_yield < 90:
            high_defect_steps = [s for s in steps if s.defect_rate > 3]
            if high_defect_steps:
                recommendations.append(
                    f"Address quality issues in: {', '.join(s.name for s in high_defect_steps[:3])}"
                )

        if oee < 60:
            recommendations.append("Implement OEE improvement program targeting availability and performance")

        if bottleneck.utilization > 90:
            recommendations.append(f"Increase capacity at bottleneck: {bottleneck.name}")

        if len(recommendations) == 0:
            recommendations.append("Process performing well - monitor for continuous improvement opportunities")

        return recommendations[:5]

    def _generate_bottleneck_recommendations(
        self,
        step: ProcessStep,
        constraint_type: str,
        capacity_gap: float,
        utilization: float
    ) -> List[str]:
        """Generate recommendations for bottleneck resolution."""
        recommendations = []

        if constraint_type == "capacity":
            recommendations.append(f"Increase {step.name} capacity by {capacity_gap:.0f} units/hour")
            if utilization > 95:
                recommendations.append("Consider adding parallel processing capability")

        elif constraint_type == "quality":
            recommendations.append("Implement statistical process control (SPC)")
            recommendations.append("Review and update quality inspection procedures")

        elif constraint_type == "resource":
            recommendations.append("Cross-train additional personnel")
            recommendations.append("Evaluate automation opportunities")

        elif constraint_type == "time":
            recommendations.append("Reduce batch sizes to improve flow")
            recommendations.append("Implement pull-based scheduling")

        return recommendations
