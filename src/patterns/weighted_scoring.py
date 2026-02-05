"""
Weighted Scoring Pattern - Operations Intelligence

Multi-component scoring engine for operational performance.
Calculates weighted aggregate scores from multiple metrics.

Use cases:
- Process efficiency scoring
- Resource utilization assessment
- Quality performance ratings
- Operational health scores
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ScoreDirection(Enum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


@dataclass
class ScoreComponent:
    name: str
    weight: float
    direction: ScoreDirection = ScoreDirection.HIGHER_IS_BETTER
    min_value: float = 0.0
    max_value: float = 100.0
    description: str = ""

    def normalize(self, value: float) -> float:
        if self.max_value == self.min_value:
            return 100.0 if value >= self.max_value else 0.0
        value = max(self.min_value, min(self.max_value, value))
        normalized = ((value - self.min_value) / (self.max_value - self.min_value)) * 100
        if self.direction == ScoreDirection.LOWER_IS_BETTER:
            normalized = 100 - normalized
        return round(normalized, 2)


@dataclass
class ScoreResult:
    entity_id: str
    overall_score: float
    grade: str
    component_scores: Dict[str, float]
    component_details: Dict[str, Dict[str, Any]]
    risk_level: str
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "overall_score": self.overall_score,
            "grade": self.grade,
            "component_scores": self.component_scores,
            "component_details": self.component_details,
            "risk_level": self.risk_level,
            "recommendations": self.recommendations,
            "metadata": self.metadata
        }


class WeightedScoringEngine:
    """Multi-component weighted scoring for operational metrics."""

    DEFAULT_GRADE_THRESHOLDS = {90: "A", 80: "B", 70: "C", 60: "D", 0: "F"}
    DEFAULT_RISK_THRESHOLDS = {80: "Low", 60: "Medium", 40: "High", 0: "Critical"}

    def __init__(
        self,
        components: List[ScoreComponent],
        grade_thresholds: Optional[Dict[int, str]] = None,
        risk_thresholds: Optional[Dict[int, str]] = None,
        recommendation_rules: Optional[List[Dict[str, Any]]] = None
    ):
        self.components = {c.name: c for c in components}
        self.grade_thresholds = grade_thresholds or self.DEFAULT_GRADE_THRESHOLDS
        self.risk_thresholds = risk_thresholds or self.DEFAULT_RISK_THRESHOLDS
        self.recommendation_rules = recommendation_rules or []

        total_weight = sum(c.weight for c in components)
        if abs(total_weight - 1.0) > 0.01:
            for c in components:
                c.weight = c.weight / total_weight

    def score(
        self,
        values: Dict[str, float],
        entity_id: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None
    ) -> ScoreResult:
        component_scores = {}
        component_details = {}
        weighted_sum = 0.0

        for name, component in self.components.items():
            raw_value = values.get(name, component.min_value)
            normalized = component.normalize(raw_value)
            component_scores[name] = normalized
            weighted_contribution = normalized * component.weight
            weighted_sum += weighted_contribution

            component_details[name] = {
                "raw_value": raw_value,
                "normalized_score": normalized,
                "weight": component.weight,
                "weighted_contribution": round(weighted_contribution, 2),
                "direction": component.direction.value,
                "description": component.description
            }

        overall_score = round(weighted_sum, 2)
        grade = self._determine_grade(overall_score)
        risk_level = self._determine_risk_level(overall_score)
        recommendations = self._generate_recommendations(component_scores, overall_score)

        return ScoreResult(
            entity_id=entity_id,
            overall_score=overall_score,
            grade=grade,
            component_scores=component_scores,
            component_details=component_details,
            risk_level=risk_level,
            recommendations=recommendations,
            metadata=metadata or {}
        )

    def score_batch(
        self,
        entities: List[Dict[str, Any]],
        id_field: str = "id"
    ) -> List[ScoreResult]:
        results = []
        for entity in entities:
            entity_id = str(entity.get(id_field, "unknown"))
            values = {f: entity.get(f) for f in self.components.keys()}
            metadata = {k: v for k, v in entity.items() if k not in self.components and k != id_field}
            try:
                result = self.score(values, entity_id=entity_id, metadata=metadata)
                results.append(result)
            except Exception as e:
                logger.error(f"Error scoring entity {entity_id}: {e}")
        return results

    def _determine_grade(self, score: float) -> str:
        for threshold, grade in sorted(self.grade_thresholds.items(), reverse=True):
            if score >= threshold:
                return grade
        return "F"

    def _determine_risk_level(self, score: float) -> str:
        for threshold, level in sorted(self.risk_thresholds.items(), reverse=True):
            if score >= threshold:
                return level
        return "Critical"

    def _generate_recommendations(self, component_scores: Dict[str, float], overall_score: float) -> List[str]:
        recommendations = []
        for name, score in component_scores.items():
            if score < 50:
                component = self.components[name]
                recommendations.append(f"Critical: Improve {name} (currently {score:.0f}/100) - {component.description}")
            elif score < 70:
                component = self.components[name]
                recommendations.append(f"Warning: Monitor {name} (currently {score:.0f}/100)")
        return recommendations


class AggregatedScoringEngine:
    """Aggregates scores across multiple entities."""

    def __init__(self, base_engine: WeightedScoringEngine, aggregation_method: str = "weighted_average"):
        self.base_engine = base_engine
        self.aggregation_method = aggregation_method

    def aggregate(
        self,
        entities: List[Dict[str, Any]],
        group_id: str,
        weight_field: Optional[str] = None,
        id_field: str = "id"
    ) -> ScoreResult:
        if not entities:
            return ScoreResult(
                entity_id=group_id, overall_score=0.0, grade="F",
                component_scores={}, component_details={},
                risk_level="Unknown", recommendations=["No data available"]
            )

        individual_scores = self.base_engine.score_batch(entities, id_field=id_field)

        if weight_field and self.aggregation_method == "weighted_average":
            weights = [e.get(weight_field, 1) for e in entities]
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
        else:
            weights = [1 / len(entities)] * len(entities)

        overall_score = sum(s.overall_score * w for s, w in zip(individual_scores, weights))
        overall_score = round(overall_score, 2)

        return ScoreResult(
            entity_id=group_id,
            overall_score=overall_score,
            grade=self.base_engine._determine_grade(overall_score),
            component_scores={},
            component_details={"entity_count": len(entities)},
            risk_level=self.base_engine._determine_risk_level(overall_score),
            metadata={"weight_field": weight_field}
        )


# Factory Functions for Operations Intelligence

def create_process_efficiency_engine() -> WeightedScoringEngine:
    """Create a process efficiency scoring engine."""
    components = [
        ScoreComponent(
            name="cycle_time_efficiency",
            weight=0.25,
            direction=ScoreDirection.HIGHER_IS_BETTER,
            min_value=0,
            max_value=100,
            description="Actual vs target cycle time"
        ),
        ScoreComponent(
            name="throughput_rate",
            weight=0.25,
            direction=ScoreDirection.HIGHER_IS_BETTER,
            min_value=0,
            max_value=100,
            description="Output volume vs capacity"
        ),
        ScoreComponent(
            name="first_pass_yield",
            weight=0.20,
            direction=ScoreDirection.HIGHER_IS_BETTER,
            min_value=0,
            max_value=100,
            description="Work completed correctly first time"
        ),
        ScoreComponent(
            name="rework_rate",
            weight=0.15,
            direction=ScoreDirection.LOWER_IS_BETTER,
            min_value=0,
            max_value=30,
            description="Percentage requiring rework"
        ),
        ScoreComponent(
            name="process_adherence",
            weight=0.15,
            direction=ScoreDirection.HIGHER_IS_BETTER,
            min_value=0,
            max_value=100,
            description="Compliance with standard procedures"
        ),
    ]
    return WeightedScoringEngine(components)


def create_resource_utilization_engine() -> WeightedScoringEngine:
    """Create a resource utilization scoring engine."""
    components = [
        ScoreComponent(
            name="labor_utilization",
            weight=0.30,
            direction=ScoreDirection.HIGHER_IS_BETTER,
            min_value=0,
            max_value=100,
            description="Productive hours vs available hours"
        ),
        ScoreComponent(
            name="equipment_utilization",
            weight=0.25,
            direction=ScoreDirection.HIGHER_IS_BETTER,
            min_value=0,
            max_value=100,
            description="Equipment uptime vs available time"
        ),
        ScoreComponent(
            name="space_utilization",
            weight=0.15,
            direction=ScoreDirection.HIGHER_IS_BETTER,
            min_value=0,
            max_value=100,
            description="Facility space efficiency"
        ),
        ScoreComponent(
            name="overtime_ratio",
            weight=0.15,
            direction=ScoreDirection.LOWER_IS_BETTER,
            min_value=0,
            max_value=30,
            description="Overtime as % of regular hours"
        ),
        ScoreComponent(
            name="idle_time_ratio",
            weight=0.15,
            direction=ScoreDirection.LOWER_IS_BETTER,
            min_value=0,
            max_value=30,
            description="Non-productive waiting time"
        ),
    ]
    return WeightedScoringEngine(components)
