"""
Risk Classification Pattern - Operations Intelligence

Flexible risk classification for operational health assessment.
Converts continuous scores into discrete risk levels.

Use cases:
- Process health classification
- Capacity risk tiers
- Quality risk assessment
- Delivery risk ratings
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Standard risk levels with associated properties."""
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    MINIMAL = "Minimal"

    @property
    def priority(self) -> int:
        return {
            RiskLevel.CRITICAL: 1,
            RiskLevel.HIGH: 2,
            RiskLevel.MEDIUM: 3,
            RiskLevel.LOW: 4,
            RiskLevel.MINIMAL: 5
        }[self]

    @property
    def color(self) -> str:
        return {
            RiskLevel.CRITICAL: "#dc3545",
            RiskLevel.HIGH: "#fd7e14",
            RiskLevel.MEDIUM: "#ffc107",
            RiskLevel.LOW: "#28a745",
            RiskLevel.MINIMAL: "#17a2b8"
        }[self]

    @property
    def icon(self) -> str:
        return {
            RiskLevel.CRITICAL: "🔴",
            RiskLevel.HIGH: "🟠",
            RiskLevel.MEDIUM: "🟡",
            RiskLevel.LOW: "🟢",
            RiskLevel.MINIMAL: "🔵"
        }[self]


@dataclass
class RiskThreshold:
    """Configuration for a single risk threshold."""
    level: RiskLevel
    min_score: float
    max_score: float
    description: str = ""
    action_required: str = ""


@dataclass
class RiskClassification:
    """Result of classifying an entity's risk."""
    entity_id: str
    score: float
    level: RiskLevel
    description: str
    action_required: str
    threshold_details: Dict[str, Any] = field(default_factory=dict)
    factors: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "score": self.score,
            "level": self.level.value,
            "level_priority": self.level.priority,
            "level_color": self.level.color,
            "level_icon": self.level.icon,
            "description": self.description,
            "action_required": self.action_required,
            "factors": self.factors,
            "metadata": self.metadata
        }


class RiskClassifier:
    """Classifies continuous scores into discrete risk levels."""

    DEFAULT_HEALTH_THRESHOLDS = [
        RiskThreshold(RiskLevel.CRITICAL, 0, 30, "Critical operational issues requiring immediate attention",
                     "Escalate to leadership; halt non-critical operations"),
        RiskThreshold(RiskLevel.HIGH, 30, 50, "Significant operational concerns",
                     "Prioritize remediation; daily monitoring"),
        RiskThreshold(RiskLevel.MEDIUM, 50, 70, "Moderate operational issues",
                     "Monitor closely; address in next review cycle"),
        RiskThreshold(RiskLevel.LOW, 70, 85, "Minor concerns within acceptable range",
                     "Routine monitoring; no immediate action needed"),
        RiskThreshold(RiskLevel.MINIMAL, 85, 100, "Excellent operational health",
                     "Continue standard monitoring"),
    ]

    DEFAULT_RISK_THRESHOLDS = [
        RiskThreshold(RiskLevel.MINIMAL, 0, 15, "Very low operational risk",
                     "Standard procedures apply"),
        RiskThreshold(RiskLevel.LOW, 15, 30, "Low risk with minor concerns",
                     "Monitor as part of routine oversight"),
        RiskThreshold(RiskLevel.MEDIUM, 30, 50, "Moderate risk requiring attention",
                     "Implement additional controls; regular review"),
        RiskThreshold(RiskLevel.HIGH, 50, 70, "High risk requiring active management",
                     "Develop mitigation plan; frequent monitoring"),
        RiskThreshold(RiskLevel.CRITICAL, 70, 100, "Critical risk threatening operations",
                     "Immediate executive attention; contingency activation"),
    ]

    def __init__(
        self,
        direction: str = "lower_is_riskier",
        thresholds: Optional[List[RiskThreshold]] = None
    ):
        self.direction = direction
        if thresholds:
            self.thresholds = sorted(thresholds, key=lambda t: t.min_score)
        elif direction == "lower_is_riskier":
            self.thresholds = self.DEFAULT_HEALTH_THRESHOLDS
        else:
            self.thresholds = self.DEFAULT_RISK_THRESHOLDS
        self._validate_thresholds()

    def _validate_thresholds(self) -> None:
        if not self.thresholds:
            raise ValueError("At least one threshold must be defined")

    def classify(
        self,
        score: float,
        entity_id: str = "unknown",
        factors: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> RiskClassification:
        min_score = self.thresholds[0].min_score
        max_score = self.thresholds[-1].max_score
        clamped_score = max(min_score, min(max_score, score))

        matched_threshold = None
        for threshold in self.thresholds:
            if threshold.min_score <= clamped_score < threshold.max_score:
                matched_threshold = threshold
                break

        if matched_threshold is None and clamped_score == max_score:
            matched_threshold = self.thresholds[-1]

        if matched_threshold is None:
            matched_threshold = self.thresholds[-1]

        return RiskClassification(
            entity_id=entity_id,
            score=round(score, 2),
            level=matched_threshold.level,
            description=matched_threshold.description,
            action_required=matched_threshold.action_required,
            threshold_details={
                "min_score": matched_threshold.min_score,
                "max_score": matched_threshold.max_score,
            },
            factors=factors or [],
            metadata=metadata or {}
        )

    def classify_batch(
        self,
        entities: List[Dict[str, Any]],
        score_field: str = "score",
        id_field: str = "id"
    ) -> List[RiskClassification]:
        results = []
        for entity in entities:
            score = entity.get(score_field)
            if score is None:
                continue
            entity_id = str(entity.get(id_field, "unknown"))
            metadata = {k: v for k, v in entity.items() if k not in [score_field, id_field]}
            result = self.classify(score, entity_id=entity_id, metadata=metadata)
            results.append(result)
        return results

    def get_risk_distribution(self, classifications: List[RiskClassification]) -> Dict[str, Any]:
        if not classifications:
            return {"total": 0, "distribution": {}}

        distribution = {}
        for level in RiskLevel:
            count = sum(1 for c in classifications if c.level == level)
            distribution[level.value] = {
                "count": count,
                "percentage": round(count / len(classifications) * 100, 1),
                "color": level.color
            }

        return {
            "total": len(classifications),
            "distribution": distribution,
            "highest_risk": max(classifications, key=lambda c: c.level.priority).level.value,
            "average_score": round(sum(c.score for c in classifications) / len(classifications), 2)
        }


class MultiDimensionalRiskClassifier:
    """Classifies risk based on multiple operational dimensions."""

    def __init__(
        self,
        dimensions: Dict[str, RiskClassifier],
        weights: Optional[Dict[str, float]] = None,
        aggregation: str = "weighted_average"
    ):
        self.dimensions = dimensions
        self.aggregation = aggregation

        if weights:
            self.weights = weights
        else:
            equal_weight = 1.0 / len(dimensions)
            self.weights = {name: equal_weight for name in dimensions}

        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

    def classify(
        self,
        scores: Dict[str, float],
        entity_id: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None
    ) -> RiskClassification:
        dimension_results = {}
        factors = []

        for dim_name, classifier in self.dimensions.items():
            score = scores.get(dim_name)
            if score is None:
                continue

            result = classifier.classify(score, entity_id=f"{entity_id}_{dim_name}")
            dimension_results[dim_name] = result

            factors.append({
                "dimension": dim_name,
                "score": score,
                "level": result.level.value,
                "weight": self.weights.get(dim_name, 0),
            })

        if self.aggregation == "weighted_average":
            aggregate_score = sum(
                scores.get(dim, 0) * self.weights.get(dim, 0)
                for dim in self.dimensions.keys()
            )
        elif self.aggregation == "worst_case":
            aggregate_score = min(scores.get(dim, 100) for dim in self.dimensions.keys())
        else:
            aggregate_score = max(scores.get(dim, 0) for dim in self.dimensions.keys())

        first_classifier = list(self.dimensions.values())[0]
        return first_classifier.classify(
            aggregate_score,
            entity_id=entity_id,
            factors=factors,
            metadata={
                **(metadata or {}),
                "dimension_results": {k: v.to_dict() for k, v in dimension_results.items()},
                "aggregation_method": self.aggregation
            }
        )


# Factory Functions

def create_health_score_classifier() -> RiskClassifier:
    """Create a classifier for health scores (0-100, higher = healthier)."""
    return RiskClassifier(direction="lower_is_riskier")


def create_operational_risk_classifier() -> MultiDimensionalRiskClassifier:
    """Create a multi-dimensional operational risk classifier."""
    return MultiDimensionalRiskClassifier(
        dimensions={
            "efficiency": create_health_score_classifier(),
            "quality": create_health_score_classifier(),
            "capacity": create_health_score_classifier(),
            "reliability": create_health_score_classifier(),
        },
        weights={
            "efficiency": 0.30,
            "quality": 0.30,
            "capacity": 0.20,
            "reliability": 0.20,
        },
        aggregation="weighted_average"
    )
