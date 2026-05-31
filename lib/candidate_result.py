from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CandidateResult:
    candidate_id: str
    model_family: str
    feature_set: str
    objective_metric: str
    objective_score: float
    split_strategy: str
    status: str
    notes: str
    metrics: dict[str, dict[str, float]]
    hyperparameters: dict[str, Any]
    selected_features: list[str]

    def result_payload(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "model_family": self.model_family,
            "feature_set": self.feature_set,
            "objective_metric": self.objective_metric,
            "objective_score": self.objective_score,
            "split_strategy": self.split_strategy,
            "status": self.status,
            "notes": self.notes,
            "metrics": self.metrics,
            "hyperparameters": self.hyperparameters,
            "selected_features": self.selected_features,
        }
