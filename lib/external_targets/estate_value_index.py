from __future__ import annotations

from pathlib import Path

from lib.candidate_result import CandidateResult
from lib.external_targets.common import (
    ROOT,
    UvProjectHelperRequest,
    candidate_result_from_payload,
    resolve_external_repo,
    run_uv_project_helper,
)

OBJECTIVE_METRIC = "val_neg_mae"
SPLIT_STRATEGY = "temporal_60_20_20_by_sold_date"
EXPERIMENT_ID = "estate-value-index"
HELPER = ROOT / "lib" / "external_targets" / "estate_value_index_eval.py"


def _repo() -> Path:
    return resolve_external_repo("ESTATE_VALUE_INDEX_REPO", "estate-value-index")


def run_candidate(
    candidate_id: str,
    experiment_dir: Path | None = None,
    *,
    include_test: bool = False,
) -> CandidateResult:
    repo = _repo()
    extra_args: list[str] = []
    if experiment_dir is not None:
        extra_args.extend(["--experiment-dir", str(experiment_dir.resolve())])
    if include_test:
        extra_args.append("--include-test")
    payload = run_uv_project_helper(
        UvProjectHelperRequest(
            repo=repo,
            helper=HELPER,
            candidate_id=candidate_id,
            extra_args=extra_args or None,
            env={
                "GCS_ENABLED": "false",
                "EVI_CONFIG_FILE": str(repo / "config" / "pipeline_config.yaml"),
            },
            timeout_seconds=900,
        )
    )
    return candidate_result_from_payload(payload)


def run_listing_price_baseline(_splits: object = None) -> CandidateResult:
    return run_candidate("listing-price-baseline")


def run_median_baseline(_splits: object = None) -> CandidateResult:
    return run_candidate("median-baseline")


def run_lgbm_lean6(_splits: object = None) -> CandidateResult:
    return run_candidate("lgbm-lean6")


def run_lgbm_lean8(_splits: object = None) -> CandidateResult:
    return run_candidate("lgbm-lean8")


def run_lgbm_lean10(_splits: object = None) -> CandidateResult:
    return run_candidate("lgbm-lean10")


def run_lgbm_lean15_area(_splits: object = None) -> CandidateResult:
    return run_candidate("lgbm-lean15-area")


def run_lgbm_lean10_pruned(_splits: object = None) -> CandidateResult:
    return run_candidate("lgbm-lean10-pruned")


def run_lgbm_full_registry(_splits: object = None) -> CandidateResult:
    return run_candidate("lgbm-full-registry")


CANDIDATE_RUNNERS = {
    "listing-price-baseline": run_listing_price_baseline,
    "median-baseline": run_median_baseline,
    "lgbm-lean6": run_lgbm_lean6,
    "lgbm-lean8": run_lgbm_lean8,
    "lgbm-lean10": run_lgbm_lean10,
    "lgbm-lean15-area": run_lgbm_lean15_area,
    "lgbm-lean10-pruned": run_lgbm_lean10_pruned,
    "lgbm-full-registry": run_lgbm_full_registry,
}


def load_splits() -> None:
    return None
