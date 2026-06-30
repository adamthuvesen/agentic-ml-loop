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

OBJECTIVE_METRIC = "val_r2_log"
SPLIT_STRATEGY = "expanding_year_folds_2015_2022_val_2023_test"
EXPERIMENT_ID = "box-office-oracle"
HELPER = ROOT / "lib" / "external_targets" / "box_office_oracle_eval.py"

BOX_OFFICE_DUMMY_ENV = {
    "AWS_S3_BUCKET": "dummy",
    "SAGEMAKER_ROLE_ARN": "arn:aws:iam::123456789012:role/dummy",
    "SNOWFLAKE_USER": "dummy",
    "SNOWFLAKE_ACCOUNT": "dummy",
    "SNOWFLAKE_DATABASE": "BOX_OFFICE",
    "SNOWFLAKE_WAREHOUSE": "DUMMY",
    "SNOWFLAKE_ROLE": "DBT_RUNNER",
    "SNOWFLAKE_PRIVATE_KEY_PATH": "/tmp/dummy.pem",
    "AWS_REGION": "eu-north-1",
    "TMDB_API_TOKEN": "dummy",
    "SNOWFLAKE_SCHEMA_RAW": "RAW",
    "SNOWFLAKE_SCHEMA_STAGING": "STAGING",
    "SNOWFLAKE_SCHEMA_ML_TRAINING": "ML_TRAINING",
    "SNOWFLAKE_SCHEMA_FEATURE_STORE": "FEATURE_STORE",
}


def _repo() -> Path:
    return resolve_external_repo("BOX_OFFICE_ORACLE_REPO", "box-office-oracle")


def run_candidate(
    candidate_id: str,
    experiment_dir: Path | None = None,
    *,
    include_test: bool = False,
) -> CandidateResult:
    extra_args: list[str] = []
    if experiment_dir is not None:
        extra_args.extend(["--experiment-dir", str(experiment_dir.resolve())])
    if include_test:
        extra_args.append("--include-test")
    payload = run_uv_project_helper(
        UvProjectHelperRequest(
            repo=_repo(),
            helper=HELPER,
            candidate_id=candidate_id,
            extra_args=extra_args or None,
            env=BOX_OFFICE_DUMMY_ENV,
            timeout_seconds=900,
        )
    )
    return candidate_result_from_payload(payload)


def run_log_budget_baseline(_splits: object = None) -> CandidateResult:
    return run_candidate("log-budget-baseline")


def run_ridge_production_13(_splits: object = None) -> CandidateResult:
    return run_candidate("ridge-production-13")


def run_xgb_production_13(_splits: object = None) -> CandidateResult:
    return run_candidate("xgb-production-13")


def run_xgb_compact_budget_demand(_splits: object = None) -> CandidateResult:
    return run_candidate("xgb-compact-budget-demand")


def run_xgb_no_covid_weekend(_splits: object = None) -> CandidateResult:
    return run_candidate("xgb-no-covid-weekend")


def run_xgb_regularized_shallow(_splits: object = None) -> CandidateResult:
    return run_candidate("xgb-regularized-shallow")


def run_random_forest_production_13(_splits: object = None) -> CandidateResult:
    return run_candidate("random-forest-production-13")


CANDIDATE_RUNNERS = {
    "log-budget-baseline": run_log_budget_baseline,
    "ridge-production-13": run_ridge_production_13,
    "xgb-production-13": run_xgb_production_13,
    "xgb-compact-budget-demand": run_xgb_compact_budget_demand,
    "xgb-no-covid-weekend": run_xgb_no_covid_weekend,
    "xgb-regularized-shallow": run_xgb_regularized_shallow,
    "random-forest-production-13": run_random_forest_production_13,
}


def load_splits() -> None:
    return None
