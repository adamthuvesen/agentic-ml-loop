from __future__ import annotations

import textwrap
from typing import Any

from lib.notebook_recipe import NotebookRecipe, NotebookRecipeError


def _notebook_cells(recipe: NotebookRecipe, *, include_sensitive: bool) -> list[dict[str, Any]]:
    cells = [
        _markdown_cell(
            f"# {recipe.title}\n\nPortable notebook export for `{recipe.experiment_id}`."
        ),
        _markdown_cell(
            "This notebook is self-contained: it reads a parquet file from `DATA_PATH` "
            "and does not import repository-local experiment modules."
        ),
        _code_cell(_configuration_source(recipe)),
        _code_cell(_shared_imports_source(recipe)),
        _code_cell(_data_loading_source(recipe)),
    ]

    if recipe.recipe_type == "final_model":
        cells.extend(
            [
                _markdown_cell("## Final Model Path"),
                _code_cell(_final_model_source(recipe, include_sensitive=include_sensitive)),
            ]
        )
    elif recipe.recipe_type == "analysis_pipeline":
        cells.extend(
            [
                _markdown_cell("## Analysis Pipeline"),
                _code_cell(_analysis_pipeline_source(recipe)),
            ]
        )
    else:  # pragma: no cover - validation prevents this branch.
        raise NotebookRecipeError(f"Unsupported recipe type: {recipe.recipe_type}")

    cells.append(_markdown_cell("## Declared Outputs"))
    cells.append(_code_cell(_outputs_source(recipe, include_sensitive=include_sensitive)))
    if recipe.sensitive_outputs and not include_sensitive:
        cells.append(
            _markdown_cell(
                "Sensitive outputs were excluded from this notebook export. "
                "Regenerate with `--include-sensitive` only when row-level sharing is intended."
            )
        )
    return cells


def _configuration_source(recipe: NotebookRecipe) -> str:
    return _dedent(
        f"""
        from __future__ import annotations

        from pathlib import Path

        DATA_PATH = Path({recipe.data_path_hint!r})
        OUTPUT_DIR = Path("notebook_outputs")
        RUN_PIPELINE = {recipe.run_pipeline_default!r}

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        """
    )


def _shared_imports_source(recipe: NotebookRecipe) -> str:
    lines = [
        "import numpy as np",
        "import pandas as pd",
    ]
    if recipe.recipe_type == "final_model":
        lines.append(
            "from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score"
        )
        if recipe.model_family == "xgboost":
            lines.append("from xgboost import XGBClassifier")
        else:
            lines.extend(
                [
                    "from sklearn.impute import SimpleImputer",
                    "from sklearn.linear_model import LogisticRegression",
                    "from sklearn.pipeline import Pipeline",
                    "from sklearn.preprocessing import StandardScaler",
                ]
            )
    else:
        lines.extend(
            [
                "import json",
                "from sklearn.metrics import roc_auc_score",
                "from scipy.cluster.hierarchy import fcluster, linkage",
                "from scipy.spatial.distance import squareform",
            ]
        )
    return "\n".join(lines) + "\n"


def _data_loading_source(recipe: NotebookRecipe) -> str:
    return _dedent(
        f"""
        TARGET_COLUMN = {recipe.target_column!r}
        TIME_COLUMN = {recipe.time_column!r}
        ELIGIBILITY_REGION_COLUMN = {recipe.eligibility_region_column!r}
        ELIGIBILITY_GROUP_COLUMN = {recipe.eligibility_group_column!r}
        ELIGIBLE_REGIONS = {recipe.eligible_regions!r}
        ELIGIBLE_GROUPS = {recipe.eligible_groups!r}
        TRAIN_DATES = {recipe.train_dates!r}
        VALIDATION_DATES = {recipe.validation_dates!r}
        DEVELOPMENT_DATES = {recipe.development_dates!r}
        TEST_DATES = {recipe.test_dates!r}
        HOLDOUT_DATES = {recipe.holdout_dates!r}
        EXCLUDE_COLUMNS = {recipe.exclude_columns!r}

        def load_dataset(path):
            if not path.exists():
                raise FileNotFoundError(
                    f"Parquet file not found at {{path}}. Update DATA_PATH in the configuration cell."
                )
            df = pd.read_parquet(path)
            df[TIME_COLUMN] = df[TIME_COLUMN].astype(str)
            mask = pd.Series(True, index=df.index)
            if ELIGIBILITY_REGION_COLUMN and ELIGIBLE_REGIONS:
                mask &= df[ELIGIBILITY_REGION_COLUMN].isin(ELIGIBLE_REGIONS)
            if ELIGIBILITY_GROUP_COLUMN and ELIGIBLE_GROUPS:
                mask &= df[ELIGIBILITY_GROUP_COLUMN].isin(ELIGIBLE_GROUPS)
            return df.loc[mask].reset_index(drop=True)

        def split_by_dates(df, dates):
            return df.loc[df[TIME_COLUMN].isin(dates)].reset_index(drop=True)

        def frame_stats(frame):
            return {{
                "rows": int(len(frame)),
                "positives": int(frame[TARGET_COLUMN].sum()),
                "prevalence": float(frame[TARGET_COLUMN].mean()) if len(frame) else 0.0,
            }}

        df = load_dataset(DATA_PATH)
        print(f"Eligible rows: {{len(df):,}}")
        """
    )


def _final_model_source(recipe: NotebookRecipe, *, include_sensitive: bool) -> str:
    sensitive_scoring_source = ""
    if include_sensitive:
        sensitive_scoring_source = """
        scored_test = test.assign(
            final_model_score=test_scores,
            rank=np.argsort(-test_scores).argsort() + 1,
        ).sort_values("rank")
        scored_columns = [
            TIME_COLUMN,
            "rank",
            "final_model_score",
            TARGET_COLUMN,
            *[column for column in ["ENTITY_ID", "REGION", "SEGMENT"] if column in scored_test.columns],
            *SELECTED_FEATURES,
        ]
        scored_test[scored_columns].to_csv(OUTPUT_DIR / "top_accounts_q4.csv", index=False)
        """
    if recipe.model_family == "xgboost":
        pipeline_factory_source = """
        def build_xgb_model(scale_pos_weight, n_estimators, early_stopping_rounds=None):
            params = dict(
                max_depth=4,
                learning_rate=0.05,
                n_estimators=n_estimators,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=1.0,
                objective="binary:logistic",
                eval_metric="auc",
                tree_method="hist",
                scale_pos_weight=scale_pos_weight,
                random_state=42,
            )
            if early_stopping_rounds is not None:
                params["early_stopping_rounds"] = early_stopping_rounds
            return XGBClassifier(**params)

        def compute_scale_pos_weight(y):
            positives = int(y.sum())
            negatives = int(len(y) - positives)
            return float(negatives / positives) if positives > 0 else 1.0
        """
        fit_predict_source = """
        selection_y = selection_train[TARGET_COLUMN]
        selection_model = build_xgb_model(
            scale_pos_weight=compute_scale_pos_weight(selection_y),
            n_estimators=2000,
            early_stopping_rounds=50,
        )
        selection_model.fit(
            selection_train[SELECTED_FEATURES],
            selection_y,
            eval_set=[(validation[SELECTED_FEATURES], validation[TARGET_COLUMN])],
            verbose=False,
        )
        best_iteration = int(selection_model.best_iteration) + 1
        validation_scores = selection_model.predict_proba(validation[SELECTED_FEATURES])[:, 1]

        development_y = development[TARGET_COLUMN]
        final_model = build_xgb_model(
            scale_pos_weight=compute_scale_pos_weight(development_y),
            n_estimators=best_iteration,
        )
        final_model.fit(development[SELECTED_FEATURES], development_y, verbose=False)
        test_scores = final_model.predict_proba(test[SELECTED_FEATURES])[:, 1]
        print(f"XGB best_iteration on validation: {best_iteration}")
        """
    else:
        pipeline_factory_source = """
        def build_lr_pipeline():
            return Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scaler", StandardScaler()),
                    (
                        "estimator",
                        LogisticRegression(
                            C=1.0,
                            l1_ratio=0.0,
                            class_weight="balanced",
                            solver="lbfgs",
                            max_iter=2000,
                            random_state=42,
                        ),
                    ),
                ]
            )
        """
        fit_predict_source = """
        selection_pipeline = build_lr_pipeline()
        selection_pipeline.fit(selection_train[SELECTED_FEATURES], selection_train[TARGET_COLUMN])
        validation_scores = selection_pipeline.predict_proba(validation[SELECTED_FEATURES])[:, 1]

        final_pipeline = build_lr_pipeline()
        final_pipeline.fit(development[SELECTED_FEATURES], development[TARGET_COLUMN])
        test_scores = final_pipeline.predict_proba(test[SELECTED_FEATURES])[:, 1]
        """
    return _dedent(
        f"""
        SELECTED_FEATURES = {recipe.selected_features!r}
        MODEL_LABEL = {recipe.model_label or "final_model"!r}
        MODEL_FAMILY = {recipe.model_family!r}
        INCUMBENT_SCORE_COLUMN = {recipe.incumbent_score_column!r}
        INCUMBENT_PROBABILITY_SCALE = {recipe.incumbent_probability_scale!r}
        TOP_K_VALUES = (100, 500, 1000)
{pipeline_factory_source}
        def recall_at_k(y_true, scores, k):
            positives = float(y_true.sum())
            if positives == 0 or len(y_true) == 0:
                return 0.0
            top = np.argsort(-scores)[: min(k, len(y_true))]
            return float(y_true[top].sum()) / positives

        def lift_at_k(y_true, scores, k):
            base = float(y_true.mean())
            if base == 0 or len(y_true) == 0:
                return 0.0
            top = np.argsort(-scores)[: min(k, len(y_true))]
            return float(y_true[top].mean()) / base

        def evaluate_predictions(y_true, rank_scores, probability_scores=None):
            y = y_true.to_numpy().astype(int)
            ranks = np.asarray(rank_scores, dtype=float)
            probabilities = ranks if probability_scores is None else np.asarray(probability_scores, dtype=float)
            clipped_probabilities = np.clip(probabilities, 1e-9, 1 - 1e-9)
            metrics = {{
                "auc": float(roc_auc_score(y, ranks)),
                "average_precision": float(average_precision_score(y, ranks)),
                "log_loss": float(log_loss(y, clipped_probabilities, labels=[0, 1])),
                "brier": float(brier_score_loss(y, clipped_probabilities)),
                "positive_rate": float(y.mean()),
            }}
            for k in TOP_K_VALUES:
                top = np.argsort(-ranks)[: min(k, len(y))]
                metrics[f"precision_at_{{k}}"] = float(y[top].mean()) if len(top) else 0.0
                metrics[f"recall_at_{{k}}"] = recall_at_k(y, ranks, k)
                metrics[f"lift_at_{{k}}"] = lift_at_k(y, ranks, k)
            return metrics

        selection_train = split_by_dates(df, TRAIN_DATES)
        validation = split_by_dates(df, VALIDATION_DATES)
        development = split_by_dates(df, DEVELOPMENT_DATES)
        test = split_by_dates(df, TEST_DATES)
        print("Split stats:")
        print(pd.DataFrame([
            {{"split": "selection_train", **frame_stats(selection_train)}},
            {{"split": "validation", **frame_stats(validation)}},
            {{"split": "development", **frame_stats(development)}},
            {{"split": "test", **frame_stats(test)}},
        ]).to_string(index=False))

        {fit_predict_source}
        metrics = pd.DataFrame([
            {{"model": MODEL_LABEL, "split": "Q3 validation", **evaluate_predictions(validation[TARGET_COLUMN], validation_scores)}},
            {{"model": MODEL_LABEL, "split": "Q4 final test", **evaluate_predictions(test[TARGET_COLUMN], test_scores)}},
        ])

        if INCUMBENT_SCORE_COLUMN and INCUMBENT_SCORE_COLUMN in test.columns:
            incumbent_rank_scores = test[INCUMBENT_SCORE_COLUMN].to_numpy(dtype=float)
            incumbent_probability_scores = (
                incumbent_rank_scores / INCUMBENT_PROBABILITY_SCALE
                if INCUMBENT_PROBABILITY_SCALE
                else incumbent_rank_scores
            )
            metrics = pd.concat(
                [
                    metrics,
                    pd.DataFrame([
                        {{
                            "model": INCUMBENT_SCORE_COLUMN,
                            "split": "Q4 final test",
                            **evaluate_predictions(
                                test[TARGET_COLUMN],
                                incumbent_rank_scores,
                                incumbent_probability_scores,
                            ),
                        }}
                    ]),
                ],
                ignore_index=True,
            )

        display(metrics)
        metrics.to_csv(OUTPUT_DIR / "final_model_metrics.csv", index=False)
        {sensitive_scoring_source}
        """
    )


def _analysis_pipeline_source(recipe: NotebookRecipe) -> str:
    return _dedent(
        """
        def candidate_features(frame):
            return [
                column
                for column in frame.columns
                if column not in set(EXCLUDE_COLUMNS)
                and pd.api.types.is_numeric_dtype(frame[column])
            ]

        def profile_features(train, features):
            rows = []
            for feature in features:
                series = train[feature]
                non_null = series.dropna()
                top_share = (
                    float(non_null.value_counts(normalize=True).iloc[0])
                    if not non_null.empty
                    else 0.0
                )
                rows.append(
                    {
                        "feature": feature,
                        "dtype": str(series.dtype),
                        "n_rows": int(len(series)),
                        "n_null": int(series.isna().sum()),
                        "null_rate": float(series.isna().mean()),
                        "n_unique": int(non_null.nunique(dropna=True)),
                        "top_value_share": top_share,
                        "is_constant": bool(non_null.nunique(dropna=True) <= 1),
                        "is_near_constant": bool(top_share > 0.99 and non_null.nunique(dropna=True) > 1),
                    }
                )
            return pd.DataFrame(rows)

        def safe_auc(y, scores):
            mask = ~np.isnan(scores)
            if mask.sum() < 50 or len(np.unique(y[mask])) < 2:
                return np.nan
            auc = roc_auc_score(y[mask], scores[mask])
            return float(max(auc, 1.0 - auc))

        def univariate_auc(train, features):
            y = train[TARGET_COLUMN].astype(int).to_numpy()
            rows = []
            for feature in features:
                scores = train[feature].astype(float).to_numpy()
                rows.append({"feature": feature, "univariate_auc": safe_auc(y, scores)})
            return pd.DataFrame(rows)

        def cluster_features(train, ranked):
            features = ranked["feature"].tolist()
            corr = train[features].corr(method="spearman").abs().fillna(0.0)
            distance = 1.0 - corr
            np.fill_diagonal(distance.values, 0.0)
            condensed = squareform(distance.to_numpy(), checks=False)
            labels = fcluster(linkage(condensed, method="average"), t=0.30, criterion="distance")
            cluster_map = pd.DataFrame({"feature": features, "cluster_id": labels})
            cluster_sizes = cluster_map.groupby("cluster_id").size().rename("cluster_size")
            return cluster_map.join(cluster_sizes, on="cluster_id")

        def stability_flags(analysis, features):
            rows = []
            for feature in features:
                per_quarter = []
                for quarter, sub in analysis.groupby(TIME_COLUMN):
                    auc = safe_auc(
                        sub[TARGET_COLUMN].astype(int).to_numpy(),
                        sub[feature].astype(float).to_numpy(),
                    )
                    per_quarter.append(
                        {
                            "quarter": quarter,
                            "auc": auc,
                            "null_rate": float(sub[feature].isna().mean()),
                            "mean": float(sub[feature].mean()) if sub[feature].notna().any() else np.nan,
                        }
                    )
                per_q = pd.DataFrame(per_quarter)
                auc_range = float(per_q["auc"].max() - per_q["auc"].min())
                null_range = float(per_q["null_rate"].max() - per_q["null_rate"].min())
                rows.append(
                    {
                        "feature": feature,
                        "auc_range_3q": auc_range,
                        "null_rate_range_3q": null_range,
                        "quarter_rank_flag": "red" if auc_range > 0.05 else "yellow" if auc_range > 0.025 else "green",
                        "null_rate_drift_flag": "red" if null_range > 0.10 else "yellow" if null_range > 0.03 else "green",
                    }
                )
            return pd.DataFrame(rows)

        train = split_by_dates(df, TRAIN_DATES)
        validation = split_by_dates(df, VALIDATION_DATES)
        holdout = split_by_dates(df, HOLDOUT_DATES)
        analysis = pd.concat([train, validation], ignore_index=True)
        print("Holdout policy: Q4/reserved holdout rows are counted but not used in feature analysis.")
        print(pd.DataFrame([
            {"split": "train", **frame_stats(train)},
            {"split": "validation", **frame_stats(validation)},
            {"split": "holdout_reserved", **frame_stats(holdout)},
        ]).to_string(index=False))

        if RUN_PIPELINE:
            features = candidate_features(train)
            profile = profile_features(train, features)
            rankable = profile.loc[~profile["is_constant"] & ~profile["is_near_constant"], "feature"].tolist()
            uni = univariate_auc(train, rankable)
            ranked = profile.merge(uni, on="feature", how="left")
            ranked["stakeholder_rank"] = ranked["univariate_auc"].rank(ascending=False, method="first")
            ranked = ranked.sort_values(["stakeholder_rank", "feature"]).reset_index(drop=True)
            cluster_map = cluster_features(train, ranked.dropna(subset=["univariate_auc"]))
            stability = stability_flags(analysis, cluster_map["feature"].tolist())
            ranked = ranked.merge(cluster_map, on="feature", how="left").merge(stability, on="feature", how="left")
            shortlist = (
                ranked.dropna(subset=["cluster_id"])
                .sort_values(["cluster_id", "stakeholder_rank"])
                .groupby("cluster_id")
                .head(1)
                .sort_values("stakeholder_rank")
                .head(30)
            )

            ranked.to_csv(OUTPUT_DIR / "all_features_ranked.csv", index=False)
            cluster_map.to_csv(OUTPUT_DIR / "cluster_map.csv", index=False)
            stability.to_csv(OUTPUT_DIR / "stability_flags.csv", index=False)
            (OUTPUT_DIR / "shortlist.json").write_text(
                json.dumps(
                    {
                        "n_features": int(len(shortlist)),
                        "shortlist": shortlist["feature"].tolist(),
                    },
                    indent=2,
                )
            )
        else:
            print("RUN_PIPELINE is False. Flip it to True to recompute outputs from the parquet.")
            ranked = pd.read_csv(OUTPUT_DIR / "all_features_ranked.csv") if (OUTPUT_DIR / "all_features_ranked.csv").exists() else pd.DataFrame()
            stability = pd.read_csv(OUTPUT_DIR / "stability_flags.csv") if (OUTPUT_DIR / "stability_flags.csv").exists() else pd.DataFrame()
            shortlist = pd.DataFrame()

        display(ranked.head(20) if not ranked.empty else ranked)
        """
    )


def _outputs_source(recipe: NotebookRecipe, *, include_sensitive: bool) -> str:
    sensitive = recipe.sensitive_outputs if include_sensitive else []
    included = recipe.safe_outputs + sensitive
    return _dedent(
        f"""
        DECLARED_OUTPUTS = {included!r}
        SENSITIVE_OUTPUTS = {sensitive!r}
        SENSITIVE_OUTPUTS_INCLUDED = {include_sensitive!r}

        print("Declared outputs in this portable export:")
        for path in DECLARED_OUTPUTS:
            label = "sensitive" if path in SENSITIVE_OUTPUTS else "safe"
            print(f"- {{path}} ({{label}})")
        """
    )


def _markdown_cell(source: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": _source_lines(source)}


def _code_cell(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source_lines(source),
    }


def _source_lines(source: str) -> list[str]:
    return source.splitlines(keepends=True)


def _dedent(source: str) -> str:
    return textwrap.dedent(source).strip() + "\n"
