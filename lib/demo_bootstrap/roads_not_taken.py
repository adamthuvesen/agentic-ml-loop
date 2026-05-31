"""Cycle 0005: Falsification tests on untried model families and techniques.

Tests whether target encoding, Naive Bayes, or SVM can break the ~0.60 AUC ceiling.
All use the same 2 signal features (sessions_7d, is_enterprise) for fair comparison.
"""

from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from lib.demo_bootstrap.data import TARGET_COLUMN, load_dataset, split_dataset


def _prepare_splits():
    """Load data and prepare the 2-feature matrix."""
    df = load_dataset()
    splits = split_dataset(df)

    dfs = {}
    for name, frame in [
        ("train", splits.train),
        ("val", splits.validation),
        ("test", splits.test),
    ]:
        f = frame.copy()
        f["is_enterprise"] = (f["segment"] == "enterprise").astype(int)
        dfs[name] = f

    features = ["sessions_7d", "is_enterprise"]
    return dfs, features, splits


def _bootstrap_ci(y_true, y_prob, n_boot=2000, seed=42):
    """Bootstrap 95% CI for AUC."""
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        yt, yp = y_true.iloc[idx], y_prob[idx]
        if yt.nunique() < 2:
            continue
        aucs.append(roc_auc_score(yt, yp))
    aucs = np.array(aucs)
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


def test_target_encoding():
    """Target encoding of segment using train-only means (no leakage)."""
    dfs, _, splits = _prepare_splits()

    # Compute segment-level target mean on train only
    seg_means = dfs["train"].groupby("segment")[TARGET_COLUMN].mean()
    print("Segment target encoding (train-only means):")
    print(seg_means.to_string())
    print()

    for name in ["train", "val", "test"]:
        dfs[name]["segment_target_enc"] = dfs[name]["segment"].map(seg_means)
        # Fill unseen segments with global mean
        dfs[name]["segment_target_enc"] = dfs[name]["segment_target_enc"].fillna(
            dfs["train"][TARGET_COLUMN].mean()
        )

    # Test with target-encoded segment replacing is_enterprise
    features_te = ["sessions_7d", "segment_target_enc"]

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=500, random_state=42, solver="lbfgs")),
        ]
    )
    pipe.fit(dfs["train"][features_te], dfs["train"][TARGET_COLUMN])

    results = {}
    for name in ["train", "val", "test"]:
        probs = pipe.predict_proba(dfs[name][features_te])[:, 1]
        auc = roc_auc_score(dfs[name][TARGET_COLUMN], probs)
        results[name] = {"auc": auc, "probs": probs}

    lo, hi = _bootstrap_ci(dfs["val"][TARGET_COLUMN], results["val"]["probs"])
    print(
        f"Target-encoded logreg: train={results['train']['auc']:.4f}, "
        f"val={results['val']['auc']:.4f}, test={results['test']['auc']:.4f}"
    )
    print(f"  Bootstrap 95% CI: [{lo:.3f}, {hi:.3f}]")
    print()
    return results


def test_naive_bayes():
    """Gaussian Naive Bayes on 2 signal features."""
    dfs, features, _ = _prepare_splits()

    # Scale for consistency
    scaler = StandardScaler()
    X_train = scaler.fit_transform(dfs["train"][features])
    X_val = scaler.transform(dfs["val"][features])
    X_test = scaler.transform(dfs["test"][features])

    nb = GaussianNB()
    nb.fit(X_train, dfs["train"][TARGET_COLUMN])

    results = {}
    for name, X in [("train", X_train), ("val", X_val), ("test", X_test)]:
        probs = nb.predict_proba(X)[:, 1]
        auc = roc_auc_score(dfs[name][TARGET_COLUMN], probs)
        results[name] = {"auc": auc, "probs": probs}

    lo, hi = _bootstrap_ci(dfs["val"][TARGET_COLUMN], results["val"]["probs"])
    print(
        f"Gaussian NB: train={results['train']['auc']:.4f}, "
        f"val={results['val']['auc']:.4f}, test={results['test']['auc']:.4f}"
    )
    print(f"  Bootstrap 95% CI: [{lo:.3f}, {hi:.3f}]")
    print()
    return results


def test_svm_rbf():
    """SVM with RBF kernel (nonlinear decision boundary)."""
    dfs, features, _ = _prepare_splits()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(dfs["train"][features])
    X_val = scaler.transform(dfs["val"][features])
    X_test = scaler.transform(dfs["test"][features])

    # Platt-calibrated SVM for probability estimates
    svm = CalibratedClassifierCV(
        SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42),
        cv=5,
        method="sigmoid",
    )
    svm.fit(X_train, dfs["train"][TARGET_COLUMN])

    results = {}
    for name, X in [("train", X_train), ("val", X_val), ("test", X_test)]:
        probs = svm.predict_proba(X)[:, 1]
        auc = roc_auc_score(dfs[name][TARGET_COLUMN], probs)
        brier = brier_score_loss(dfs[name][TARGET_COLUMN], probs)
        results[name] = {"auc": auc, "brier": brier, "probs": probs}

    lo, hi = _bootstrap_ci(dfs["val"][TARGET_COLUMN], results["val"]["probs"])
    print(
        f"SVM-RBF (calibrated): train={results['train']['auc']:.4f}, "
        f"val={results['val']['auc']:.4f}, test={results['test']['auc']:.4f}"
    )
    print(f"  Val Brier: {results['val']['brier']:.4f}")
    print(f"  Bootstrap 95% CI: [{lo:.3f}, {hi:.3f}]")
    print()
    return results


def test_svm_linear():
    """Linear SVM (different loss function than logreg — hinge vs log-loss)."""
    dfs, features, _ = _prepare_splits()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(dfs["train"][features])
    X_val = scaler.transform(dfs["val"][features])
    X_test = scaler.transform(dfs["test"][features])

    svm = CalibratedClassifierCV(
        SVC(kernel="linear", C=1.0, random_state=42),
        cv=5,
        method="sigmoid",
    )
    svm.fit(X_train, dfs["train"][TARGET_COLUMN])

    results = {}
    for name, X in [("train", X_train), ("val", X_val), ("test", X_test)]:
        probs = svm.predict_proba(X)[:, 1]
        auc = roc_auc_score(dfs[name][TARGET_COLUMN], probs)
        results[name] = {"auc": auc, "probs": probs}

    lo, hi = _bootstrap_ci(dfs["val"][TARGET_COLUMN], results["val"]["probs"])
    print(
        f"SVM-Linear (calibrated): train={results['train']['auc']:.4f}, "
        f"val={results['val']['auc']:.4f}, test={results['test']['auc']:.4f}"
    )
    print(f"  Bootstrap 95% CI: [{lo:.3f}, {hi:.3f}]")
    print()
    return results


def main():
    print("=" * 60)
    print("Cycle 0005: Roads Not Taken — Falsification Tests")
    print("=" * 60)
    print()
    print("Testing untried approaches against the ~0.60 AUC ceiling.")
    print("Reference: logreg-poly val AUC = 0.608, CI [0.519, 0.697]")
    print()

    te_results = test_target_encoding()
    nb_results = test_naive_bayes()
    svm_rbf_results = test_svm_rbf()
    svm_lin_results = test_svm_linear()

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print()
    print("| Model | Val AUC | Train AUC | Gap |")
    print("|---|---|---|---|")

    for label, r in [
        ("logreg-poly (ref)", {"train": {"auc": 0.6594}, "val": {"auc": 0.6080}}),
        ("Target-enc logreg", te_results),
        ("Gaussian NB", nb_results),
        ("SVM-RBF", svm_rbf_results),
        ("SVM-Linear", svm_lin_results),
    ]:
        t_auc = r["train"]["auc"]
        v_auc = r["val"]["auc"]
        gap = t_auc - v_auc
        print(f"| {label} | {v_auc:.4f} | {t_auc:.4f} | {gap:.4f} |")


if __name__ == "__main__":
    main()
