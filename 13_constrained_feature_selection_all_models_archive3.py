"""Constrained feature selection across ALL archive3 models (GPU where available).

Same experiment as 12_constrained_feature_selection (keep BOTH parts of the fusion --
top clinical + top lifestyle -- so every subset contains features from both groups),
but run for the full project model suite instead of XGBoost only:

    KNN      -> cuML (GPU), n_neighbors tuned on validation F1
    SVM      -> cuML RBF (GPU), trained on a stratified 40k subsample (see config)
    XGBoost  -> device="cuda" (GPU)
    MLP      -> sklearn (CPU; no GPU implementation exists, same as the rest of archive3)

Protocol matches the rest of archive3: shared 80/20 patient split, SMOTE on the training
folds only, decision threshold tuned on a held-out validation slice (never the test set),
test metrics on the identical held-out test set. Within-group ranking is XGBoost gain.
"""
import json
import time

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from cuml.neighbors import KNeighborsClassifier as KNeighborsClassifierGPU
from cuml.svm import SVC as SVCGPU
from xgboost import XGBClassifier

from config_archive3 import ID_COLUMN, OUTPUT_DIR, RANDOM_STATE, SVM_MAX_TRAIN_ROWS, TARGET_COLUMN, TEST_SIZE
from data_utils_archive3 import load_original_data

# within-group rankings by XGBoost gain (from merged_xgboost_no_pca), most -> least
CLINICAL_RANKED = [
    "Age", "HighBP", "GenHlth", "HighChol", "Sex", "PhysHlth", "Income", "BMI",
    "Education", "DiffWalk", "MentHlth", "Stroke", "CholCheck", "Diabetes",
]
LIFESTYLE_RANKED = [
    "Smoker", "HvyAlcoholConsump", "Veggies", "Fruits", "PhysActivity",
    "NoDocbcCost", "AnyHealthcare",
]

# (n_clinical, n_lifestyle) combinations to test -- every one keeps both parts
COMBOS = [
    (3, 1), (3, 2), (4, 2), (5, 2), (5, 3), (4, 3), (6, 2), (6, 3), (7, 3), (5, 5),
]
VALIDATION_FRACTION = 0.15
# KNN hyperparameter grid -- identical to fit_best_pipeline() in training_utils_archive3.py.
# cuML's KNN ignores `p` (always Euclidean unless overridden), so distance choice goes
# through `metric`, exactly as the rest of the project does.
KNN_PARAM_GRID = {
    "model__n_neighbors": [5, 7, 9, 11, 15, 21],
    "model__metric": ["manhattan", "euclidean"],
}


def best_f1_threshold(probs, y_true):
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return float(thresholds[int(np.argmax(f1[:-1]))])


def make_pipeline(model):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("model", model),
    ])


def make_xgb():
    return XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.85,
        colsample_bytree=0.85, objective="binary:logistic", eval_metric="logloss",
        random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist", device="cuda",
    )


def make_svm():
    return SVCGPU(kernel="rbf", C=1.0, gamma="scale", probability=True,
                  class_weight="balanced", random_state=RANDOM_STATE)


def make_mlp():
    return MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
                         alpha=0.0001, learning_rate_init=0.001, max_iter=500,
                         early_stopping=True, validation_fraction=0.15, random_state=RANDOM_STATE)


def metrics(probs, y_true, threshold):
    pred = (probs >= threshold).astype(int)
    return {
        "threshold": round(threshold, 4),
        "accuracy": round(accuracy_score(y_true, pred), 4),
        "precision": round(precision_score(y_true, pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, probs), 4),
    }


def eval_single(model_factory, feats, tr, va, te, y_tr, y_va, y_te, subsample=False):
    """Fit one model, tune the decision threshold on validation, score on test."""
    Xtr, ytr = tr[feats], y_tr
    if subsample and len(Xtr) > SVM_MAX_TRAIN_ROWS:
        Xtr, _, ytr, _ = train_test_split(
            Xtr, ytr, train_size=SVM_MAX_TRAIN_ROWS, random_state=RANDOM_STATE, stratify=ytr)
    pipe = make_pipeline(model_factory())
    pipe.fit(Xtr, ytr)
    vp = np.asarray(pipe.predict_proba(va[feats]))[:, 1]
    thr = best_f1_threshold(vp, y_va)
    tp = np.asarray(pipe.predict_proba(te[feats]))[:, 1]
    row = metrics(tp, y_te, thr)
    row["train_rows"] = int(len(Xtr))
    row["n_neighbors"] = None
    row["knn_metric"] = None
    return row


def eval_knn(feats, tr, va, te, y_tr, y_va, y_te):
    """cuML KNN: pick n_neighbors + metric with GridSearchCV (5-fold, F1) -- same as the
    rest of archive3 (training_utils_archive3.fit_best_pipeline). The decision threshold is
    then tuned on the held-out validation slice, matching the other models in this script."""
    pipe = make_pipeline(KNeighborsClassifierGPU(weights="distance"))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(pipe, KNN_PARAM_GRID, scoring="f1", cv=cv, n_jobs=4, refit=True)
    search.fit(tr[feats], y_tr)
    best = search.best_estimator_
    vp = np.asarray(best.predict_proba(va[feats]))[:, 1]
    thr = best_f1_threshold(vp, y_va)
    tp = np.asarray(best.predict_proba(te[feats]))[:, 1]
    row = metrics(tp, y_te, thr)
    row["train_rows"] = int(len(tr))
    row["n_neighbors"] = search.best_params_["model__n_neighbors"]
    row["knn_metric"] = search.best_params_["model__metric"]
    return row


MODELS = {
    "KNN": lambda feats, *a: eval_knn(feats, *a),
    "SVM": lambda feats, *a: eval_single(make_svm, feats, *a, subsample=True),
    "XGBoost": lambda feats, *a: eval_single(make_xgb, feats, *a),
    "MLP": lambda feats, *a: eval_single(make_mlp, feats, *a),
}


def main():
    full = load_original_data()
    df = pd.read_csv(OUTPUT_DIR / "merged_dataset.csv")
    train_ids, test_ids = train_test_split(
        full[ID_COLUMN], test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=full[TARGET_COLUMN])
    train_ids, test_ids = set(train_ids), set(test_ids)
    train_full = df[df[ID_COLUMN].isin(train_ids)].copy()
    test_df = df[df[ID_COLUMN].isin(test_ids)].copy()
    train_df, val_df = train_test_split(
        train_full, test_size=VALIDATION_FRACTION, random_state=RANDOM_STATE, stratify=train_full[TARGET_COLUMN])
    y_tr, y_va, y_te = train_df[TARGET_COLUMN], val_df[TARGET_COLUMN], test_df[TARGET_COLUMN]
    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    # (label, clinical_feats, lifestyle_feats): reference all-21 first, then the combos
    subsets = [("all-21 (both parts)", 14, 7, CLINICAL_RANKED, LIFESTYLE_RANKED)]
    for c, l in COMBOS:
        subsets.append((f"{c}C+{l}L", c, l, CLINICAL_RANKED[:c], LIFESTYLE_RANKED[:l]))

    args = (train_df, val_df, test_df, y_tr, y_va, y_te)
    rows = []
    for label, nc, nl, cf, lf in subsets:
        feats = cf + lf
        for model_name, fn in MODELS.items():
            t0 = time.perf_counter()
            m = fn(feats, *args)
            dt = time.perf_counter() - t0
            rows.append({
                "combo": label, "model": model_name, "n_clinical": nc, "n_lifestyle": nl,
                "k": len(feats), "clinical": ",".join(cf), "lifestyle": ",".join(lf),
                "train_seconds": round(dt, 1), **m,
            })
            kn = f" K={m['n_neighbors']}" if m["n_neighbors"] else ""
            print(f"{label:20s} {model_name:8s} (k={len(feats):2d}) | "
                  f"F1={m['f1']:.4f} AUC={m['roc_auc']:.4f}{kn} | {dt:5.1f}s")

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "constrained_feature_selection_all_models.csv", index=False, encoding="utf-8-sig")
    json.dump(rows, open(OUTPUT_DIR / "constrained_feature_selection_all_models.json", "w"))

    # per-model: retained fraction of the all-21 F1
    print("\n=== retained F1 vs all-21, per model ===")
    for model_name in MODELS:
        mrows = [r for r in rows if r["model"] == model_name]
        full_f1 = mrows[0]["f1"]  # all-21 reference is first
        best = max(mrows[1:], key=lambda r: r["f1"])
        print(f"\n{model_name}: all-21 F1={full_f1:.4f}")
        for r in mrows[1:]:
            star = " <-- best trimmed" if r is best else ""
            print(f"  {r['combo']:8s} k={r['k']:2d} | F1={r['f1']:.4f} "
                  f"({100*r['f1']/full_f1:5.1f}%) AUC={r['roc_auc']:.4f}{star}")


if __name__ == "__main__":
    main()
