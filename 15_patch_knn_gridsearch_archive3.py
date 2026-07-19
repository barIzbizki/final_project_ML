"""Recompute ONLY the KNN rows of constrained_feature_selection_all_models with
GridSearchCV (5-fold, F1) over n_neighbors + metric -- identical to the rest of archive3
(training_utils_archive3.fit_best_pipeline) -- and patch them into the existing results.

SVM / XGBoost / MLP are deterministic and unchanged, so their rows are kept as-is; only
KNN switched from a manual validation loop to GridSearchCV model selection. This avoids
re-running the slow CPU MLP fits. Produces the same table 13_..._all_models.py would now
produce for KNN. Run 13 end-to-end instead if you want everything regenerated from scratch.
"""
import json

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from cuml.neighbors import KNeighborsClassifier as KNeighborsClassifierGPU

from config_archive3 import ID_COLUMN, OUTPUT_DIR, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE
from data_utils_archive3 import load_original_data

CLINICAL_RANKED = [
    "Age", "HighBP", "GenHlth", "HighChol", "Sex", "PhysHlth", "Income", "BMI",
    "Education", "DiffWalk", "MentHlth", "Stroke", "CholCheck", "Diabetes",
]
LIFESTYLE_RANKED = [
    "Smoker", "HvyAlcoholConsump", "Veggies", "Fruits", "PhysActivity",
    "NoDocbcCost", "AnyHealthcare",
]
COMBOS = [(3, 1), (3, 2), (4, 2), (5, 2), (5, 3), (4, 3), (6, 2), (6, 3), (7, 3), (5, 5)]
VALIDATION_FRACTION = 0.15
KNN_PARAM_GRID = {
    "model__n_neighbors": [5, 7, 9, 11, 15, 21],
    "model__metric": ["manhattan", "euclidean"],
}


def best_f1_threshold(probs, y_true):
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return float(thresholds[int(np.argmax(f1[:-1]))])


def eval_knn(feats, tr, va, te, y_tr, y_va, y_te):
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("smote", SMOTE(random_state=RANDOM_STATE)),
                     ("model", KNeighborsClassifierGPU(weights="distance"))])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(pipe, KNN_PARAM_GRID, scoring="f1", cv=cv, n_jobs=4, refit=True)
    search.fit(tr[feats], y_tr)
    best = search.best_estimator_
    vp = np.asarray(best.predict_proba(va[feats]))[:, 1]
    thr = best_f1_threshold(vp, y_va)
    tp = np.asarray(best.predict_proba(te[feats]))[:, 1]
    pred = (tp >= thr).astype(int)
    return {
        "threshold": round(thr, 4),
        "accuracy": round(accuracy_score(y_te, pred), 4),
        "precision": round(precision_score(y_te, pred, zero_division=0), 4),
        "recall": round(recall_score(y_te, pred, zero_division=0), 4),
        "f1": round(f1_score(y_te, pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_te, tp), 4),
        "train_rows": int(len(tr)),
        "n_neighbors": int(search.best_params_["model__n_neighbors"]),
        "knn_metric": search.best_params_["model__metric"],
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
    args = (train_df, val_df, test_df, y_tr, y_va, y_te)
    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    subsets = [("all-21 (both parts)", 14, 7, CLINICAL_RANKED, LIFESTYLE_RANKED)]
    for c, l in COMBOS:
        subsets.append((f"{c}C+{l}L", c, l, CLINICAL_RANKED[:c], LIFESTYLE_RANKED[:l]))

    new_knn = []
    for label, nc, nl, cf, lf in subsets:
        feats = cf + lf
        m = eval_knn(feats, *args)
        new_knn.append({
            "combo": label, "model": "KNN", "n_clinical": nc, "n_lifestyle": nl,
            "k": len(feats), "clinical": ",".join(cf), "lifestyle": ",".join(lf),
            "train_seconds": None, **m,
        })
        print(f"{label:20s} KNN (k={len(feats):2d}) | F1={m['f1']:.4f} AUC={m['roc_auc']:.4f} "
              f"| best K={m['n_neighbors']} metric={m['knn_metric']}")

    # patch: drop old KNN rows, append the GridSearchCV ones, keep original combo order
    csv_path = OUTPUT_DIR / "constrained_feature_selection_all_models.csv"
    old = pd.read_csv(csv_path)
    if "knn_metric" not in old.columns:
        old["knn_metric"] = np.nan
    kept = old[old["model"] != "KNN"]
    combined = pd.concat([kept, pd.DataFrame(new_knn)], ignore_index=True)
    order = {label: i for i, (label, *_rest) in enumerate(subsets)}
    model_order = {"KNN": 0, "SVM": 1, "XGBoost": 2, "MLP": 3}
    combined["_c"] = combined["combo"].map(order)
    combined["_m"] = combined["model"].map(model_order)
    combined = combined.sort_values(["_c", "_m"]).drop(columns=["_c", "_m"]).reset_index(drop=True)
    combined.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json.dump(combined.to_dict("records"), open(OUTPUT_DIR / "constrained_feature_selection_all_models.json", "w"))
    print(f"\npatched {csv_path} ({len(new_knn)} KNN rows replaced via GridSearchCV)")

    full_f1 = combined[(combined.model == "KNN") & (combined.combo == "all-21 (both parts)")]["f1"].iloc[0]
    print(f"\n=== KNN retained F1 vs all-21 ({full_f1:.4f}) ===")
    for r in new_knn[1:]:
        print(f"  {r['combo']:8s} k={r['k']:2d} | F1={r['f1']:.4f} ({100*r['f1']/full_f1:5.1f}%) "
              f"AUC={r['roc_auc']:.4f} | K={r['n_neighbors']} {r['knn_metric']}")


if __name__ == "__main__":
    main()
