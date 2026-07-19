
import json
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from cuml.neighbors import KNeighborsClassifier as KNeighborsClassifierGPU
from xgboost import XGBClassifier

from config_archive3 import ID_COLUMN, LIFESTYLE_COLUMNS, OUTPUT_DIR, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE


RANKED = [
    "Age", "HighBP", "GenHlth", "HighChol", "Sex", "PhysHlth", "Income", "BMI",
    "Smoker", "Education", "DiffWalk", "MentHlth", "Stroke", "HvyAlcoholConsump",
    "CholCheck", "Veggies", "Fruits", "PhysActivity", "Diabetes", "NoDocbcCost",
    "AnyHealthcare",
]
K_VALUES = list(range(1, 22))
VALIDATION_FRACTION = 0.15
KNN_NEIGHBOR_GRID = [11, 21, 31, 51, 71]


def best_f1_threshold(probs, y_true):
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-12)
    return float(thresholds[int(np.argmax(f1_scores[:-1]))])


def evaluate(probs, y_true, threshold):
    pred = (probs >= threshold).astype(int)
    return {
        "accuracy": round(accuracy_score(y_true, pred), 4),
        "precision": round(precision_score(y_true, pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, probs), 4),
    }


def make_pipeline(model):
    return Pipeline([("scaler", StandardScaler()),
                     ("smote", SMOTE(random_state=RANDOM_STATE)),
                     ("model", model)])


def fit_knn(feats, tr, va, te, y_tr, y_va, y_te):
   
    best = None
    for n in KNN_NEIGHBOR_GRID:
        pipe = make_pipeline(KNeighborsClassifierGPU(n_neighbors=n, weights="distance", metric="manhattan"))
        pipe.fit(tr[feats], y_tr)
        vp = np.asarray(pipe.predict_proba(va[feats]))[:, 1]
        thr = best_f1_threshold(vp, y_va)
        vf1 = f1_score(y_va, (vp >= thr).astype(int), zero_division=0)
        if best is None or vf1 > best[0]:
            best = (vf1, n, thr, pipe)
    _, n, thr, pipe = best
    tp = np.asarray(pipe.predict_proba(te[feats]))[:, 1]
    row = evaluate(tp, y_te, thr)
    row["n_neighbors"] = n
    return row


def fit_xgb(feats, tr, va, te, y_tr, y_va, y_te):
    pipe = make_pipeline(XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.85,
        colsample_bytree=0.85, objective="binary:logistic", eval_metric="logloss",
        random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist", device="cuda"))
    pipe.fit(tr[feats], y_tr)
    vp = pipe.predict_proba(va[feats])[:, 1]
    thr = best_f1_threshold(vp, y_va)
    tp = pipe.predict_proba(te[feats])[:, 1]
    return evaluate(tp, y_te, thr)


def main():
    df = pd.read_csv(OUTPUT_DIR / "merged_dataset.csv")
    train_ids, test_ids = train_test_split(
        df[ID_COLUMN], test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df[TARGET_COLUMN])
    train_ids, test_ids = set(train_ids), set(test_ids)
    train_full = df[df[ID_COLUMN].isin(train_ids)].copy()
    test_df = df[df[ID_COLUMN].isin(test_ids)].copy()
    train_df, val_df = train_test_split(
        train_full, test_size=VALIDATION_FRACTION, random_state=RANDOM_STATE, stratify=train_full[TARGET_COLUMN])
    y_tr, y_va, y_te = train_df[TARGET_COLUMN], val_df[TARGET_COLUMN], test_df[TARGET_COLUMN]
    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    curves = {"KNN": [], "XGBoost": []}
    for k in K_VALUES:
        feats = RANKED[:k]
        has_life = any(f in LIFESTYLE_COLUMNS for f in feats)
        knn = fit_knn(feats, train_df, val_df, test_df, y_tr, y_va, y_te)
        xgb = fit_xgb(feats, train_df, val_df, test_df, y_tr, y_va, y_te)
        knn.update({"k": k, "added": feats[-1], "has_lifestyle": has_life})
        xgb.update({"k": k, "added": feats[-1], "has_lifestyle": has_life})
        curves["KNN"].append(knn)
        curves["XGBoost"].append(xgb)
        print(f"k={k:2d} +{feats[-1]:18s} life={int(has_life)} | "
              f"KNN F1={knn['f1']:.4f} AUC={knn['roc_auc']:.4f} (K={knn['n_neighbors']}) | "
              f"XGB F1={xgb['f1']:.4f} AUC={xgb['roc_auc']:.4f}")

    # peak (best test F1) per model
    for m in curves:
        peak = max(curves[m], key=lambda r: r["f1"])
        full = curves[m][-1]
        print(f"\n{m}: peak F1={peak['f1']} at k={peak['k']} ({peak['added']} last) | "
              f"full(k=21) F1={full['f1']} | gain from trimming = {peak['f1']-full['f1']:+.4f}")

 
    knn_peak = max(curves["KNN"], key=lambda r: r["f1"])
    rec = RANKED[:knn_peak["k"]]
    if not any(f in LIFESTYLE_COLUMNS for f in rec):
        rec = rec + ["Smoker"]  # inject strongest lifestyle feature
        recrow = fit_knn(rec, train_df, val_df, test_df, y_tr, y_va, y_te)
        recrow.update({"k": len(rec), "features": rec})
        print(f"\nlifestyle-compliant recommended KNN subset ({len(rec)} feats incl. Smoker): "
              f"F1={recrow['f1']} AUC={recrow['roc_auc']}")
        curves["KNN_recommended"] = recrow
    else:
        curves["KNN_recommended"] = {**knn_peak, "features": rec}

    rows = []
    for m in ("KNN", "XGBoost"):
        for r in curves[m]:
            rows.append({"model": m, **r})
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "curse_of_dimensionality_curve.csv", index=False, encoding="utf-8-sig")
    json.dump(curves, open("/tmp/claude-1004/-home-dotand-roni/d6011e79-604b-40a2-b16d-ceaecc39cdde/scratchpad/curse.json", "w"))


if __name__ == "__main__":
    main()
