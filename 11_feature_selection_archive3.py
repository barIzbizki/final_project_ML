"""Forward-selection experiment on the Merged feature set.

Ranks features by XGBoost gain importance (the fully-trained tree model -- CatBoost
early-stopped at 42 iterations and zeroed out most features, so its importances are
unreliable here), then trains XGBoost on the top-k features for increasing k. The goal
is to find the smallest feature subset that retains most of the full-model F1/AUC.

Consistent with the rest of archive3: same shared 80/20 patient split, SMOTE on the
training folds only, decision threshold tuned on a held-out validation slice (never the
test set), test metrics measured on the identical 50,736-row test set.
"""
import json
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config_archive3 import ID_COLUMN, OUTPUT_DIR, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE
from data_utils_archive3 import load_original_data

# XGBoost gain ranking from merged_xgboost_no_pca (see 11 analysis); most→least important.
RANKED_FEATURES = [
    "Age", "HighBP", "GenHlth", "HighChol", "Sex", "PhysHlth", "Income", "BMI",
    "Smoker", "Education", "DiffWalk", "MentHlth", "Stroke", "HvyAlcoholConsump",
    "CholCheck", "Veggies", "Fruits", "PhysActivity", "Diabetes", "NoDocbcCost",
    "AnyHealthcare",
]
K_VALUES = [1, 2, 3, 4, 5, 7, 10, 14, 21]
VALIDATION_FRACTION = 0.15


def make_xgb():
    return XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.85,
        colsample_bytree=0.85, objective="binary:logistic", eval_metric="logloss",
        random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist", device="cuda",
    )


def best_f1_threshold(probs, y_true):
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-12)
    return float(thresholds[int(np.argmax(f1_scores[:-1]))])


def main():
    full = load_original_data()
    df = pd.read_csv(OUTPUT_DIR / "merged_dataset.csv")

    train_ids, test_ids = train_test_split(
        full[ID_COLUMN], test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=full[TARGET_COLUMN]
    )
    train_ids, test_ids = set(train_ids), set(test_ids)

    train_full = df[df[ID_COLUMN].isin(train_ids)].copy()
    test_df = df[df[ID_COLUMN].isin(test_ids)].copy()
    train_df, val_df = train_test_split(
        train_full, test_size=VALIDATION_FRACTION, random_state=RANDOM_STATE, stratify=train_full[TARGET_COLUMN]
    )
    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    rows = []
    for k in K_VALUES:
        feats = RANKED_FEATURES[:k]
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", make_xgb()),
        ])
        pipe.fit(train_df[feats], train_df[TARGET_COLUMN])

        val_prob = pipe.predict_proba(val_df[feats])[:, 1]
        thr = best_f1_threshold(val_prob, val_df[TARGET_COLUMN])

        test_prob = pipe.predict_proba(test_df[feats])[:, 1]
        pred = (test_prob >= thr).astype(int)
        row = {
            "k": k, "features": ",".join(feats), "threshold": round(thr, 4),
            "accuracy": round(accuracy_score(test_df[TARGET_COLUMN], pred), 4),
            "precision": round(precision_score(test_df[TARGET_COLUMN], pred, zero_division=0), 4),
            "recall": round(recall_score(test_df[TARGET_COLUMN], pred, zero_division=0), 4),
            "f1": round(f1_score(test_df[TARGET_COLUMN], pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(test_df[TARGET_COLUMN], test_prob), 4),
        }
        rows.append(row)
        print(f"k={k:2d} | added={feats[-1]:18s} | F1={row['f1']:.4f} | AUC={row['roc_auc']:.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "feature_selection_curve.csv", index=False, encoding="utf-8-sig")

    full_f1, full_auc = rows[-1]["f1"], rows[-1]["roc_auc"]
    print("\n=== retained fraction of full-model performance ===")
    for r in rows:
        print(f"k={r['k']:2d} | F1 {100*r['f1']/full_f1:5.1f}% | AUC {100*r['roc_auc']/full_auc:5.1f}%")

    json.dump(rows, open("/tmp/claude-1004/-home-dotand-roni/d6011e79-604b-40a2-b16d-ceaecc39cdde/scratchpad/featsel.json", "w"))


if __name__ == "__main__":
    main()
