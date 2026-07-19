
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


CLINICAL_RANKED = [
    "Age", "HighBP", "GenHlth", "HighChol", "Sex", "PhysHlth", "Income", "BMI",
    "Education", "DiffWalk", "MentHlth", "Stroke", "CholCheck", "Diabetes",
]
LIFESTYLE_RANKED = [
    "Smoker", "HvyAlcoholConsump", "Veggies", "Fruits", "PhysActivity",
    "NoDocbcCost", "AnyHealthcare",
]

# (n_clinical, n_lifestyle) 
COMBOS = [
    (3, 1), (3, 2), (4, 2), (5, 2), (5, 3), (4, 3), (6, 2), (6, 3), (7, 3), (5, 5),
]
VALIDATION_FRACTION = 0.15


def make_xgb():
    return XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.85,
        colsample_bytree=0.85, objective="binary:logistic", eval_metric="logloss",
        random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist", device="cuda",
    )


def best_f1_threshold(probs, y_true):
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return float(thresholds[int(np.argmax(f1[:-1]))])


def evaluate(feats, train_df, val_df, test_df):
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
    return {
        "threshold": round(thr, 4),
        "accuracy": round(accuracy_score(test_df[TARGET_COLUMN], pred), 4),
        "precision": round(precision_score(test_df[TARGET_COLUMN], pred, zero_division=0), 4),
        "recall": round(recall_score(test_df[TARGET_COLUMN], pred, zero_division=0), 4),
        "f1": round(f1_score(test_df[TARGET_COLUMN], pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(test_df[TARGET_COLUMN], test_prob), 4),
    }


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
    
    for name, feats in [("all-21 (both parts)", CLINICAL_RANKED + LIFESTYLE_RANKED)]:
        m = evaluate(feats, train_df, val_df, test_df)
        rows.append({"combo": name, "n_clinical": 14, "n_lifestyle": 7, "k": 21,
                     "clinical": ",".join(CLINICAL_RANKED), "lifestyle": ",".join(LIFESTYLE_RANKED), **m})
        print(f"[ref] {name:22s} F1={m['f1']:.4f} AUC={m['roc_auc']:.4f}")

    for c, l in COMBOS:
        cf, lf = CLINICAL_RANKED[:c], LIFESTYLE_RANKED[:l]
        feats = cf + lf
        m = evaluate(feats, train_df, val_df, test_df)
        rows.append({"combo": f"{c}C+{l}L", "n_clinical": c, "n_lifestyle": l, "k": c + l,
                     "clinical": ",".join(cf), "lifestyle": ",".join(lf), **m})
        print(f"{c}C+{l}L (k={c+l:2d}) | clin={cf} + life={lf} | F1={m['f1']:.4f} AUC={m['roc_auc']:.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "constrained_feature_selection.csv", index=False, encoding="utf-8-sig")
    json.dump(rows, open("/tmp/claude-1004/-home-dotand-roni/d6011e79-604b-40a2-b16d-ceaecc39cdde/scratchpad/constrained_featsel.json", "w"))

    full_f1 = rows[0]["f1"]
    print("\n=== retained F1 vs all-21 ===")
    for r in rows[1:]:
        print(f"{r['combo']:8s} k={r['k']:2d} | F1={r['f1']:.4f} ({100*r['f1']/full_f1:5.1f}%) | AUC={r['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
