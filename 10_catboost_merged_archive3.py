import time
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from config_archive3 import ID_COLUMN, MODELS_DIR, OUTPUT_DIR, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE
from data_utils_archive3 import load_original_data
from new_roni2 import train_catboost_model


def create_shared_split(df: pd.DataFrame):

    train_ids, test_ids = train_test_split(
        df[ID_COLUMN], test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df[TARGET_COLUMN]
    )
    return set(train_ids), set(test_ids)


def main():
    full = load_original_data()
    path = OUTPUT_DIR / "merged_dataset.csv"
    if not path.exists():
        raise FileNotFoundError("Run 05_fusion_archive3.py first.")
    df = pd.read_csv(path)
    train_ids, test_ids = create_shared_split(full)

    feature_columns = [c for c in df.columns if c not in [ID_COLUMN, TARGET_COLUMN]]
    train_df = df[df[ID_COLUMN].isin(train_ids)]
    test_df = df[df[ID_COLUMN].isin(test_ids)]
    X_train, y_train = train_df[feature_columns], train_df[TARGET_COLUMN]
    X_test, y_test = test_df[feature_columns], test_df[TARGET_COLUMN]

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Dataset: Merged | features={len(feature_columns)} | train={len(X_train)} | test={len(X_test)} | scale_pos_weight={scale_pos_weight:.3f}")

    start = time.perf_counter()
    model = train_catboost_model(X_train, y_train, X_test, y_test, scale_pos_weight)
    elapsed = time.perf_counter() - start

    prob = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test).astype(int)
    cm = confusion_matrix(y_test, pred)

    row = {
        "Dataset": "Merged", "Model": "CatBoost", "Uses_PCA": False,
        "Original_Features": len(feature_columns), "Train_Rows_Used": len(X_train),
        "Best_Threshold": 0.5, "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, prob),
        "Training_Time_Seconds": elapsed,
        "TN": int(cm[0, 0]), "FP": int(cm[0, 1]), "FN": int(cm[1, 0]), "TP": int(cm[1, 1]),
    }
    print(f"\nF1={row['F1']:.4f} | AUC={row['ROC_AUC']:.4f} | Accuracy={row['Accuracy']:.4f} | time={elapsed:.1f}s")

    out_path = OUTPUT_DIR / "extra_models_results.csv"
    existing = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame(columns=["Model"])
    existing = existing[existing["Model"] != "CatBoost"]
    pd.concat([existing, pd.DataFrame([row])], ignore_index=True).to_csv(out_path, index=False, encoding="utf-8-sig")
    model.save_model(str(MODELS_DIR / "merged_catboost.cbm"))


if __name__ == "__main__":
    main()
