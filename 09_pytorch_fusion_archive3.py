
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, WeightedRandomSampler

from config_archive3 import CLINICAL_COLUMNS, ID_COLUMN, LIFESTYLE_COLUMNS, MODELS_DIR, OUTPUT_DIR, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE
from data_utils_archive3 import load_original_data
from new_roni import HeartDiseaseDataset, MultimodalLateFusionNN

BATCH_SIZE = 1024
MAX_EPOCHS = 100
PATIENCE = 10
VALIDATION_FRACTION = 0.15


def create_shared_split(df: pd.DataFrame):
    train_ids, test_ids = train_test_split(
        df[ID_COLUMN], test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df[TARGET_COLUMN]
    )
    return set(train_ids), set(test_ids)


def preprocess_branch(train_df, val_df, test_df, columns):
    imputer = SimpleImputer(strategy="median").fit(train_df[columns])
    scaler = StandardScaler().fit(imputer.transform(train_df[columns]))
    tr = pd.DataFrame(scaler.transform(imputer.transform(train_df[columns])), columns=columns, index=train_df.index)
    va = pd.DataFrame(scaler.transform(imputer.transform(val_df[columns])), columns=columns, index=val_df.index)
    te = pd.DataFrame(scaler.transform(imputer.transform(test_df[columns])), columns=columns, index=test_df.index)
    return tr, va, te


def run_eval(model, loader, device):
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for xc, xl, y in loader:
            p = model(xc.to(device), xl.to(device)).cpu().numpy().ravel()
            probs.append(p); ys.append(y.numpy().ravel())
    return np.concatenate(probs), np.concatenate(ys)


def best_f1_threshold(probs, y_true):
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-12)
    return float(thresholds[int(np.argmax(f1_scores[:-1]))])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else " -- no GPU visible to torch"))

    full = load_original_data()
    path = OUTPUT_DIR / "merged_dataset.csv"
    if not path.exists():
        raise FileNotFoundError("Run 05_fusion_archive3.py first.")
    df = pd.read_csv(path)
    train_ids, test_ids = create_shared_split(full)

    train_full_df = df[df[ID_COLUMN].isin(train_ids)].copy()
    test_df = df[df[ID_COLUMN].isin(test_ids)].copy()
    train_df, val_df = train_test_split(
        train_full_df, test_size=VALIDATION_FRACTION, random_state=RANDOM_STATE, stratify=train_full_df[TARGET_COLUMN]
    )
    print(f"Dataset: Merged | train={len(train_df)} | val={len(val_df)} | test={len(test_df)}")

    clin_tr, clin_va, clin_te = preprocess_branch(train_df, val_df, test_df, CLINICAL_COLUMNS)
    life_tr, life_va, life_te = preprocess_branch(train_df, val_df, test_df, LIFESTYLE_COLUMNS)
    y_tr, y_va, y_te = train_df[TARGET_COLUMN], val_df[TARGET_COLUMN], test_df[TARGET_COLUMN]

    train_ds = HeartDiseaseDataset(clin_tr, life_tr, y_tr)
    val_ds = HeartDiseaseDataset(clin_va, life_va, y_va)
    test_ds = HeartDiseaseDataset(clin_te, life_te, y_te)

    # WeightedRandomSampler stands in for the SMOTE oversampling used elsewhere in the
    # pipeline: positives are drawn roughly as often as negatives per epoch, without
    # touching the model or loss function defined in new_roni.py.
    class_counts = y_tr.value_counts()
    sample_weights = y_tr.map(lambda c: 1.0 / class_counts[c]).values.copy()
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False)

    model = MultimodalLateFusionNN(clinical_dim=len(CLINICAL_COLUMNS), lifestyle_dim=len(LIFESTYLE_COLUMNS)).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_f1, best_state, epochs_no_improve = -1.0, None, 0
    start = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xc, xl, y in train_loader:
            xc, xl, y = xc.to(device), xl.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xc, xl), y)
            loss.backward()
            optimizer.step()

        val_probs, val_y = run_eval(model, val_loader, device)
        val_f1 = f1_score(val_y, (val_probs >= 0.5).astype(int), zero_division=0)
        if val_f1 > best_val_f1:
            best_val_f1, best_state, epochs_no_improve = val_f1, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            epochs_no_improve += 1
        if epoch == 1 or epoch % 5 == 0:
            print(f"epoch {epoch:3d} | val F1@0.5={val_f1:.4f} | best={best_val_f1:.4f}")
        if epochs_no_improve >= PATIENCE:
            print(f"early stop at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

    model.load_state_dict(best_state)
    elapsed = time.perf_counter() - start

    val_probs, val_y = run_eval(model, val_loader, device)
    threshold = best_f1_threshold(val_probs, val_y)

    test_probs, test_y = run_eval(model, test_loader, device)
    pred = (test_probs >= threshold).astype(int)
    cm = confusion_matrix(test_y, pred)

    row = {
        "Dataset": "Merged", "Model": "PyTorch-LateFusion", "Uses_PCA": False,
        "Original_Features": len(CLINICAL_COLUMNS) + len(LIFESTYLE_COLUMNS), "Train_Rows_Used": len(train_df),
        "Best_Threshold": threshold, "Accuracy": accuracy_score(test_y, pred),
        "Precision": precision_score(test_y, pred, zero_division=0),
        "Recall": recall_score(test_y, pred, zero_division=0),
        "F1": f1_score(test_y, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(test_y, test_probs),
        "Training_Time_Seconds": elapsed,
        "TN": int(cm[0, 0]), "FP": int(cm[0, 1]), "FN": int(cm[1, 0]), "TP": int(cm[1, 1]),
    }
    print(f"\nThreshold={threshold:.3f} | F1={row['F1']:.4f} | AUC={row['ROC_AUC']:.4f} | Accuracy={row['Accuracy']:.4f} | time={elapsed:.1f}s")

    out_path = OUTPUT_DIR / "extra_models_results.csv"
    existing = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame(columns=["Model"])
    existing = existing[existing["Model"] != "PyTorch-LateFusion"]
    pd.concat([existing, pd.DataFrame([row])], ignore_index=True).to_csv(out_path, index=False, encoding="utf-8-sig")
    torch.save(best_state, MODELS_DIR / "merged_pytorch_latefusion.pt")


if __name__ == "__main__":
    main()
