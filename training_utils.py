from __future__ import annotations

import time, joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from config import ID_COLUMN, MODELS_DIR, PCA_VARIANCE_OPTIONS, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE
from models import create_models
from preprocessing import create_preprocessor


def create_shared_split(df: pd.DataFrame):
    train_ids, test_ids = train_test_split(
        df[ID_COLUMN], test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df[TARGET_COLUMN]
    )
    return set(train_ids), set(test_ids)


def create_model_pipeline(X_train: pd.DataFrame, model: object, use_pca: bool = True) -> Pipeline:
    steps = [
        ("preprocessing", create_preprocessor(X_train)),
        ("scaler", StandardScaler()),
    ]
    if use_pca:
        steps.append(("pca", PCA(svd_solver="full", random_state=RANDOM_STATE)))
    # SMOTE only resamples during .fit() (training folds); it is a no-op during
    # .predict()/.score(), so validation and test data are never touched by it.
    steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
    steps.append(("model", model))
    return Pipeline(steps)


def fit_best_pipeline(X_train: pd.DataFrame, y_train: pd.Series, model: object, use_pca: bool = True) -> GridSearchCV:
    pipeline = create_model_pipeline(X_train, model, use_pca=use_pca)
    param_grid = {}
    if use_pca:
        param_grid["pca__n_components"] = PCA_VARIANCE_OPTIONS
    if isinstance(model, KNeighborsClassifier):
        param_grid["model__n_neighbors"] = [5, 7, 9, 11, 15, 21]
        param_grid["model__p"] = [1, 2]  # 1=Manhattan, 2=Euclidean
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(pipeline, param_grid=param_grid, scoring="f1", cv=cv, n_jobs=-1, refit=True)
    search.fit(X_train, y_train)
    return search


def find_best_threshold(estimator, X_train: pd.DataFrame, y_train: pd.Series, cv) -> float:
    
    oof_probs = cross_val_predict(estimator, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_train, oof_probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = int(np.argmax(f1_scores[:-1]))  # last precision/recall pair has no matching threshold
    return float(thresholds[best_idx])


def evaluate_dataset(dataset_name, df, train_ids, test_ids, use_pca: bool = True):
    train_df = df[df[ID_COLUMN].isin(train_ids)].copy()
    test_df = df[df[ID_COLUMN].isin(test_ids)].copy()
    feature_columns = [c for c in df.columns if c not in [ID_COLUMN, TARGET_COLUMN]]
    X_train, y_train = train_df[feature_columns], train_df[TARGET_COLUMN]
    X_test, y_test = test_df[feature_columns], test_df[TARGET_COLUMN]
    results, roc_data = [], {}
    tag = "PCA" if use_pca else "no PCA"
    print(f"\nDataset: {dataset_name} | features={len(feature_columns)} | train={len(X_train)} | test={len(X_test)} | mode={tag}")
    for model_name, model in create_models().items():
        print(f"Training {model_name} with StandardScaler{' + PCA' if use_pca else ''} (GridSearchCV)...")
        start = time.perf_counter()
        search = fit_best_pipeline(X_train, y_train, model, use_pca=use_pca)
        pipeline = search.best_estimator_
        threshold_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        best_threshold = find_best_threshold(pipeline, X_train, y_train, threshold_cv)
        elapsed = time.perf_counter() - start
        prob = pipeline.predict_proba(X_test)[:,1]
        pred = (prob >= best_threshold).astype(int)
        cm = confusion_matrix(y_test, pred)
        fpr, tpr, _ = roc_curve(y_test, prob)

        row = {
            'Dataset': dataset_name, 'Model': model_name, 'Uses_PCA': use_pca,
            'Original_Features': len(feature_columns),
            'CV_Best_F1': float(search.best_score_),
            'Best_Threshold': best_threshold,
            'Model_N_Neighbors': search.best_params_.get('model__n_neighbors'),
            'Model_Distance': {1: 'Manhattan', 2: 'Euclidean'}.get(search.best_params_.get('model__p')),
            'Accuracy': accuracy_score(y_test,pred),
            'Precision': precision_score(y_test,pred,zero_division=0),
            'Recall': recall_score(y_test,pred,zero_division=0),
            'F1': f1_score(y_test,pred,zero_division=0),
            'ROC_AUC': roc_auc_score(y_test,prob),
            'Training_Time_Seconds': elapsed,
            'TN': int(cm[0,0]), 'FP': int(cm[0,1]), 'FN': int(cm[1,0]), 'TP': int(cm[1,1]),
        }
        if use_pca:
            pca = pipeline.named_steps['pca']
            row['PCA_Variance_Target'] = search.best_params_['pca__n_components']
            row['PCA_Components'] = int(pca.n_components_)
            row['Explained_Variance'] = float(pca.explained_variance_ratio_.sum())
        else:
            row['PCA_Variance_Target'] = None
            row['PCA_Components'] = None
            row['Explained_Variance'] = None
        results.append(row)
        roc_data[model_name] = {'fpr': fpr, 'tpr': tpr, 'auc': row['ROC_AUC']}
        suffix = "pca" if use_pca else "no_pca"
        joblib.dump(pipeline, MODELS_DIR / f"{dataset_name.lower()}_{model_name.lower()}_{suffix}.joblib")
        print(f"PCA_Components={row['PCA_Components']} | Threshold={best_threshold:.3f} | CV_F1={row['CV_Best_F1']:.4f} | Test_F1={row['F1']:.4f} | AUC={row['ROC_AUC']:.4f}")
    return results, roc_data
