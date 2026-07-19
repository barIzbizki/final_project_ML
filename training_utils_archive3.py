from __future__ import annotations

import time, joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler
from cuml.neighbors import KNeighborsClassifier as KNeighborsClassifierGPU
from cuml.svm import SVC as SVCGPU

from config_archive3 import ID_COLUMN, MODELS_DIR, PCA_VARIANCE_OPTIONS, RANDOM_STATE, SVM_MAX_TRAIN_ROWS, TARGET_COLUMN, TEST_SIZE
from models_gpu import create_models
from preprocessing import create_preprocessor


def subsample_for_svm(X_train: pd.DataFrame, y_train: pd.Series):
    # See SVM_MAX_TRAIN_ROWS in config_archive3.py for why only this model is subsampled.
    # The test set is untouched, so test metrics stay measured on the same rows as every
    # other model -- only the amount of training data the SVM sees differs.
    if len(X_train) <= SVM_MAX_TRAIN_ROWS:
        return X_train, y_train
    X_sub, _, y_sub, _ = train_test_split(
        X_train, y_train, train_size=SVM_MAX_TRAIN_ROWS, random_state=RANDOM_STATE, stratify=y_train
    )
    return X_sub, y_sub


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
    if isinstance(model, KNeighborsClassifierGPU):
        param_grid["model__n_neighbors"] = [5, 7, 9, 11, 15, 21]
        # cuML's KNN ignores `p` (swallowed by **kwargs, always Euclidean); `metric` is
        # the parameter it actually honours. See models_gpu.py.
        param_grid["model__metric"] = ["manhattan", "euclidean"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    # n_jobs capped (not -1): this dataset has 253k rows, and SMOTE-oversampled
    # copies held by too many parallel workers exhausted memory (silent OOM kill).
    search = GridSearchCV(pipeline, param_grid=param_grid, scoring="f1", cv=cv, n_jobs=4, refit=True)
    search.fit(X_train, y_train)
    return search


def find_best_threshold(estimator, X_train: pd.DataFrame, y_train: pd.Series, cv) -> float:
    # Out-of-fold probabilities on training data only: the threshold is never
    # fit against X_test/y_test, so choosing it here cannot leak into the test score.
    oof_probs = cross_val_predict(estimator, X_train, y_train, cv=cv, method="predict_proba", n_jobs=4)[:, 1]
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
        X_fit, y_fit = (subsample_for_svm(X_train, y_train) if isinstance(model, SVCGPU) else (X_train, y_train))
        if len(X_fit) < len(X_train):
            print(f"{model_name}: training on a stratified subsample of {len(X_fit)} / {len(X_train)} rows")
        print(f"Training {model_name} with StandardScaler{' + PCA' if use_pca else ''} (GridSearchCV)...")
        start = time.perf_counter()
        search = fit_best_pipeline(X_fit, y_fit, model, use_pca=use_pca)
        pipeline = search.best_estimator_
        threshold_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        best_threshold = find_best_threshold(pipeline, X_fit, y_fit, threshold_cv)
        elapsed = time.perf_counter() - start
        prob = pipeline.predict_proba(X_test)[:,1]
        pred = (prob >= best_threshold).astype(int)
        cm = confusion_matrix(y_test, pred)
        fpr, tpr, _ = roc_curve(y_test, prob)

        row = {
            'Dataset': dataset_name, 'Model': model_name, 'Uses_PCA': use_pca,
            'Original_Features': len(feature_columns),
            'Train_Rows_Used': len(X_fit),
            'CV_Best_F1': float(search.best_score_),
            'Best_Threshold': best_threshold,
            'Model_N_Neighbors': search.best_params_.get('model__n_neighbors'),
            'Model_Distance': {'manhattan': 'Manhattan', 'euclidean': 'Euclidean'}.get(search.best_params_.get('model__metric')),
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
