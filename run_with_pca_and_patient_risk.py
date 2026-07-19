from __future__ import annotations

import time

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    CLINICAL_COLUMNS,
    ID_COLUMN,
    LIFESTYLE_COLUMNS,
    MODELS_DIR,
    OUTPUT_DIR,
    PCA_VARIANCE,
    PLOTS_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from data_utils import load_original_data
from models import create_models
from preprocessing import create_preprocessor


# ============================================================
# יצירת תיקיות
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

PATIENT_PREDICTIONS_DIR = OUTPUT_DIR / "patient_predictions"
PATIENT_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# פונקציות עזר להדפסה אסתטית
# ============================================================

def print_title(title: str, width: int = 100) -> None:
    print("\n" + "═" * width)
    print(title.center(width))
    print("═" * width)


def print_subtitle(title: str, width: int = 100) -> None:
    print("\n" + "─" * width)
    print(title.center(width))
    print("─" * width)


def print_results_table(
    results: list[dict],
    dataset_name: str,
) -> None:
    table = pd.DataFrame(results)[
        [
            "Model",
            "Original_Features",
            "Features_After_Preprocessing",
            "PCA_Components",
            "Explained_Variance",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC_AUC",
            "Training_Time_Seconds",
        ]
    ].copy()

    table = table.rename(
        columns={
            "Original_Features": "Original",
            "Features_After_Preprocessing": "After Prep",
            "PCA_Components": "PCA Comp.",
            "Explained_Variance": "VAR",
            "ROC_AUC": "ROC-AUC",
            "Training_Time_Seconds": "Time (sec)",
        }
    )

    for column in [
        "VAR",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
        "Time (sec)",
    ]:
        table[column] = table[column].map(
            lambda value: f"{value:.4f}"
        )

    best_f1_model = max(
        results,
        key=lambda row: row["F1"],
    )["Model"]

    best_auc_model = max(
        results,
        key=lambda row: row["ROC_AUC"],
    )["Model"]

    print_subtitle(f"{dataset_name} – Model Results")
    print(table.to_string(index=False))
    print()
    print(f"Best F1 model     : {best_f1_model}")
    print(f"Best ROC-AUC model: {best_auc_model}")


def print_final_summary(
    results_df: pd.DataFrame,
) -> None:
    summary = results_df[
        [
            "Dataset",
            "Model",
            "PCA_Components",
            "Explained_Variance",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC_AUC",
        ]
    ].copy()

    summary = summary.rename(
        columns={
            "PCA_Components": "PCA Comp.",
            "Explained_Variance": "VAR",
            "ROC_AUC": "ROC-AUC",
        }
    )

    for column in [
        "VAR",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
    ]:
        summary[column] = summary[column].map(
            lambda value: f"{value:.4f}"
        )

    print_title("FINAL RESULTS – WITH PCA")
    print(summary.to_string(index=False))

    best_overall = results_df.loc[
        results_df["F1"].idxmax()
    ]

    print("\n" + "─" * 100)
    print(
        "Best overall model: "
        f"{best_overall['Model']} on {best_overall['Dataset']} | "
        f"F1={best_overall['F1']:.4f} | "
        f"ROC-AUC={best_overall['ROC_AUC']:.4f} | "
        f"PCA Components={int(best_overall['PCA_Components'])} | "
        f"VAR={best_overall['Explained_Variance']:.4f}"
    )
    print("─" * 100)


# ============================================================
# רמות סיכון
# ============================================================

def get_risk_level(probability: float) -> str:
    """
    חלוקת הסתברות המחלה לרמות סיכון.

    שימי לב:
    אלו רמות תצוגה שהוגדרו בפרויקט, ולא קטגוריות רפואיות רשמיות.
    """

    if probability < 0.20:
        return "Low"
    if probability < 0.50:
        return "Medium"
    if probability < 0.80:
        return "High"
    return "Very High"


def save_patient_predictions(
    dataset_name: str,
    model_name: str,
    test_df: pd.DataFrame,
    y_test: pd.Series,
    predictions,
    probabilities,
) -> Path:
    """
    שמירת תחזית מפורטת לכל מטופל בסט הבדיקה.
    """

    patient_results = pd.DataFrame({
        ID_COLUMN: test_df[ID_COLUMN].values,
        "Actual_Label": y_test.values,
        "Actual_Status": [
            "Heart Disease" if value == 1 else "No Heart Disease"
            for value in y_test.values
        ],
        "Predicted_Label": predictions,
        "Predicted_Status": [
            "Heart Disease" if value == 1 else "No Heart Disease"
            for value in predictions
        ],
        "Risk_Probability": probabilities,
        "Risk_Percentage": probabilities * 100,
        "Risk_Level": [
            get_risk_level(probability)
            for probability in probabilities
        ],
        "Correct_Prediction": (
            predictions == y_test.values
        ),
    })

    patient_results = patient_results.sort_values(
        by="Risk_Probability",
        ascending=False,
    ).reset_index(drop=True)

    output_path = (
        PATIENT_PREDICTIONS_DIR
        / f"{dataset_name.lower()}_{model_name.lower()}_patient_risk.csv"
    )

    patient_results.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    return output_path


# ============================================================
# Pipeline עם PCA
# ============================================================

def create_pipeline_with_pca(
    X_train: pd.DataFrame,
    model: object,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessing", create_preprocessor(X_train)),
            ("scaler", StandardScaler()),
            (
                "pca",
                PCA(
                    n_components=PCA_VARIANCE,
                    svd_solver="full",
                    random_state=RANDOM_STATE,
                ),
            ),
            ("model", model),
        ]
    )


# ============================================================
# חלוקת Train/Test משותפת
# ============================================================

def create_shared_split(
    df: pd.DataFrame,
) -> tuple[set[int], set[int]]:
    train_ids, test_ids = train_test_split(
        df[ID_COLUMN],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN],
    )

    return set(train_ids), set(test_ids)


# ============================================================
# אימון והערכת המודלים עם PCA
# ============================================================

def evaluate_dataset_with_pca(
    dataset_name: str,
    df: pd.DataFrame,
    train_ids: set[int],
    test_ids: set[int],
) -> tuple[list[dict], list[dict]]:
    train_df = df[
        df[ID_COLUMN].isin(train_ids)
    ].copy()

    test_df = df[
        df[ID_COLUMN].isin(test_ids)
    ].copy()

    feature_columns = [
        column
        for column in df.columns
        if column not in [ID_COLUMN, TARGET_COLUMN]
    ]

    X_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df[feature_columns]
    y_test = test_df[TARGET_COLUMN]

    results = []
    roc_rows = []

    print_title(f"DATASET: {dataset_name.upper()} – WITH PCA")

    print(f"Configuration            : StandardScaler + PCA + Model")
    print(f"Original features        : {len(feature_columns)}")
    print(f"Training rows            : {len(X_train)}")
    print(f"Testing rows             : {len(X_test)}")
    print(f"Requested PCA variance   : {PCA_VARIANCE:.2%}")

    for model_name, model in create_models().items():
        print(f"\n▶ Training {model_name}...")

        pipeline = create_pipeline_with_pca(
            X_train=X_train,
            model=model,
        )

        start_time = time.perf_counter()

        pipeline.fit(
            X_train,
            y_train,
        )

        training_time = (
            time.perf_counter()
            - start_time
        )

        predictions = pipeline.predict(X_test)

        probabilities = (
            pipeline.predict_proba(X_test)[:, 1]
        )

        # שמירת קובץ סיכון לכל מטופל
        patient_predictions_path = save_patient_predictions(
            dataset_name=dataset_name,
            model_name=model_name,
            test_df=test_df,
            y_test=y_test,
            predictions=predictions,
            probabilities=probabilities,
        )

        matrix = confusion_matrix(
            y_test,
            predictions,
        )

        fpr, tpr, _ = roc_curve(
            y_test,
            probabilities,
        )

        processed_train = (
            pipeline.named_steps[
                "preprocessing"
            ].transform(X_train)
        )

        processed_feature_count = (
            processed_train.shape[1]
        )

        pca = pipeline.named_steps["pca"]

        pca_components = int(
            pca.n_components_
        )

        explained_variance = float(
            pca.explained_variance_ratio_.sum()
        )

        row = {
            "Dataset": dataset_name,
            "Model": model_name,
            "Uses_PCA": True,
            "Original_Features": len(feature_columns),
            "Features_After_Preprocessing": processed_feature_count,
            "PCA_Components": pca_components,
            "Explained_Variance": explained_variance,
            "Accuracy": accuracy_score(
                y_test,
                predictions,
            ),
            "Precision": precision_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            "Recall": recall_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            "F1": f1_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            "ROC_AUC": roc_auc_score(
                y_test,
                probabilities,
            ),
            "Training_Time_Seconds": training_time,
            "TN": int(matrix[0, 0]),
            "FP": int(matrix[0, 1]),
            "FN": int(matrix[1, 0]),
            "TP": int(matrix[1, 1]),
            "Patient_Predictions_File": str(
                patient_predictions_path
            ),
        }

        results.append(row)

        for current_fpr, current_tpr in zip(
            fpr,
            tpr,
        ):
            roc_rows.append(
                {
                    "Dataset": dataset_name,
                    "Model": model_name,
                    "FPR": current_fpr,
                    "TPR": current_tpr,
                    "AUC": row["ROC_AUC"],
                }
            )

        model_path = (
            MODELS_DIR
            / (
                f"{dataset_name.lower()}_"
                f"{model_name.lower()}_with_pca.joblib"
            )
        )

        joblib.dump(
            pipeline,
            model_path,
        )

        print(
            f"  Done | "
            f"PCA={pca_components} | "
            f"VAR={explained_variance:.4f} | "
            f"F1={row['F1']:.4f} | "
            f"ROC-AUC={row['ROC_AUC']:.4f} | "
            f"Time={training_time:.2f}s"
        )

        print(
            f"  Patient risk file: "
            f"{patient_predictions_path.name}"
        )

    print_results_table(
        results,
        dataset_name,
    )

    return results, roc_rows


# ============================================================
# יצירת שלושת המאגרים
# ============================================================

def create_experiment_datasets(
    full_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    clinical_df = full_df[
        [ID_COLUMN]
        + CLINICAL_COLUMNS
        + [TARGET_COLUMN]
    ].copy()

    lifestyle_df = full_df[
        [ID_COLUMN]
        + LIFESTYLE_COLUMNS
        + [TARGET_COLUMN]
    ].copy()

    merged_df = clinical_df.merge(
        lifestyle_df.drop(
            columns=[TARGET_COLUMN]
        ),
        on=ID_COLUMN,
        how="inner",
        validate="one_to_one",
    )

    return {
        "Clinical": clinical_df,
        "Lifestyle": lifestyle_df,
        "Merged": merged_df,
    }


# ============================================================
# גרפי השוואה
# ============================================================

def create_comparison_plots(
    results_df: pd.DataFrame,
) -> None:
    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC",
    ]

    colors = [
        "#F4A261",
        "#E76F51",
        "#B22222",
    ]

    for metric in metrics:
        pivot = results_df.pivot(
            index="Model",
            columns="Dataset",
            values=metric,
        )

        ax = pivot.plot(
            kind="bar",
            figsize=(10, 6),
            color=colors,
        )

        ax.set_title(
            f"{metric}: "
            f"Clinical vs Lifestyle vs Merged "
            f"(With PCA)"
        )

        ax.set_xlabel("Model")
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=0)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.legend(title="Dataset")

        plt.tight_layout()

        plt.savefig(
            PLOTS_DIR
            / (
                f"with_pca_comparison_"
                f"{metric.lower()}.png"
            ),
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()


def create_roc_plot(
    roc_df: pd.DataFrame,
) -> None:
    merged_roc = roc_df[
        roc_df["Dataset"] == "Merged"
    ].copy()

    if merged_roc.empty:
        return

    plt.figure(figsize=(8, 6))

    for model_name, group in merged_roc.groupby("Model"):
        auc_value = group["AUC"].iloc[0]

        plt.plot(
            group["FPR"],
            group["TPR"],
            label=f"{model_name} (AUC={auc_value:.3f})",
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier",
    )

    plt.title("ROC Curves – Merged Dataset With PCA")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR
        / "merged_roc_curves_with_pca.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# הרצת הניסוי
# ============================================================

def main() -> None:
    full_df = load_original_data()

    datasets = create_experiment_datasets(
        full_df
    )

    train_ids, test_ids = (
        create_shared_split(full_df)
    )

    all_results = []
    all_roc_rows = []

    for dataset_name, dataset_df in datasets.items():
        results, roc_rows = evaluate_dataset_with_pca(
            dataset_name=dataset_name,
            df=dataset_df,
            train_ids=train_ids,
            test_ids=test_ids,
        )

        all_results.extend(results)
        all_roc_rows.extend(roc_rows)

    results_df = pd.DataFrame(
        all_results
    )

    roc_df = pd.DataFrame(
        all_roc_rows
    )

    results_df.to_csv(
        OUTPUT_DIR
        / "all_results_pca_pretty.csv",
        index=False,
        encoding="utf-8-sig",
    )

    roc_df.to_csv(
        OUTPUT_DIR
        / "roc_curves_with_pca.csv",
        index=False,
        encoding="utf-8-sig",
    )

    f1_comparison = results_df.pivot(
        index="Model",
        columns="Dataset",
        values="F1",
    ).reset_index()

    f1_comparison[
        "Improvement_vs_Clinical"
    ] = (
        f1_comparison["Merged"]
        - f1_comparison["Clinical"]
    )

    f1_comparison[
        "Improvement_vs_Lifestyle"
    ] = (
        f1_comparison["Merged"]
        - f1_comparison["Lifestyle"]
    )

    f1_comparison.to_csv(
        OUTPUT_DIR
        / "f1_comparison_with_pca.csv",
        index=False,
        encoding="utf-8-sig",
    )

    create_comparison_plots(
        results_df
    )

    create_roc_plot(
        roc_df
    )

    print_final_summary(
        results_df
    )

    print("\nFiles created:")
    print(
        f"  Results: "
        f"{(OUTPUT_DIR / 'all_results_pca_pretty.csv').resolve()}"
    )
    print(
        f"  F1 comparison: "
        f"{(OUTPUT_DIR / 'f1_comparison_with_pca.csv').resolve()}"
    )
    print(
        f"  Patient predictions folder: "
        f"{PATIENT_PREDICTIONS_DIR.resolve()}"
    )

    print_title(
        "PCA EXPERIMENT COMPLETED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()