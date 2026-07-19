
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config_archive3 import OUTPUT_DIR, PLOTS_DIR

METRICS = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]

MODEL_ORDER = ["KNN", "SVM", "XGBoost", "MLP", "PyTorch-LateFusion", "CatBoost"]
MODEL_COLORS = {
    "KNN": "#2a78d6",                 # blue
    "SVM": "#008300",                 # green
    "XGBoost": "#e87ba4",             # magenta
    "MLP": "#eda100",                 # yellow
    "PyTorch-LateFusion": "#1baf7a",  # aqua
    "CatBoost": "#eb6834",            # orange
}


SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BAR_EDGE = "#0b0b0b"  

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def load_all_models(use_pca: bool) -> pd.DataFrame:
    if use_pca:
        base = pd.read_csv(OUTPUT_DIR / "merged_results_pca.csv")       # KNN, SVM, XGBoost, MLP (PCA)
    else:
        # Same four models without PCA. The ablation file also holds Clinical/Lifestyle
        # rows, so restrict to Merged to match the two new models' dataset.
        ablation = pd.read_csv(OUTPUT_DIR / "all_results_no_pca_ablation.csv")
        base = ablation[ablation["Dataset"] == "Merged"]
    extra = pd.read_csv(OUTPUT_DIR / "extra_models_results.csv")        # PyTorch-LateFusion, CatBoost (already no-PCA)
    cols = ["Model"] + METRICS
    combined = pd.concat([base[cols], extra[cols]], ignore_index=True)
    combined = combined[combined["Model"].isin(MODEL_ORDER)].copy()
    combined["Model"] = pd.Categorical(combined["Model"], categories=MODEL_ORDER, ordered=True)
    return combined.sort_values("Model").reset_index(drop=True)


def style_axes(ax):
    ax.set_ylim(0, 1)  # bars must start at 0; all metrics share the [0,1] scale
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, color=GRID, alpha=0.9)
    ax.set_axisbelow(True)


def per_metric_charts(df: pd.DataFrame, prefix: str, footnote: str):
    for metric in METRICS:
        d = df[["Model", metric]].sort_values(metric, ascending=False).reset_index(drop=True)
        colors = [MODEL_COLORS[m] for m in d["Model"]]
        fig, ax = plt.subplots(figsize=(9, 5.5))
        bars = ax.bar(d["Model"].astype(str), d[metric], color=colors,
                      edgecolor=BAR_EDGE, linewidth=0.6, width=0.68, zorder=3)
        for bar, val in zip(bars, d[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.012, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=10, color=INK_SECONDARY, zorder=4)
        style_axes(ax)
        ax.set_title(f"{metric.replace('_', '-')} — all models on the Merged dataset",
                     fontsize=14, color=INK_PRIMARY, pad=12, loc="left", fontweight="bold")
        ax.set_ylabel(metric.replace("_", "-"), fontsize=11, color=INK_SECONDARY)
        ax.tick_params(axis="x", labelsize=10.5, colors=INK_SECONDARY, rotation=0)
        fig.text(0.01, 0.005, footnote, fontsize=7.5, color=INK_MUTED)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        out = PLOTS_DIR / f"{prefix}{metric.lower()}.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)
        print(f"wrote {out}")


def grouped_overview(df: pd.DataFrame, prefix: str, title: str, footnote: str):
    fig, ax = plt.subplots(figsize=(12, 6.5))
    n_models = len(MODEL_ORDER)
    x = np.arange(len(METRICS))
    width = 0.8 / n_models
    for i, model in enumerate(MODEL_ORDER):
        row = df[df["Model"] == model].iloc[0]
        vals = [row[m] for m in METRICS]
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, vals, width=width * 0.92, label=model,
               color=MODEL_COLORS[model], edgecolor=BAR_EDGE, linewidth=0.5, zorder=3)
    style_axes(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "-") for m in METRICS], fontsize=11, color=INK_SECONDARY)
    ax.set_ylabel("Score", fontsize=11, color=INK_SECONDARY)
    ax.set_title(title, fontsize=15, color=INK_PRIMARY, pad=12, loc="left", fontweight="bold")
    ax.legend(ncol=6, frameon=False, fontsize=9.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.08), labelcolor=INK_SECONDARY)
    fig.text(0.01, 0.005, footnote, fontsize=8, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = PLOTS_DIR / f"{prefix}comparison_grouped.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


FOOT_PCA = ("Base 4 models: StandardScaler→PCA→SMOTE pipeline · PyTorch-LateFusion & CatBoost: "
            "as configured (no PCA) · identical 50,736-row test set")
FOOT_NOPCA = ("All six models without PCA — StandardScaler→SMOTE for the base 4, PyTorch-LateFusion & "
              "CatBoost as configured · 21 features · identical 50,736-row test set")


def run_variant(use_pca: bool, prefix: str, title: str, footnote: str, csv_name: str):
    tag = "PCA" if use_pca else "no-PCA"
    df = load_all_models(use_pca)
    print(f"\n=== {tag} ===")
    print(df.set_index("Model")[METRICS].round(4).to_string())
    per_metric_charts(df, prefix, footnote)
    grouped_overview(df, prefix, title, footnote)
    df.to_csv(OUTPUT_DIR / csv_name, index=False, encoding="utf-8-sig")
    print(f"wrote {OUTPUT_DIR / csv_name}")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    run_variant(True, "all_models_",
                "All six models across every metric — Merged dataset (PCA pipeline)",
                FOOT_PCA, "all_models_merged_comparison.csv")
    run_variant(False, "all_models_nopca_",
                "All six models across every metric — Merged dataset (no PCA)",
                FOOT_NOPCA, "all_models_merged_comparison_nopca.csv")


if __name__ == "__main__":
    main()
