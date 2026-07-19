"""Grouped-bar figure for the constrained-feature-selection-all-models results.

Matches the house style of the archive3 PCA figures: two panels (ROC AUC, Test F1),
one grouped bar per model (KNN / SVM / XGBoost / MLP), value labels on top. Here the
x-axis is the feature subset (top clinical + top lifestyle, both parts kept) instead of
the Clinical/Lifestyle/Merged datasets.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config_archive3 import OUTPUT_DIR, PLOTS_DIR

# house palette (same model->color mapping as the existing archive3 charts)
MODEL_COLORS = {
    "KNN": "#3F72C4",      # blue
    "SVM": "#4C924C",      # green
    "XGBoost": "#E080B0",  # pink
    "MLP": "#E8A93A",      # gold
}
MODELS = list(MODEL_COLORS)

# x-order: increasing k; all-21 reference last
COMBO_ORDER = [
    "3C+1L", "3C+2L", "4C+2L", "5C+2L", "4C+3L", "5C+3L",
    "6C+2L", "6C+3L", "7C+3L", "5C+5L", "all-21 (both parts)",
]
def tick_label(combo, k):
    name = "all-21" if combo.startswith("all-21") else combo
    return f"{name}\nk={k}"


def draw(ax, piv, klab, value_col, title, ylabel, ymax, ystep):
    x = np.arange(len(COMBO_ORDER))
    width = 0.2
    for i, model in enumerate(MODELS):
        vals = piv.loc[COMBO_ORDER, (value_col, model)].values
        offset = (i - (len(MODELS) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=model, color=MODEL_COLORS[model],
                      edgecolor="white", linewidth=0.6, zorder=3)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=5.5, rotation=90)

    ax.set_ylim(0, ymax)
    ax.set_yticks(np.arange(0, ymax + 1e-9, ystep))
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xticks(x)
    labels = [tick_label(c, klab[c]) for c in COMBO_ORDER]
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(axis="y", color="#CCCCCC", alpha=0.6, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.margins(x=0.01)


def main():
    df = pd.read_csv(OUTPUT_DIR / "constrained_feature_selection_all_models.csv")
    klab = {c: int(df[df["combo"] == c]["k"].iloc[0]) for c in COMBO_ORDER}
    piv = df.pivot_table(index="combo", columns="model", values=["roc_auc", "f1"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6.2))
    fig.suptitle("Constrained feature selection on Merged (top clinical + top lifestyle, both parts kept)",
                 fontsize=14, fontweight="bold", y=0.99)

    draw(ax1, piv, klab, "roc_auc", "ROC AUC by feature subset", "ROC AUC", 1.0, 0.25)
    draw(ax2, piv, klab, "f1", "Test F1 by feature subset", "Test F1", 0.50, 0.10)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.955), fontsize=11)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / "constrained_all_models_auc_f1.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
