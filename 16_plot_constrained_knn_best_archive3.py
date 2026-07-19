
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config_archive3 import OUTPUT_DIR, PLOTS_DIR

MODEL_COLORS = {"KNN": "#3F72C4", "SVM": "#4C924C", "XGBoost": "#E080B0", "MLP": "#E8A93A"}
MODELS = list(MODEL_COLORS)              
OTHER = ["SVM", "XGBoost", "MLP"]        

COMBO_ORDER = [
    "3C+1L", "3C+2L", "4C+2L", "5C+2L", "4C+3L", "5C+3L",
    "6C+2L", "6C+3L", "7C+3L", "5C+5L", "all-21 (both parts)",
]
TRIMMED = [c for c in COMBO_ORDER if not c.startswith("all-21")]  
WIDTH = 0.2


def tick_label(combo, k):
    name = "all-21" if combo.startswith("all-21") else combo
    return f"{name}\nk={k}"


def make_figure(df, piv, klab, value_col, title, ylabel, ymax, ystep, out):
    # KNN's best trimmed subset for this metric
    knn_vals = {c: piv.loc[c, (value_col, "KNN")] for c in TRIMMED}
    knn_best = max(knn_vals, key=knn_vals.get)

    x = np.arange(len(COMBO_ORDER))
    fig, ax = plt.subplots(figsize=(16, 6.4))

    for i, model in enumerate(MODELS):
        offset = (i - (len(MODELS) - 1) / 2) * WIDTH
        if model == "KNN":
            xi = np.array([COMBO_ORDER.index(knn_best)])
            vals = np.array([piv.loc[knn_best, (value_col, "KNN")]])
        else:
            xi = x
            vals = piv.loc[COMBO_ORDER, (value_col, model)].values
        bars = ax.bar(xi + offset, vals, WIDTH, label=model, color=MODEL_COLORS[model],
                      edgecolor="white", linewidth=0.6, zorder=3)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=6, rotation=90)

    
    kb_x = COMBO_ORDER.index(knn_best) + (0 - (len(MODELS) - 1) / 2) * WIDTH
    kb_y = piv.loc[knn_best, (value_col, "KNN")]
    kn = int(df[(df.model == "KNN") & (df.combo == knn_best)]["n_neighbors"].iloc[0])
    ax.annotate(f"KNN best subset\n({knn_best}, K={kn})",
                xy=(kb_x, kb_y), xytext=(kb_x - 0.3, ymax * 0.93),
                fontsize=9, color="#2B4C86", ha="center", va="top",
                arrowprops=dict(arrowstyle="->", color="#2B4C86", lw=1.2))

    ax.set_ylim(0, ymax)
    ax.set_yticks(np.arange(0, ymax + 1e-9, ystep))
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([tick_label(c, klab[c]) for c in COMBO_ORDER], fontsize=8)
    ax.grid(axis="y", color="#CCCCCC", alpha=0.6, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.margins(x=0.01)
   
    ax.legend(loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.09), fontsize=11)

    fig.suptitle("Constrained feature selection on Merged (top clinical + top lifestyle, both parts kept)",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved {out} | KNN best = {knn_best} ({value_col}={kb_y:.4f}, K={kn})")


def main():
    df = pd.read_csv(OUTPUT_DIR / "constrained_feature_selection_all_models.csv")
    klab = {c: int(df[df["combo"] == c]["k"].iloc[0]) for c in COMBO_ORDER}
    piv = df.pivot_table(index="combo", columns="model", values=["roc_auc", "f1"])
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    make_figure(df, piv, klab, "f1", "Test F1 by feature subset (KNN shown only at its best subset)",
                "Test F1", 0.50, 0.10, PLOTS_DIR / "constrained_knnbest_f1.png")
    make_figure(df, piv, klab, "roc_auc", "ROC AUC by feature subset (KNN shown only at its best subset)",
                "ROC AUC", 1.0, 0.25, PLOTS_DIR / "constrained_knnbest_auc.png")


if __name__ == "__main__":
    main()
