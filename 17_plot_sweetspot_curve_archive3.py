"""Performance vs. number of features -- the 'sweet spot' curve.

For each model and each feature-count k, take the BEST subset of that size (max metric),
so the line is the best achievable performance at each k. Shows F1 and ROC AUC peaking
around k=7 and then declining (SVM/MLP) or staying flat (XGBoost), with KNN rising toward
the full set. Same house palette as the other archive3 charts.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config_archive3 import OUTPUT_DIR, PLOTS_DIR

MODEL_COLORS = {"KNN": "#3F72C4", "SVM": "#4C924C", "XGBoost": "#E080B0", "MLP": "#E8A93A"}
MODELS = list(MODEL_COLORS)
SWEET = 7


def draw(ax, best, ks, col, title, ylabel):
    for m in MODELS:
        ax.plot(ks, [best[(m, k)][col] for k in ks], marker="o", ms=6, lw=2,
                color=MODEL_COLORS[m], label=m, zorder=3, clip_on=False)
    ax.axvline(SWEET, ls="--", lw=1.3, color="#9A8F7A", zorder=1)
    ymax = ax.get_ylim()[1]
    ax.text(SWEET, ymax, "  sweet spot (7)", color="#6F6552", fontsize=9,
            va="top", ha="left")
    ax.set_title(title, fontsize=12, pad=8)
    ax.set_xlabel("number of features (k)", fontsize=10.5)
    ax.set_ylabel(ylabel, fontsize=10.5)
    ax.set_xticks(ks)
    ax.set_xticklabels(ks)
    ax.grid(True, color="#D8D2C6", alpha=0.6, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.margins(x=0.03)


def main():
    df = pd.read_csv(OUTPUT_DIR / "constrained_feature_selection_all_models.csv")
    df["k"] = df["k"].astype(int)
    ks = sorted(df["k"].unique())
    # best subset of each size, per model
    best = {}
    for m in MODELS:
        sub = df[df.model == m]
        for k in ks:
            rows = sub[sub.k == k]
            best[(m, k)] = rows.loc[rows.f1.idxmax()]  # best-F1 subset of this size

    # for AUC use the best-AUC subset of each size instead
    bestA = {}
    for m in MODELS:
        sub = df[df.model == m]
        for k in ks:
            rows = sub[sub.k == k]
            bestA[(m, k)] = rows.loc[rows.roc_auc.idxmax()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Performance vs. number of features (best subset of each size, both fusion parts kept)",
                 fontsize=13.5, fontweight="bold", y=0.98)
    draw(ax1, best, ks, "f1", "Test F1 by feature count", "Test F1")
    draw(ax2, bestA, ks, "roc_auc", "ROC AUC by feature count", "ROC AUC")

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.93), fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / "sweetspot_curve_f1_auc.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
