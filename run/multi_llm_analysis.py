"""
multi_llm_analysis.py — Cross-model comparison of bias coupling matrices.

Loads result CSVs for multiple LLMs (jury-confirmed rows only), computes
per-model coupling matrices, and produces comparative visualisations:

  1. Side-by-side coupling heatmaps for all loaded models
  2. Replication map  — fraction of models with |mean Δ| ≥ threshold per cell
  3. Target susceptibility radar chart (row means per model)
  4. Capability vs. bias-susceptibility scatter  (MMLU proxy)
  5. Cross-model coupling correlation matrix  (Spearman ρ of M vectors)

Usage:
    python multi_llm_analysis.py                             # auto-discovers CSVs
    python multi_llm_analysis.py --models GPT-4o GPT-4o-Mini
    python multi_llm_analysis.py --csv path/a.csv path/b.csv

Outputs → run/data/plots/multi_llm/
"""

import os, glob, argparse, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")

_RUN_DIR     = os.path.dirname(os.path.abspath(__file__))
_RESULTS_DIR = os.path.join(_RUN_DIR, "data", "conditional_decision_results")
_PLOT_DIR    = os.path.join(_RUN_DIR, "data", "plots", "multi_llm")

BIAS_ORDER = [
    "Anchoring", "Availability Heuristic", "Bandwagon Effect",
    "Confirmation Bias", "Framing Effect", "In Group Bias",
    "Loss Aversion", "Planning Fallacy", "Status Quo Bias",
]

# MMLU 5-shot benchmark scores (0-100) — public leaderboard values, early 2025
CAPABILITY_SCORES = {
    # Closed-weight frontier
    "GPT-4o":             88.0,
    "GPT-4o-Mini":        82.0,
    "Claude-3.5-Sonnet":  88.3,
    "Claude-3.5-Haiku":   79.0,
    "Gemini-1.5-Pro":     85.9,
    # Open-weight large
    "Llama-3.1-70B":      83.6,
    "Qwen-2.5-72B-Instruct": 83.3,
    # Open-weight small / cross-pipeline
    "Llama-3.1-8B":       66.7,
    "Phi-3.5":            69.0,
    "Gemma-2-9B-IT":      71.3,
    # Reasoning / distinct architecture
    "DeepSeek-V3":        88.5,
    # Additional
    "Gemini-1.5-Flash":   78.9,
    "Mistral-Large-2":    84.0,
}

# Model groupings for structured visualizations
MODEL_GROUPS = {
    "Closed Frontier": ["GPT-4o", "Claude-3.5-Haiku", "Gemini-1.5-Pro"],
    "Open Large":      ["Llama-3.1-70B", "Qwen-2.5-72B-Instruct"],
    "Open Small":      ["Llama-3.1-8B", "Phi-3.5", "Gemma-2-9B-IT"],
    "Reasoning":       ["DeepSeek-V3"],
}

# Canonical display names (short labels for plots)
MODEL_LABELS = {
    "GPT-4o":                "GPT-4o",
    "Claude-3.5-Haiku":      "Claude-3.5\nHaiku",
    "Gemini-1.5-Pro":        "Gemini-1.5\nPro",
    "Llama-3.1-70B":         "Llama-70B",
    "Qwen-2.5-72B-Instruct": "Qwen-72B",
    "Llama-3.1-8B":          "Llama-8B",
    "Phi-3.5":               "Phi-3.5",
    "Gemma-2-9B-IT":         "Gemma-9B",
    "DeepSeek-V3":           "DeepSeek\nV3",
}

PALETTE = ["#4A9EFF", "#4CC98A", "#F5C542", "#E55A4E",
           "#C084FC", "#FB923C", "#38BDF8", "#A3E635"]

_DARK_BG  = "#0F172A"
_PANEL_BG = "#151F35"


def _savefig(name):
    os.makedirs(_PLOT_DIR, exist_ok=True)
    path = os.path.join(_PLOT_DIR, name)
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor=plt.gcf().get_facecolor())
    plt.close()
    print(f"  saved → {path}")


def _dark_ax(ax):
    ax.set_facecolor(_PANEL_BG)
    for sp in ax.spines.values():
        sp.set_color("#4A9EFF")
    ax.tick_params(colors="white")
    return ax


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def discover_csvs(models=None):
    """Auto-discover model result CSVs in _RESULTS_DIR."""
    found = glob.glob(os.path.join(_RESULTS_DIR, "*.csv"))
    valid = []
    for f in found:
        try:
            cols = pd.read_csv(f, nrows=1).columns.tolist()
            if "delta_score" in cols and "jury_passed" in cols:
                name = os.path.basename(f).replace("_pilot.csv", "").replace(".csv", "")
                if models is None or name in models:
                    valid.append((name, f))
        except Exception:
            pass
    return sorted(valid, key=lambda x: x[0])


def load_model(csv_path):
    df = pd.read_csv(csv_path)
    jp = df[(df["status"] == "OK") & (df["jury_passed"] == True)].copy()
    pivot = jp.pivot_table(values="delta_score", index="bias",
                           columns="human_bias", aggfunc="mean")
    pivot = pivot.reindex(index=BIAS_ORDER, columns=BIAS_ORDER)
    return jp, pivot


# ══════════════════════════════════════════════════════════════════════════════
# 1. SIDE-BY-SIDE COUPLING MATRICES
# ══════════════════════════════════════════════════════════════════════════════

def plot_coupling_comparison(model_data):
    n = len(model_data)
    if n == 0:
        print("  No models loaded."); return

    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(9 * ncols, 7.5 * nrows),
                             facecolor=_DARK_BG)
    axes_flat = np.array(axes).flatten() if n > 1 else np.array([axes])

    for idx, (name, jp, pivot) in enumerate(model_data):
        ax = axes_flat[idx]
        ax.set_facecolor(_PANEL_BG)
        mask = pivot.isnull()
        sns.heatmap(
            pivot.round(2), ax=ax,
            cmap=sns.diverging_palette(220, 20, as_cmap=True),
            center=0, annot=True, fmt=".2f", annot_kws={"size": 6.5},
            vmin=-1, vmax=1, mask=mask,
            linewidths=0.3, linecolor="#1E293B",
            cbar_kws={"shrink": 0.55, "label": "Mean Δ"},
        )
        ax.set_title(f"{name}  (N={len(jp)} jury-confirmed)",
                     color="white", fontsize=10, pad=6)
        ax.set_xticklabels([b.replace(" ", "\n") for b in BIAS_ORDER],
                           rotation=0, ha="center", fontsize=6.5, color="white")
        ax.set_yticklabels(BIAS_ORDER, fontsize=6.5, color="white")
        ax.xaxis.tick_top(); ax.xaxis.set_label_position("top")
        ax.tick_params(colors="white")
        for sp in ax.spines.values(): sp.set_color("#4A9EFF")
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.label.set_color("white")
        cbar.ax.tick_params(colors="white")

    for idx in range(len(model_data), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Bias Coupling Matrix M[β,γ] — Cross-model comparison  (jury-confirmed)",
                 color="white", fontsize=13, y=1.01)
    fig.tight_layout()
    _savefig("coupling_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# 2. REPLICATION MAP
# ══════════════════════════════════════════════════════════════════════════════

def plot_replication_map(model_data, threshold=0.05):
    if len(model_data) < 2:
        print("  Need ≥ 2 models for replication map."); return

    rep_amp  = np.zeros((10, 10))
    rep_supp = np.zeros((10, 10))
    n_models = len(model_data)

    for name, jp, pivot in model_data:
        for i, b in enumerate(BIAS_ORDER):
            for j, h in enumerate(BIAS_ORDER):
                val = pivot.loc[b, h] if (b in pivot.index and h in pivot.columns) else np.nan
                if np.isnan(val): continue
                if val >=  threshold: rep_amp[i, j]  += 1
                if val <= -threshold: rep_supp[i, j] += 1

    frac_amp  = rep_amp  / n_models
    frac_supp = rep_supp / n_models
    net       = frac_amp - frac_supp

    fig, axes = plt.subplots(1, 3, figsize=(21, 7), facecolor=_DARK_BG)
    specs = [
        (frac_amp,  f"Amplification replication\n(fraction of {n_models} models with Δ≥{threshold})",
         "Greens", None),
        (frac_supp, f"Suppression replication\n(fraction of {n_models} models with Δ≤−{threshold})",
         "Reds", None),
        (net,       f"Net replication\n(frac_amp − frac_supp)",
         sns.diverging_palette(220, 20, as_cmap=True), 0),
    ]
    for ax, (data, title, cmap, center) in zip(axes, specs):
        ax.set_facecolor(_PANEL_BG)
        df_p = pd.DataFrame(data, index=BIAS_ORDER, columns=BIAS_ORDER)
        kw = {"center": center} if center is not None else {}
        sns.heatmap(df_p.round(2), cmap=cmap, annot=True, fmt=".2f",
                    annot_kws={"size": 7}, linewidths=0.3, linecolor="#0F172A",
                    cbar_kws={"shrink": 0.65}, ax=ax, **kw)
        ax.set_title(title, color="white", fontsize=9, pad=6)
        ax.set_xticklabels([b.replace(" ", "\n") for b in BIAS_ORDER],
                           rotation=0, ha="center", fontsize=6.5, color="white")
        ax.set_yticklabels(BIAS_ORDER, fontsize=6.5, color="white")
        ax.xaxis.tick_top(); ax.xaxis.set_label_position("top")
        ax.tick_params(colors="white")
        for sp in ax.spines.values(): sp.set_color("#4A9EFF")
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(colors="white")

    fig.suptitle(f"Cross-Model Replication  ({n_models} models, |Δ| threshold={threshold})",
                 color="white", fontsize=12, y=1.01)
    fig.tight_layout()
    _savefig("replication_map.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3. TARGET SUSCEPTIBILITY RADAR CHART
# ══════════════════════════════════════════════════════════════════════════════

def plot_susceptibility_radar(model_data):
    if not model_data: return

    row_means = {}
    for name, jp, pivot in model_data:
        row_means[name] = (jp.groupby("bias")["delta_score"]
                           .mean()
                           .reindex(BIAS_ORDER)
                           .fillna(0)
                           .values)

    n_biases = len(BIAS_ORDER)
    angles   = np.linspace(0, 2 * np.pi, n_biases, endpoint=False).tolist()
    angles  += angles[:1]

    fig = plt.figure(figsize=(9, 8), facecolor=_DARK_BG)
    ax  = fig.add_subplot(111, polar=True)
    ax.set_facecolor(_PANEL_BG)
    ax.tick_params(colors="white")
    ax.spines["polar"].set_color("#4A9EFF")

    for i, (name, vals) in enumerate(row_means.items()):
        data = vals.tolist() + vals[:1].tolist()
        col  = PALETTE[i % len(PALETTE)]
        ax.plot(angles, data, "o-", lw=2, color=col, alpha=0.9, markersize=5, label=name)
        ax.fill(angles, data, alpha=0.12, color=col)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([b.replace(" ", "\n") for b in BIAS_ORDER],
                       fontsize=8.5, color="white")
    ax.set_title("Target bias susceptibility by model\n(row means of M, jury-confirmed)",
                 color="white", fontsize=10, pad=20)
    ax.tick_params(axis="y", colors="white", labelsize=7)
    ax.legend(loc="upper right", bbox_to_anchor=(1.38, 1.12),
              fontsize=9.5, facecolor=_DARK_BG, edgecolor="#4A9EFF", labelcolor="white")

    _savefig("susceptibility_radar.png")


# ══════════════════════════════════════════════════════════════════════════════
# 4. CAPABILITY VS. SUSCEPTIBILITY SCATTER
# ══════════════════════════════════════════════════════════════════════════════

def plot_capability_vs_susceptibility(model_data):
    rows = []
    for name, jp, pivot in model_data:
        cap = CAPABILITY_SCORES.get(name)
        if cap is None:
            print(f"  [WARN] No MMLU score for {name} — skipping in scatter.")
            continue
        rows.append({
            "model":          name,
            "capability":     cap,
            "mean_abs_delta": jp["delta_score"].abs().mean(),
            "mean_delta":     jp["delta_score"].mean(),
            "n":              len(jp),
        })
    if len(rows) < 2:
        print("  Need ≥ 2 models with known capability scores."); return
    df = pd.DataFrame(rows)

    fig = plt.figure(figsize=(8, 6), facecolor=_DARK_BG)
    ax  = _dark_ax(fig.add_subplot(111))

    for i, row in df.iterrows():
        col = PALETTE[i % len(PALETTE)]
        ax.scatter(row["capability"], row["mean_abs_delta"],
                   s=220, color=col, zorder=4, edgecolors="white", lw=0.8)
        ax.annotate(row["model"],
                    (row["capability"], row["mean_abs_delta"]),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=9, color=col)

    if len(df) >= 3:
        m, b_r = np.polyfit(df["capability"], df["mean_abs_delta"], 1)
        xs = np.linspace(df["capability"].min() - 2, df["capability"].max() + 2, 100)
        ax.plot(xs, m * xs + b_r, color="#F5C542", lw=1.5, ls="--", alpha=0.7,
                label=f"OLS slope={m:.4f}")
        rho, p = spearmanr(df["capability"], df["mean_abs_delta"])
        ax.text(0.05, 0.95, f"Spearman ρ={rho:.3f}  p={p:.3f}",
                transform=ax.transAxes, fontsize=10, color="white", va="top")
        ax.legend(fontsize=9, facecolor=_DARK_BG, edgecolor="#4A9EFF", labelcolor="white")

    ax.set_xlabel("MMLU score  (benchmark capability proxy)", color="white", fontsize=10)
    ax.set_ylabel("Mean |Δ|  (overall bias susceptibility)", color="white", fontsize=10)
    ax.set_title("Does higher capability → lower bias susceptibility?\n(jury-confirmed rows)",
                 color="white", fontsize=10, pad=6)
    fig.tight_layout()
    _savefig("capability_vs_susceptibility.png")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 5. CROSS-MODEL COUPLING CORRELATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_cross_model_correlation(model_data):
    if len(model_data) < 2:
        print("  Need ≥ 2 models for correlation matrix."); return

    vectors = {name: pivot.values.flatten() for name, jp, pivot in model_data}
    names   = list(vectors.keys())
    n       = len(names)
    corr_mat = np.eye(n)

    for i in range(n):
        for j in range(i + 1, n):
            v1, v2 = vectors[names[i]], vectors[names[j]]
            mask = ~(np.isnan(v1) | np.isnan(v2))
            if mask.sum() < 5:
                corr_mat[i, j] = corr_mat[j, i] = np.nan
                continue
            rho, _ = spearmanr(v1[mask], v2[mask])
            corr_mat[i, j] = corr_mat[j, i] = rho

    side = max(5, n * 1.6)
    fig = plt.figure(figsize=(side, side * 0.85), facecolor=_DARK_BG)
    ax  = fig.add_subplot(111)
    ax.set_facecolor(_PANEL_BG)
    df_corr = pd.DataFrame(corr_mat, index=names, columns=names)
    sns.heatmap(df_corr.round(2), cmap="RdYlGn", center=0,
                annot=True, fmt=".2f", annot_kws={"size": 12},
                linewidths=0.5, linecolor="#0F172A",
                cbar_kws={"shrink": 0.65, "label": "Spearman ρ"}, ax=ax)
    ax.set_title("Cross-model coupling matrix correlation\n(Spearman ρ of flattened M[β,γ])",
                 color="white", fontsize=11, pad=6)
    ax.set_xticklabels(names, rotation=30, ha="right", color="white", fontsize=10)
    ax.set_yticklabels(names, color="white", fontsize=10)
    ax.tick_params(colors="white")
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors="white")
    cbar.ax.yaxis.label.set_color("white")
    for sp in ax.spines.values(): sp.set_color("#4A9EFF")
    fig.tight_layout()
    _savefig("cross_model_correlation.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(model_names=None, csv_paths=None):
    os.makedirs(_PLOT_DIR, exist_ok=True)

    if csv_paths:
        pairs = [(os.path.basename(p).replace("_pilot.csv", "").replace(".csv", ""), p)
                 for p in csv_paths]
    else:
        pairs = discover_csvs(model_names)

    if not pairs:
        print(f"No valid result CSVs found in {_RESULTS_DIR}")
        print("Run the pilot for at least one model first:")
        print("  python run_pilot.py --model GPT-4o --n_samples 10 ...")
        return

    print(f"\nFound {len(pairs)} model CSV(s): {[p[0] for p in pairs]}")
    model_data = []
    for name, path in pairs:
        jp, pivot = load_model(path)
        print(f"  {name}: {len(jp)} jury-confirmed rows")
        model_data.append((name, jp, pivot))

    print("\n[1/5] Side-by-side coupling matrices …")
    plot_coupling_comparison(model_data)

    print("\n[2/5] Replication map …")
    plot_replication_map(model_data)

    print("\n[3/5] Susceptibility radar chart …")
    plot_susceptibility_radar(model_data)

    print("\n[4/5] Capability vs. susceptibility scatter …")
    plot_capability_vs_susceptibility(model_data)

    print("\n[5/5] Cross-model coupling correlation …")
    plot_cross_model_correlation(model_data)

    print(f"\nAll multi-LLM plots saved → {_PLOT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cross-model comparison of bias coupling matrices."
    )
    parser.add_argument("--models", nargs="*", default=None,
                        help="Model name(s) to include (auto-discovers if omitted)")
    parser.add_argument("--csv", nargs="*", default=None,
                        help="Explicit CSV paths (overrides --models)")
    args = parser.parse_args()
    main(model_names=args.models, csv_paths=args.csv)
