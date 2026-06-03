"""
extended_analysis.py — Additional scientific analyses on jury-confirmed pilot data.

Analyses:
  1. Semantic similarity correlation  (latent-priming test, Panel C)
  2. Bootstrap confidence intervals   (stability of coupling matrix cells)
  3. SVD / rank decomposition         (intrinsic dimensionality of M)
  4. Bias taxonomy grouping test      (within- vs. across-group induction)
  5. Message quality regression       (does stronger bias signal → higher Δ?)
  6. Adversarial neutral check        (is neutral_score truly zero?)
  7. Directed network graph           (visualise induction structure)

Outputs → run/data/plots/extended/
Summary → run/data/conditional_decision_results/extended_summary.csv

Usage:
    python extended_analysis.py
    python extended_analysis.py --input path/to/pilot.csv --skip_semantic
"""

import os, sys, json, warnings, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")

_RUN_DIR   = os.path.dirname(os.path.abspath(__file__))
_PILOT_CSV = os.path.join(_RUN_DIR, "data", "conditional_decision_results", "GPT-4o_pilot.csv")
_PLOT_DIR  = os.path.join(_RUN_DIR, "data", "plots", "extended")
_EMBED_CACHE = os.path.join(_RUN_DIR, "data", "bias_embeddings.json")
_SUMMARY_CSV = os.path.join(_RUN_DIR, "data", "conditional_decision_results", "extended_summary.csv")

BIAS_ORDER = [
    "Anchoring", "Availability Heuristic", "Bandwagon Effect",
    "Confirmation Bias", "Framing Effect", "In Group Bias",
    "Loss Aversion", "Planning Fallacy", "Status Quo Bias",
]

# Cognitive bias taxonomy groupings
TAXONOMY = {
    "Probability & Estimation": ["Anchoring", "Availability Heuristic", "Framing Effect"],
    "Social & Conformity":      ["Bandwagon Effect", "In Group Bias", "Confirmation Bias"],
    "Temporal & Planning":      ["Planning Fallacy"],
    "Value & Risk":             ["Loss Aversion", "Status Quo Bias"],
}
BIAS_TO_GROUP = {b: g for g, bs in TAXONOMY.items() for b in bs}
GROUP_COLORS  = {
    "Probability & Estimation": "#4A9EFF",
    "Social & Conformity":      "#E55A4E",
    "Temporal & Planning":      "#4CC98A",
    "Value & Risk":             "#F5C542",
}

_DARK_BG = "#0F172A"
_PANEL_BG = "#151F35"

def _savefig(name):
    os.makedirs(_PLOT_DIR, exist_ok=True)
    path = os.path.join(_PLOT_DIR, name)
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor=plt.gcf().get_facecolor())
    plt.close()
    print(f"  saved → {path}")

def _dark_fig(figsize):
    fig = plt.figure(figsize=figsize, facecolor=_DARK_BG)
    return fig

def _dark_ax(ax):
    ax.set_facecolor(_PANEL_BG)
    for sp in ax.spines.values(): sp.set_color("#4A9EFF")
    ax.tick_params(colors="white")
    return ax


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

def load(path):
    df = pd.read_csv(path)
    jp = df[(df["status"] == "OK") & (df["jury_passed"] == True)].copy()
    jp["scenario_id"] = jp["id"].astype(str) + "_" + jp["bias"]
    jp["is_diagonal"]  = (jp["bias"] == jp["human_bias"]).astype(int)
    jp["bias_group"]   = jp["bias"].map(BIAS_TO_GROUP)
    jp["hbias_group"]  = jp["human_bias"].map(BIAS_TO_GROUP)
    jp["same_group"]   = (jp["bias_group"] == jp["hbias_group"]).astype(int)
    pivot = jp.pivot_table(values="delta_score", index="bias",
                           columns="human_bias", aggfunc="mean")
    pivot = pivot.reindex(index=BIAS_ORDER, columns=BIAS_ORDER)
    print(f"Loaded {len(jp)} jury-confirmed rows | {jp['bias'].nunique()} target biases")
    return jp, pivot


# ══════════════════════════════════════════════════════════════════════════════
# 1. SEMANTIC SIMILARITY CORRELATION
# ══════════════════════════════════════════════════════════════════════════════

def _get_embeddings(biases):
    """Load cached embeddings or fetch from OpenAI."""
    cache = {}
    if os.path.exists(_EMBED_CACHE):
        with open(_EMBED_CACHE) as f:
            cache = json.load(f)
    missing = [b for b in biases if b not in cache]
    if missing:
        try:
            from openai import OpenAI
            client = OpenAI()
            resp = client.embeddings.create(
                model="text-embedding-3-small",
                input=[f"{b} cognitive bias" for b in missing],
            )
            for b, item in zip(missing, resp.data):
                cache[b] = item.embedding
            os.makedirs(os.path.dirname(_EMBED_CACHE), exist_ok=True)
            with open(_EMBED_CACHE, "w") as f:
                json.dump(cache, f)
            print(f"  Fetched embeddings for {len(missing)} biases and cached.")
        except Exception as e:
            print(f"  [WARNING] Could not fetch embeddings: {e}")
            print("  Set OPENAI_API_KEY and re-run to enable semantic similarity analysis.")
            return None
    vecs = np.array([cache[b] for b in biases])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return (vecs / norms) @ (vecs / norms).T


def run_semantic_similarity(jp, pivot):
    print("\n[1/7] Semantic similarity correlation …")
    sim = _get_embeddings(BIAS_ORDER)
    if sim is None:
        print("  Skipping — no embeddings available.")
        return {"status": "skipped", "spearman_r": np.nan, "spearman_p": np.nan}

    sim_df = pd.DataFrame(sim, index=BIAS_ORDER, columns=BIAS_ORDER)

    # Off-diagonal pairs only
    pairs = []
    for i, b in enumerate(BIAS_ORDER):
        for j, h in enumerate(BIAS_ORDER):
            if i == j: continue
            delta = pivot.loc[b, h] if (b in pivot.index and h in pivot.columns) else np.nan
            if np.isnan(delta): continue
            pairs.append({"target": b, "human": h,
                          "delta": delta, "similarity": sim_df.loc[b, h],
                          "same_group": int(BIAS_TO_GROUP.get(b) == BIAS_TO_GROUP.get(h))})
    pairs_df = pd.DataFrame(pairs)

    rho, p = spearmanr(pairs_df["similarity"], pairs_df["delta"])
    print(f"  Spearman ρ = {rho:.4f}  p = {p:.4f}  n = {len(pairs_df)}")

    # Mantel-style permutation test
    n_perm = 2000
    perm_rhos = []
    for _ in range(n_perm):
        shuffled = pairs_df["similarity"].sample(frac=1).values
        perm_rhos.append(spearmanr(shuffled, pairs_df["delta"])[0])
    mantel_p = np.mean(np.abs(perm_rhos) >= abs(rho))
    print(f"  Mantel permutation p = {mantel_p:.4f}  (n_perm={n_perm})")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=_DARK_BG)

    # Left: scatter with regression
    ax = _dark_ax(axes[0])
    group_cols = [GROUP_COLORS.get(BIAS_TO_GROUP.get(r["target"], ""), "#8A93A8")
                  for _, r in pairs_df.iterrows()]
    ax.scatter(pairs_df["similarity"], pairs_df["delta"],
               c=group_cols, s=55, alpha=0.85, edgecolors="none", zorder=3)
    m, b_reg = np.polyfit(pairs_df["similarity"], pairs_df["delta"], 1)
    xs = np.linspace(pairs_df["similarity"].min(), pairs_df["similarity"].max(), 100)
    ax.plot(xs, m * xs + b_reg, color="#F5C542", lw=2, label=f"OLS slope={m:.2f}")
    ax.axhline(0, color="white", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("Semantic similarity  s(target, human)", color="white", fontsize=10)
    ax.set_ylabel("Mean Δ-score", color="white", fontsize=10)
    ax.set_title(f"Latent priming test\nSpearman ρ={rho:.3f}  p={p:.3f}  Mantel p={mantel_p:.3f}",
                 color="white", fontsize=10, pad=6)
    legend_els = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()]
    ax.legend(handles=legend_els, fontsize=8, facecolor=_DARK_BG,
              edgecolor="#4A9EFF", labelcolor="white")
    ax.legend(handles=legend_els + [plt.Line2D([0],[0],color="#F5C542",lw=2,label="OLS fit")],
              fontsize=8, facecolor=_DARK_BG, edgecolor="#4A9EFF", labelcolor="white")

    # Right: similarity matrix heatmap
    ax2 = axes[1]
    ax2.set_facecolor(_PANEL_BG)
    sns.heatmap(sim_df.round(2), cmap="YlOrRd", annot=True, fmt=".2f",
                annot_kws={"size": 7}, linewidths=0.3, linecolor="#0F172A",
                cbar_kws={"shrink": 0.7, "label": "Cosine similarity"}, ax=ax2)
    ax2.set_title("Bias semantic similarity matrix S[β,γ]", color="white", fontsize=10, pad=6)
    ax2.set_xticklabels([b.replace(" ", "\n") for b in BIAS_ORDER], rotation=0,
                        ha="center", fontsize=7, color="white")
    ax2.set_yticklabels(BIAS_ORDER, fontsize=7, color="white")
    ax2.tick_params(colors="white")
    ax2.xaxis.tick_top()
    cbar = ax2.collections[0].colorbar
    cbar.ax.yaxis.label.set_color("white")
    cbar.ax.tick_params(colors="white")
    for sp in ax2.spines.values(): sp.set_color("#4A9EFF")

    fig.suptitle("Semantic Similarity vs. Induction Effect (jury-confirmed, off-diagonal only)",
                 color="white", fontsize=12, y=1.01)
    fig.tight_layout()
    _savefig("semantic_similarity.png")

    return {"status": "ok", "spearman_r": rho, "spearman_p": p, "mantel_p": mantel_p,
            "n_pairs": len(pairs_df), "sim_matrix": sim_df}


# ══════════════════════════════════════════════════════════════════════════════
# 2. BOOTSTRAP CONFIDENCE INTERVALS
# ══════════════════════════════════════════════════════════════════════════════

def run_bootstrap_ci(jp, pivot, n_boot=2000):
    print("\n[2/7] Bootstrap confidence intervals …")
    ci_lo = np.full((10, 10), np.nan)
    ci_hi = np.full((10, 10), np.nan)
    ci_width = np.full((10, 10), np.nan)
    point   = np.full((10, 10), np.nan)

    rng = np.random.default_rng(42)
    for i, bias in enumerate(BIAS_ORDER):
        for j, hbias in enumerate(BIAS_ORDER):
            cell = jp[(jp["bias"]==bias) & (jp["human_bias"]==hbias)]["delta_score"].dropna().values
            if len(cell) < 2: continue
            boot_means = [rng.choice(cell, size=len(cell), replace=True).mean()
                          for _ in range(n_boot)]
            ci_lo[i, j]   = np.percentile(boot_means, 2.5)
            ci_hi[i, j]   = np.percentile(boot_means, 97.5)
            ci_width[i, j] = ci_hi[i, j] - ci_lo[i, j]
            point[i, j]   = cell.mean()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor=_DARK_BG)

    for ax, data, title, fmt, cmap, center in [
        (axes[0], point,    "Coupling Matrix M  (point estimates)", ".2f",
         sns.diverging_palette(220, 20, as_cmap=True), 0),
        (axes[1], ci_width, "Bootstrap 95% CI Width  (stability)",  ".2f",
         "YlOrRd", None),
    ]:
        ax.set_facecolor(_PANEL_BG)
        df_plot = pd.DataFrame(data, index=BIAS_ORDER, columns=BIAS_ORDER)
        kw = {"center": center} if center is not None else {}
        sns.heatmap(df_plot.round(3), cmap=cmap, annot=True, fmt=fmt,
                    annot_kws={"size": 7.5}, linewidths=0.4, linecolor="#0F172A",
                    cbar_kws={"shrink": 0.65}, ax=ax, **kw)
        ax.set_title(title, color="white", fontsize=10, pad=6)
        ax.set_xticklabels([b.replace(" ", "\n") for b in BIAS_ORDER],
                           rotation=0, ha="center", fontsize=7, color="white")
        ax.set_yticklabels(BIAS_ORDER, fontsize=7, color="white")
        ax.xaxis.tick_top()
        ax.tick_params(colors="white")
        for sp in ax.spines.values(): sp.set_color("#4A9EFF")
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(colors="white")

    fig.suptitle(f"Bootstrap Stability of Coupling Matrix  (n_boot={n_boot}, jury-confirmed)",
                 color="white", fontsize=12, y=1.01)
    fig.tight_layout()
    _savefig("bootstrap_ci.png")

    avg_width = np.nanmean(ci_width)
    stable = np.sum(ci_width < 0.5) if not np.all(np.isnan(ci_width)) else 0
    print(f"  Mean CI width: {avg_width:.3f}  |  Cells with width < 0.5: {stable}")
    return {"mean_ci_width": avg_width, "n_stable_cells": int(stable),
            "ci_width_matrix": ci_width}


# ══════════════════════════════════════════════════════════════════════════════
# 3. SVD / RANK ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def run_svd(pivot):
    print("\n[3/7] SVD / rank decomposition …")
    M = pivot.fillna(0).values.astype(float)
    U, S, Vt = np.linalg.svd(M)
    total_var = np.sum(S**2)
    explained = np.cumsum(S**2) / total_var

    print(f"  Singular values: {np.round(S, 4)}")
    print(f"  Variance explained by top-1: {explained[0]:.3f}, top-2: {explained[1]:.3f}, top-3: {explained[2]:.3f}")

    fig = _dark_fig((14, 5))
    gs  = fig.add_gridspec(1, 3, wspace=0.35)

    # Left: singular value spectrum
    ax0 = _dark_ax(fig.add_subplot(gs[0]))
    colors_sv = ["#4A9EFF" if i < 3 else "#8A93A8" for i in range(len(S))]
    ax0.bar(range(1, len(S)+1), S, color=colors_sv, edgecolor="none")
    ax0.set_xlabel("Singular value index", color="white", fontsize=9)
    ax0.set_ylabel("Magnitude", color="white", fontsize=9)
    ax0.set_title("Singular value spectrum\n(intrinsic rank of M)", color="white", fontsize=9)
    for i, (sv, ex) in enumerate(zip(S[:3], explained[:3])):
        ax0.text(i+1, sv + 0.01, f"{ex:.0%}", ha="center", fontsize=8, color="#F5C542")

    # Middle: top-1 right singular vector (how biases contribute as "primers")
    ax1 = _dark_ax(fig.add_subplot(gs[1]))
    v1 = Vt[0]
    cols1 = ["#4CC98A" if v > 0 else "#E55A4E" for v in v1]
    ax1.barh(BIAS_ORDER, v1, color=cols1, edgecolor="none")
    ax1.axvline(0, color="white", lw=0.8, ls="--")
    ax1.set_title("Top singular vector V₁\n(bias as primer direction)", color="white", fontsize=9)
    ax1.set_xlabel("Loading", color="white", fontsize=9)
    ax1.tick_params(colors="white", labelsize=7.5)

    # Right: top-1 left singular vector (how biases contribute as "targets")
    ax2 = _dark_ax(fig.add_subplot(gs[2]))
    u1 = U[:, 0]
    cols2 = ["#4CC98A" if v > 0 else "#E55A4E" for v in u1]
    ax2.barh(BIAS_ORDER, u1, color=cols2, edgecolor="none")
    ax2.axvline(0, color="white", lw=0.8, ls="--")
    ax2.set_title("Top singular vector U₁\n(bias as target direction)", color="white", fontsize=9)
    ax2.set_xlabel("Loading", color="white", fontsize=9)
    ax2.tick_params(colors="white", labelsize=7.5)

    fig.suptitle("SVD Decomposition of Coupling Matrix  M = UΣVᵀ",
                 color="white", fontsize=11, y=1.02)
    _savefig("svd_rank.png")
    return {"singular_values": S.tolist(), "var_explained_top3": float(explained[2])}


# ══════════════════════════════════════════════════════════════════════════════
# 4. TAXONOMY GROUPING TEST
# ══════════════════════════════════════════════════════════════════════════════

def run_taxonomy_test(jp):
    print("\n[4/7] Bias taxonomy grouping test …")
    within  = jp[jp["same_group"] == 1]["delta_score"].dropna()
    across  = jp[jp["same_group"] == 0]["delta_score"].dropna()
    diag    = jp[jp["is_diagonal"] == 1]["delta_score"].dropna()

    t_wa, p_wa = stats.ttest_ind(within, across)
    d_wa = (within.mean() - across.mean()) / (
        np.sqrt((within.std()**2 + across.std()**2) / 2) + 1e-9)

    print(f"  Within-group:  n={len(within):3d}  mean={within.mean():.4f}  std={within.std():.4f}")
    print(f"  Across-group:  n={len(across):3d}  mean={across.mean():.4f}  std={across.std():.4f}")
    print(f"  Diagonal:      n={len(diag):3d}  mean={diag.mean():.4f}")
    print(f"  Within vs Across: t={t_wa:.3f}  p={p_wa:.4f}  d={d_wa:.3f}")

    # Permutation test for within vs across
    obs_diff = within.mean() - across.mean()
    combined = np.concatenate([within.values, across.values])
    perm_diffs = []
    rng = np.random.default_rng(42)
    for _ in range(5000):
        shuffled = rng.permutation(combined)
        perm_diffs.append(shuffled[:len(within)].mean() - shuffled[len(within):].mean())
    perm_p = np.mean(np.abs(perm_diffs) >= abs(obs_diff))
    print(f"  Permutation p = {perm_p:.4f}  (n_perm=5000)")

    # Per-group mean delta heatmap
    group_pairs = {}
    for g_t in TAXONOMY:
        for g_h in TAXONOMY:
            sub = jp[(jp["bias_group"] == g_t) & (jp["hbias_group"] == g_h)]["delta_score"].dropna()
            group_pairs[(g_t, g_h)] = sub.mean() if len(sub) > 0 else np.nan

    groups = list(TAXONOMY.keys())
    gmat = pd.DataFrame(
        [[group_pairs.get((r, c), np.nan) for c in groups] for r in groups],
        index=groups, columns=groups,
    )
    short = {g: g.split(" & ")[0] for g in groups}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=_DARK_BG)

    # Left: group-level heatmap
    ax0 = _dark_ax(axes[0])
    sns.heatmap(gmat.round(3), cmap=sns.diverging_palette(220, 20, as_cmap=True),
                center=0, annot=True, fmt=".3f", annot_kws={"size": 12},
                linewidths=1, linecolor="#0F172A",
                cbar_kws={"shrink": 0.7, "label": "Mean Δ"}, ax=ax0)
    ax0.set_title("Group-level coupling matrix\n(mean Δ per taxonomy pair)",
                  color="white", fontsize=10, pad=6)
    ax0.set_xticklabels([short[g] for g in groups], color="white", fontsize=9, rotation=20)
    ax0.set_yticklabels([short[g] for g in groups], color="white", fontsize=9)
    ax0.xaxis.tick_top(); ax0.xaxis.set_label_position("top")
    ax0.tick_params(colors="white")
    cbar = ax0.collections[0].colorbar
    cbar.ax.tick_params(colors="white"); cbar.ax.yaxis.label.set_color("white")
    for sp in ax0.spines.values(): sp.set_color("#4A9EFF")

    # Right: violin — within vs across vs diagonal
    ax1 = _dark_ax(axes[1])
    plot_df = pd.DataFrame({
        "delta": np.concatenate([within, across, diag]),
        "type":  (["Within-group"]*len(within) +
                  ["Across-group"]*len(across) +
                  ["Diagonal\n(same bias)"]*len(diag)),
    })
    sns.violinplot(data=plot_df, x="type", y="delta",
                   palette={"Within-group": "#4A9EFF", "Across-group": "#8A93A8",
                             "Diagonal\n(same bias)": "#F5C542"},
                   inner="box", ax=ax1)
    ax1.axhline(0, color="white", lw=0.8, ls="--", alpha=0.6)
    for xp, val, label in [(0, within.mean(), f"μ={within.mean():.3f}"),
                            (1, across.mean(), f"μ={across.mean():.3f}"),
                            (2, diag.mean(),   f"μ={diag.mean():.3f}")]:
        ax1.text(xp, val + 0.03, label, ha="center", fontsize=9, color="#F5C542")
    ax1.set_xlabel("", color="white"); ax1.set_ylabel("Δ-score", color="white", fontsize=9)
    ax1.tick_params(colors="white")
    ax1.set_title(f"Within vs. across taxonomy group\nPermutation p={perm_p:.3f}  d={d_wa:.3f}",
                  color="white", fontsize=10, pad=6)
    for sp in ax1.spines.values(): sp.set_color("#4A9EFF")

    fig.tight_layout()
    _savefig("taxonomy_test.png")
    return {"within_mean": float(within.mean()), "across_mean": float(across.mean()),
            "t": float(t_wa), "p": float(p_wa), "d": float(d_wa), "perm_p": float(perm_p)}


# ══════════════════════════════════════════════════════════════════════════════
# 5. MESSAGE QUALITY REGRESSION
# ══════════════════════════════════════════════════════════════════════════════

def run_message_quality_regression(jp):
    print("\n[5/7] Message quality regression …")
    feats = ["jury_target_rating", "jury_purity", "jury_naturalness",
             "sim_assertiveness", "sim_emotional_intensity", "sim_sentiment", "sim_word_count"]
    feats = [f for f in feats if f in jp.columns]

    sub = jp[feats + ["delta_score", "bias", "human_bias", "scenario_id"]].dropna().copy()
    sub["bias_code"]  = pd.Categorical(sub["bias"],       categories=BIAS_ORDER).codes.astype(float)
    sub["hbias_code"] = pd.Categorical(sub["human_bias"], categories=BIAS_ORDER).codes.astype(float)

    formula = "delta_score ~ " + " + ".join(feats) + " + C(bias_code) + C(hbias_code)"
    model = smf.ols(formula, data=sub).fit(cov_type="HC3")
    print(f"  R² = {model.rsquared:.4f}  adj-R² = {model.rsquared_adj:.4f}  n = {len(sub)}")

    coef_df = pd.DataFrame({
        "coef": model.params[feats],
        "se":   model.bse[feats],
        "p":    model.pvalues[feats],
        "ci_lo": model.conf_int().loc[feats, 0],
        "ci_hi": model.conf_int().loc[feats, 1],
    })
    for _, row in coef_df.iterrows():
        sig = "**" if row["p"] < 0.05 else "*" if row["p"] < 0.10 else ""
        print(f"  {row.name:30s}  coef={row['coef']:+.4f}  p={row['p']:.4f} {sig}")

    fig = _dark_fig((9, 5))
    ax  = _dark_ax(fig.add_subplot(111))
    y   = np.arange(len(coef_df))
    colors = ["#4CC98A" if p < 0.05 else "#F5C542" if p < 0.10 else "#8A93A8"
              for p in coef_df["p"]]
    ax.barh(y, coef_df["coef"],
            xerr=[coef_df["coef"] - coef_df["ci_lo"], coef_df["ci_hi"] - coef_df["coef"]],
            color=colors, height=0.55, edgecolor="none",
            error_kw={"ecolor": "white", "capsize": 4, "lw": 1.2})
    ax.axvline(0, color="white", lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([f.replace("sim_","sim ").replace("jury_","jury ") for f in coef_df.index],
                       fontsize=10, color="white")
    ax.set_xlabel("OLS coefficient  (95% CI, HC3 robust SE)", color="white", fontsize=10)
    ax.set_title(f"Does stronger bias signal → higher Δ?\n"
                 f"R² = {model.rsquared:.3f}  (target & human bias FEs included)",
                 color="white", fontsize=10, pad=6)
    legend_els = [mpatches.Patch(color="#4CC98A", label="p<0.05"),
                  mpatches.Patch(color="#F5C542", label="p<0.10"),
                  mpatches.Patch(color="#8A93A8", label="n.s.")]
    ax.legend(handles=legend_els, fontsize=9, facecolor=_DARK_BG,
              edgecolor="#4A9EFF", labelcolor="white")
    fig.tight_layout()
    _savefig("message_quality_regression.png")
    return {"r2": float(model.rsquared), "coefs": coef_df.to_dict()}


# ══════════════════════════════════════════════════════════════════════════════
# 6. ADVERSARIAL NEUTRAL CHECK
# ══════════════════════════════════════════════════════════════════════════════

def run_neutral_check(jp):
    print("\n[6/7] Adversarial neutral check …")
    ns = jp["neutral_score"].dropna()
    t, p = stats.ttest_1samp(ns, 0)
    print(f"  neutral_score: mean={ns.mean():.4f}  std={ns.std():.4f}  t={t:.3f}  p={p:.4f}")
    print(f"  Interpretation: {'CONTAMINATED — neutral baseline ≠ 0' if p < 0.05 else 'CLEAN — neutral baseline ≈ 0'}")

    # Per-bias neutral score
    per_bias = jp.groupby("bias")["neutral_score"].agg(["mean","sem","count"]).reindex(BIAS_ORDER)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=_DARK_BG)

    ax0 = _dark_ax(axes[0])
    ax0.hist(ns, bins=25, color="#4A9EFF", edgecolor="none", alpha=0.85)
    ax0.axvline(0,         color="white",   lw=1.5, ls="--", label="zero (ideal)")
    ax0.axvline(ns.mean(), color="#F5C542", lw=1.5, ls="-",  label=f"mean={ns.mean():.3f}")
    ax0.set_xlabel("neutral_score", color="white", fontsize=10)
    ax0.set_ylabel("Count", color="white", fontsize=10)
    ax0.set_title(f"Neutral score distribution\nt={t:.3f}  p={p:.4f}  ({'≠ 0 !' if p<0.05 else '≈ 0 ✓'})",
                  color="white", fontsize=10, pad=6)
    ax0.legend(fontsize=9, facecolor=_DARK_BG, edgecolor="#4A9EFF", labelcolor="white")
    ax0.tick_params(colors="white")

    ax1 = _dark_ax(axes[1])
    y = np.arange(len(BIAS_ORDER))
    colors = ["#E55A4E" if abs(m) > 0.1 else "#4A9EFF"
              for m in per_bias["mean"].fillna(0)]
    ax1.barh(y, per_bias["mean"], xerr=per_bias["sem"],
             color=colors, height=0.6, edgecolor="none",
             error_kw={"ecolor": "white", "capsize": 3, "lw": 1})
    ax1.axvline(0, color="white", lw=0.8, ls="--")
    ax1.set_yticks(y); ax1.set_yticklabels(BIAS_ORDER, fontsize=9, color="white")
    ax1.set_xlabel("Mean neutral_score  ±SE", color="white", fontsize=10)
    ax1.set_title("Per-target neutral score\n(red = |mean| > 0.1, possible contamination)",
                  color="white", fontsize=10, pad=6)
    ax1.tick_params(colors="white")

    fig.suptitle("Adversarial Neutral Check — Is the neutral baseline truly neutral?",
                 color="white", fontsize=11, y=1.01)
    fig.tight_layout()
    _savefig("neutral_check.png")
    return {"neutral_mean": float(ns.mean()), "neutral_t": float(t), "neutral_p": float(p)}


# ══════════════════════════════════════════════════════════════════════════════
# 7. DIRECTED NETWORK GRAPH
# ══════════════════════════════════════════════════════════════════════════════

def run_network_graph(jp, pivot, effect_threshold=0.08):
    print("\n[7/7] Directed network graph …")
    G = nx.DiGraph()
    G.add_nodes_from(BIAS_ORDER)

    edges_added = 0
    for bias in BIAS_ORDER:
        for hbias in BIAS_ORDER:
            if bias == hbias: continue
            cell = jp[(jp["bias"]==bias) & (jp["human_bias"]==hbias)]["delta_score"].dropna()
            if len(cell) < 2: continue
            mean_d = cell.mean()
            if abs(mean_d) < effect_threshold: continue
            G.add_edge(hbias, bias, weight=mean_d, n=len(cell))
            edges_added += 1

    print(f"  Edges shown (|mean Δ| ≥ {effect_threshold}): {edges_added}")

    # Node positions — circular layout, colour by group
    pos = nx.circular_layout(G, scale=2.5)
    node_colors = [GROUP_COLORS.get(BIAS_TO_GROUP.get(b, ""), "#8A93A8") for b in G.nodes()]

    fig = _dark_fig((11, 9))
    ax  = fig.add_subplot(111)
    ax.set_facecolor(_PANEL_BG)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1600,
                           alpha=0.95, ax=ax)
    # Labels inside nodes
    nx.draw_networkx_labels(G, pos,
                            labels={b: b.replace(" ", "\n") for b in G.nodes()},
                            font_size=6.5, font_color="white", ax=ax)

    # Draw edges coloured by direction, sized by magnitude
    amp_edges  = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] > 0]
    supp_edges = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] < 0]

    def edge_widths(elist):
        return [min(abs(G[u][v]["weight"]) * 8, 5) for u, v in elist]

    nx.draw_networkx_edges(G, pos, edgelist=amp_edges,
                           edge_color="#4CC98A", width=edge_widths(amp_edges),
                           alpha=0.8, arrows=True,
                           arrowstyle="-|>", arrowsize=18,
                           connectionstyle="arc3,rad=0.12", ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=supp_edges,
                           edge_color="#E55A4E", width=edge_widths(supp_edges),
                           alpha=0.8, arrows=True,
                           arrowstyle="-|>", arrowsize=18,
                           connectionstyle="arc3,rad=0.12", ax=ax)

    ax.set_xlim(-3.8, 3.8); ax.set_ylim(-3.8, 3.8)
    ax.axis("off")

    # Legend
    legend_els = (
        [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()] +
        [plt.Line2D([0],[0], color="#4CC98A", lw=2.5, label=f"Amplification (Δ > {effect_threshold})"),
         plt.Line2D([0],[0], color="#E55A4E", lw=2.5, label=f"Suppression  (Δ < −{effect_threshold})")]
    )
    ax.legend(handles=legend_els, fontsize=8.5, facecolor=_DARK_BG,
              edgecolor="#4A9EFF", labelcolor="white", loc="lower right")

    fig.suptitle(f"Bias Induction Network  (|mean Δ| ≥ {effect_threshold}, jury-confirmed)\n"
                 f"Arrows: human bias → target bias direction",
                 color="white", fontsize=11, y=0.98)
    _savefig("network_graph.png")
    return {"n_edges": edges_added, "threshold": effect_threshold}


# ══════════════════════════════════════════════════════════════════════════════
# 8. VARIANCE DECOMPOSITION
# ══════════════════════════════════════════════════════════════════════════════

def run_variance_decomposition(jp):
    """
    How much of individual_score is explained by neutral_score alone?

    individual_score = effect of (biased scenario + biased human) vs baseline
    neutral_score    = effect of (biased scenario + neutral human) vs baseline

    R² of individual ~ neutral tells us what fraction of the total human
    influence is attributable to the mere presence of any human turn.
    1 - R² is the fraction specific to the biased content of the message.
    """
    print("\n[8/9] Variance decomposition …")

    sub = jp[["individual_score", "neutral_score", "bias", "human_bias"]].dropna().copy()

    # Global OLS: individual_score ~ neutral_score
    global_model = smf.ols("individual_score ~ neutral_score", data=sub).fit()
    r2    = global_model.rsquared
    coef  = global_model.params["neutral_score"]
    p_val = global_model.pvalues["neutral_score"]
    print(f"  Global R² = {r2:.4f}  →  {(1 - r2) * 100:.1f}% of individual_score variance is bias-content specific")
    print(f"  neutral_score coefficient = {coef:.4f}  p = {p_val:.4f}")

    # Per-target-bias R²
    per_bias_r2 = {}
    for bias in BIAS_ORDER:
        s = sub[sub["bias"] == bias]
        if len(s) < 4:
            per_bias_r2[bias] = np.nan
            continue
        try:
            m = smf.ols("individual_score ~ neutral_score", data=s).fit()
            per_bias_r2[bias] = m.rsquared
        except Exception:
            per_bias_r2[bias] = np.nan
        print(f"  {bias:30s}  R²={per_bias_r2[bias]:.3f}" if not np.isnan(per_bias_r2[bias]) else
              f"  {bias:30s}  R²=n/a")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=_DARK_BG)

    # Left: scatter individual_score vs neutral_score, coloured by taxonomy group
    ax0 = _dark_ax(axes[0])
    group_cols = [GROUP_COLORS.get(BIAS_TO_GROUP.get(b, ""), "#8A93A8")
                  for b in sub["bias"]]
    ax0.scatter(sub["neutral_score"], sub["individual_score"],
                c=group_cols, s=45, alpha=0.75, edgecolors="none", zorder=3)
    xs = np.linspace(sub["neutral_score"].min(), sub["neutral_score"].max(), 100)
    ax0.plot(xs,
             global_model.params["Intercept"] + coef * xs,
             color="#F5C542", lw=2.2, zorder=4,
             label=f"OLS: coef={coef:.3f}  p={p_val:.4f}")
    ax0.axhline(0, color="white", lw=0.7, ls="--", alpha=0.5)
    ax0.axvline(0, color="white", lw=0.7, ls="--", alpha=0.5)
    ax0.set_xlabel("neutral_score  (any human turn effect)", color="white", fontsize=10)
    ax0.set_ylabel("individual_score  (total human effect)", color="white", fontsize=10)
    ax0.set_title(
        f"Variance decomposition\n"
        f"R²={r2:.3f}  —  {(1 - r2) * 100:.1f}% of individual_score is bias-content specific",
        color="white", fontsize=10, pad=6)
    legend_els = ([mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()] +
                  [plt.Line2D([0], [0], color="#F5C542", lw=2,
                               label=f"OLS R²={r2:.3f}  p={p_val:.4f}")])
    ax0.legend(handles=legend_els, fontsize=8, facecolor=_DARK_BG,
               edgecolor="#4A9EFF", labelcolor="white")

    # Right: per-target-bias R² bar chart
    ax1 = _dark_ax(axes[1])
    r2_vals = pd.Series(per_bias_r2).reindex(BIAS_ORDER)
    colors  = [GROUP_COLORS.get(BIAS_TO_GROUP.get(b, ""), "#8A93A8") for b in BIAS_ORDER]
    y_pos   = np.arange(len(BIAS_ORDER))
    ax1.barh(y_pos, r2_vals.fillna(0), color=colors, height=0.6, edgecolor="none")
    ax1.axvline(r2, color="#F5C542", lw=1.5, ls="--", label=f"Global R²={r2:.3f}")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(BIAS_ORDER, fontsize=9, color="white")
    ax1.set_xlabel("R²  (individual_score ~ neutral_score)", color="white", fontsize=10)
    ax1.set_title("Per-target bias: how much of the human effect\nis explained by neutral human presence alone",
                  color="white", fontsize=10, pad=6)
    ax1.set_xlim(0, 1)
    ax1.legend(fontsize=9, facecolor=_DARK_BG, edgecolor="#4A9EFF", labelcolor="white")
    ax1.tick_params(colors="white")

    fig.suptitle(
        "Variance Decomposition — What fraction of human influence is bias-content specific?",
        color="white", fontsize=11, y=1.01)
    fig.tight_layout()
    _savefig("variance_decomposition.png")

    return {"r2_global": r2, "coef": coef, "p": p_val,
            "bias_specific_pct": float((1 - r2) * 100),
            **{f"r2_{b.replace(' ', '_')}": v for b, v in per_bias_r2.items()}}


# ══════════════════════════════════════════════════════════════════════════════
# 9. NEUTRAL SCORE AS LMM MODERATOR
# ══════════════════════════════════════════════════════════════════════════════

def run_neutral_moderator_lmm(jp):
    """
    Does neutral_score moderate the bias-specific induction effect?

    LMM-1 (baseline): delta_score ~ C(human_bias) + (1|scenario_id)
    LMM-2 (full):     delta_score ~ C(human_bias) + neutral_score + (1|scenario_id)

    If neutral_score is significant in LMM-2, scenarios that are more sensitive
    to any human input also show larger bias-specific induction — meaning
    conversational sensitivity moderates the coupling effect.
    AIC comparison quantifies whether adding neutral_score improves model fit.
    """
    print("\n[9/9] Neutral score as LMM moderator …")

    sub = jp[["delta_score", "neutral_score", "human_bias", "bias",
              "scenario_id"]].dropna().copy()
    sub["group"] = sub["scenario_id"]  # random intercept grouping

    def _fit_lmm(formula):
        try:
            return smf.mixedlm(formula, sub, groups=sub["group"]).fit(
                reml=False, method="lbfgs")
        except Exception as e:
            print(f"  LMM fit failed ({formula}): {e} — using OLS fallback")
            return smf.ols(formula, data=sub).fit(cov_type="HC3")

    m1 = _fit_lmm("delta_score ~ C(human_bias)")
    m2 = _fit_lmm("delta_score ~ C(human_bias) + neutral_score")

    aic1 = m1.aic
    aic2 = m2.aic
    delta_aic = aic1 - aic2

    coef  = float(m2.params.get("neutral_score", np.nan))
    se    = float(m2.bse.get("neutral_score", np.nan))
    p_val = float(m2.pvalues.get("neutral_score", np.nan))
    ci_lo = coef - 1.96 * se
    ci_hi = coef + 1.96 * se

    print(f"  LMM-1 (baseline):  AIC = {aic1:.2f}")
    print(f"  LMM-2 (+neutral):  AIC = {aic2:.2f}  ΔAIC = {delta_aic:.2f}")
    print(f"  neutral_score coef = {coef:.4f}  SE = {se:.4f}  p = {p_val:.4f}")
    verdict = ("MODERATOR — neutral sensitivity predicts delta"
               if p_val < 0.05 else
               "NOT a significant moderator  (delta is independent of neutral contamination)")
    print(f"  → {verdict}")

    # Marginal regression for scatter (delta ~ neutral, ignoring human_bias FE)
    marginal = smf.ols("delta_score ~ neutral_score", data=sub).fit()
    m_coef   = marginal.params["neutral_score"]
    m_p      = marginal.pvalues["neutral_score"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=_DARK_BG)

    # Left: scatter delta_score vs neutral_score
    ax0 = _dark_ax(axes[0])
    group_cols = [GROUP_COLORS.get(BIAS_TO_GROUP.get(b, ""), "#8A93A8")
                  for b in sub["bias"]]
    ax0.scatter(sub["neutral_score"], sub["delta_score"],
                c=group_cols, s=45, alpha=0.75, edgecolors="none", zorder=3)
    xs = np.linspace(sub["neutral_score"].min(), sub["neutral_score"].max(), 100)
    ax0.plot(xs,
             marginal.params["Intercept"] + m_coef * xs,
             color="#F5C542", lw=2.2, zorder=4,
             label=f"Marginal OLS: coef={m_coef:.3f}  p={m_p:.4f}")
    ax0.axhline(0, color="white", lw=0.7, ls="--", alpha=0.5)
    ax0.axvline(0, color="white", lw=0.7, ls="--", alpha=0.5)
    ax0.set_xlabel("neutral_score  (baseline contamination)", color="white", fontsize=10)
    ax0.set_ylabel("delta_score  (bias-specific induction)", color="white", fontsize=10)
    ax0.set_title("Neutral score vs delta score\nDoes conversational sensitivity predict induction strength?",
                  color="white", fontsize=10, pad=6)
    legend_els = ([mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()] +
                  [plt.Line2D([0], [0], color="#F5C542", lw=2,
                               label=f"Marginal coef={m_coef:.3f}  p={m_p:.4f}")])
    ax0.legend(handles=legend_els, fontsize=8, facecolor=_DARK_BG,
               edgecolor="#4A9EFF", labelcolor="white")

    # Right: LMM coefficient plot + AIC table
    ax1 = _dark_ax(axes[1])
    col = "#4CC98A" if p_val < 0.05 else "#F5C542" if p_val < 0.10 else "#8A93A8"
    ax1.barh([0], [coef], color=col, height=0.35, edgecolor="none")
    ax1.errorbar([coef], [0],
                 xerr=[[coef - ci_lo], [ci_hi - coef]],
                 fmt="none", ecolor="white", capsize=7, lw=1.5)
    ax1.axvline(0, color="white", lw=1, ls="--")
    ax1.set_yticks([0])
    ax1.set_yticklabels(["neutral_score"], fontsize=11, color="white")
    ax1.set_xlabel("LMM coefficient  (95% CI)", color="white", fontsize=10)
    ax1.set_title(
        f"LMM-2: delta ~ human_bias + neutral_score + (1|scenario)\n"
        f"coef={coef:.4f}  p={p_val:.4f}  ΔAIC={delta_aic:.1f}",
        color="white", fontsize=10, pad=6)

    table_txt = (f"  Model          AIC\n"
                 f"  LMM-1 (base)   {aic1:.1f}\n"
                 f"  LMM-2 (+neut)  {aic2:.1f}\n"
                 f"  ΔAIC           {delta_aic:.1f}\n\n"
                 f"  {'★ neutral_score improves fit' if delta_aic > 2 else '✗ no fit improvement'}")
    ax1.text(0.04, 0.22, table_txt, transform=ax1.transAxes,
             fontsize=9.5, color="white", family="monospace",
             bbox=dict(facecolor="#1E293B", edgecolor="#4A9EFF", alpha=0.9, pad=7))

    legend_els2 = [mpatches.Patch(color="#4CC98A", label="p < 0.05"),
                   mpatches.Patch(color="#F5C542", label="p < 0.10"),
                   mpatches.Patch(color="#8A93A8", label="n.s.")]
    ax1.legend(handles=legend_els2, fontsize=9, facecolor=_DARK_BG,
               edgecolor="#4A9EFF", labelcolor="white", loc="lower right")

    fig.suptitle(
        "Neutral Score as Moderator — Does conversational sensitivity predict bias-specific induction?",
        color="white", fontsize=11, y=1.01)
    fig.tight_layout()
    _savefig("neutral_moderator.png")

    return {"lmm_coef": coef, "lmm_se": se, "lmm_p": p_val,
            "aic_baseline": aic1, "aic_full": aic2, "delta_aic": delta_aic}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(pilot_csv=_PILOT_CSV, skip_semantic=False):
    os.makedirs(_PLOT_DIR, exist_ok=True)
    jp, pivot = load(pilot_csv)
    results = {}

    if not skip_semantic:
        results["semantic"] = run_semantic_similarity(jp, pivot)
    else:
        print("\n[1/9] Semantic similarity — skipped (--skip_semantic)")
        results["semantic"] = {"status": "skipped"}

    results["bootstrap"]    = run_bootstrap_ci(jp, pivot)
    results["svd"]          = run_svd(pivot)
    results["taxonomy"]     = run_taxonomy_test(jp)
    results["regression"]   = run_message_quality_regression(jp)
    results["neutral"]      = run_neutral_check(jp)
    results["network"]      = run_network_graph(jp, pivot)
    results["variance"]     = run_variance_decomposition(jp)
    results["neutral_mod"]  = run_neutral_moderator_lmm(jp)

    # Summary CSV
    summary_rows = []
    for k, v in results.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                if np.isscalar(vv) and not isinstance(vv, (dict, list, np.ndarray, pd.DataFrame)):
                    summary_rows.append({"analysis": k, "metric": kk, "value": vv})
    pd.DataFrame(summary_rows).to_csv(_SUMMARY_CSV, index=False)

    print(f"\nAll extended analyses complete.")
    print(f"Plots   → {_PLOT_DIR}")
    print(f"Summary → {_SUMMARY_CSV}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=_PILOT_CSV)
    parser.add_argument("--skip_semantic", action="store_true",
                        help="Skip semantic similarity (needs OPENAI_API_KEY)")
    args = parser.parse_args()
    main(args.input, skip_semantic=args.skip_semantic)
