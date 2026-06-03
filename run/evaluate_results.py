"""
evaluate_results.py — All result calculations for the conditional bias pilot.

Filter: only rows where jury_passed=True (human message confirmed as genuinely biased).

Outputs:
  run/data/plots/heatmap_individual_jp.png   — individual_score heatmap
  run/data/plots/heatmap_neutral_jp.png      — neutral_score heatmap
  run/data/plots/heatmap_delta_jp.png        — delta_score heatmap
  run/data/plots/diagonal_distribution_jp.png
  run/data/plots/score_distributions_jp.png
  run/data/conditional_decision_results/significance_results_jp.csv
  run/data/conditional_decision_results/causal_summary_jp.csv
  run/data/plots/causal/                     — causal inference figures (updated)

Usage:
    python evaluate_results.py
    python evaluate_results.py --input path/to/pilot.csv
"""

import os
import sys
import warnings
import itertools

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    "font.size":        8,
    "axes.titlesize":   8,
    "axes.labelsize":   8,
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
    "legend.fontsize":  7,
    "figure.titlesize": 8,
})
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import statsmodels.formula.api as smf

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from conditional_test_decision import _compute_individual_scores

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
_RUN_DIR   = os.path.dirname(os.path.abspath(__file__))
_PILOT_CSV = os.path.join(_RUN_DIR, "data", "conditional_decision_results", "GPT-4o_pilot.csv")
_BANK_CSV  = os.path.join(_RUN_DIR, "data", "validated_message_bank.csv")
_PLOT_DIR  = os.path.join(_RUN_DIR, "data", "plots")
_CAUSAL_PLOT_DIR = os.path.join(_PLOT_DIR, "causal")
_SIG_CSV   = os.path.join(_RUN_DIR, "data", "conditional_decision_results", "significance_results_jp.csv")
_CAUSAL_CSV = os.path.join(_RUN_DIR, "data", "conditional_decision_results", "causal_summary_jp.csv")

_BIAS_ORDER = [
    "Anchoring", "Availability Heuristic", "Bandwagon Effect",
    "Confirmation Bias", "Framing Effect", "In Group Bias",
    "Loss Aversion", "Planning Fallacy", "Status Quo Bias",
]

def _savefig(name: str, subdir: str = None):
    d = os.path.join(_PLOT_DIR, subdir) if subdir else _PLOT_DIR
    os.makedirs(d, exist_ok=True)
    pdf_name = os.path.splitext(name)[0] + ".pdf"
    path = os.path.join(d, pdf_name)
    plt.savefig(path, format="pdf", dpi=600, bbox_inches="tight")
    plt.close()
    print(f"  saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data(path: str, bank_csv: str = _BANK_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    total   = len(df)
    ok      = df[df["status"] == "OK"]
    jp      = ok[ok["jury_passed"] == True].copy()
    discarded = total - len(jp)
    print(f"  Total rows      : {total}")
    print(f"  status=OK       : {len(ok)}")
    print(f"  jury_passed=True: {len(jp)}  (discarded {discarded} jury-failed rows)")

    # Back-compat: rename old column names to new schema
    if "individual_score" in jp.columns and "score_biased" not in jp.columns:
        print("  Renaming old schema columns to new schema …")
        jp = jp.rename(columns={
            "individual_score": "score_biased",
            "neutral_score":    "score_neutral",
        })

    # Derive presence/total effect columns if not already stored
    # Effects are differences of absolute bias magnitudes: |m_b| - |m_n|, etc.
    if "score_no_human" in jp.columns:
        if "effect_neutral" not in jp.columns:
            jp["effect_neutral"] = jp["score_neutral"].abs() - jp["score_no_human"].abs()
        if "effect_biased" not in jp.columns:
            jp["effect_biased"] = jp["score_biased"].abs() - jp["score_no_human"].abs()
    # Recompute delta_score as |m_b| - |m_n| for consistency
    if "score_biased" in jp.columns and "score_neutral" in jp.columns:
        jp["delta_score"] = jp["score_biased"].abs() - jp["score_neutral"].abs()

    # Compute scores if missing
    if "score_biased" not in jp.columns or jp["score_biased"].isna().all():
        print("  Computing scores from bank …")
        bank = pd.read_csv(bank_csv)
        bank["id"] = bank["scenario_id"]
        jp = jp.copy()
        jp["id"] = jp["scenario_id"]
        rows = jp.to_dict("records")
        rows = _compute_individual_scores(rows, bank)
        jp = pd.DataFrame(rows)
        jp = jp[jp["status"] == "OK"].copy()
        scored = jp["score_biased"].notna().sum() if "score_biased" in jp.columns else 0
        print(f"  Scored {scored}/{len(jp)} rows")

    if "scenario_id" not in jp.columns or jp["scenario_id"].isna().all():
        jp["scenario_id"] = jp["id"].astype(str) + "_" + jp["bias"]
    else:
        jp["scenario_id"] = jp["scenario_id"].astype(str) + "_" + jp["bias"]
    jp["is_diagonal"] = (jp["bias"] == jp["human_bias"]).astype(int)
    return jp


# ══════════════════════════════════════════════════════════════════════════════
# 1. HEATMAPS
# ══════════════════════════════════════════════════════════════════════════════

# Abbreviated bias labels for compact (small) heatmaps
_SHORT = {
    "Anchoring":             "Anchor.",
    "Availability Heuristic":"Avail.",
    "Bandwagon Effect":      "Bandwgn",
    "Confirmation Bias":     "Confirm.",
    "Framing Effect":        "Framing",
    "In Group Bias":         "In-Grp",
    "Loss Aversion":         "Loss Av.",
    "Planning Fallacy":      "Plan.F.",
    "Status Quo Bias":       "St.Quo",
    "Average":               "Avg",
}


def _impute_pivot(pivot: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN cells with row_mean + col_mean - grand_mean (bilinear imputation)."""
    if not pivot.isnull().any().any():
        return pivot
    out = pivot.copy().astype(float)
    grand = out.stack().mean()
    row_means = out.mean(axis=1)
    col_means = out.mean(axis=0)
    for i in out.index:
        for j in out.columns:
            if pd.isna(out.loc[i, j]):
                out.loc[i, j] = row_means[i] + col_means[j] - grand
    return out


def _heatmap(pivot: pd.DataFrame, title: str, fname: str, vmin=-1, vmax=1,
             figsize=(5.7, 4.5)):
    """Full-size annotated heatmap for appendix figures (~0.88 textwidth display)."""
    short_labels = [_SHORT.get(b, b) for b in pivot.index]
    short_cols   = [_SHORT.get(b, b) for b in pivot.columns]
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot.round(3),
        cmap=sns.diverging_palette(220, 20, as_cmap=True),
        center=0, annot=True, fmt=".2f",
        square=True,
        vmin=vmin, vmax=vmax,
        linewidths=0.4, linecolor="white",
        cbar_kws={"shrink": 0.7, "label": "Score"},
        ax=ax,
    )
    ax.set_xticklabels(short_cols, rotation=45, ha="left")
    ax.set_yticklabels(short_labels, rotation=0)
    ax.set_title(title, pad=10)
    ax.set_xlabel("Human Bias", labelpad=8)
    ax.set_ylabel("Target Bias", labelpad=8)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    _savefig(fname)


def plot_triptych_combined(df: pd.DataFrame):
    """Single combined 3-panel figure for the main-text triptych.

    Generated at textwidth (6.5") so 11pt fonts render correctly in the PDF.
    No cell annotations (colours convey the pattern; exact values are in appendix).
    """
    cols = [
        ("effect_neutral", "Presence\n$|m_n|-|m_\\varnothing|$"),
        ("delta_score",    "Content\n$|m_b|-|m_n|$"),
        ("effect_biased",  "Total\n$|m_b|-|m_\\varnothing|$"),
    ]
    available = [(c, t) for c, t in cols if c in df.columns]
    if not available:
        print("    triptych: required columns missing, skipping")
        return

    fig, axes = plt.subplots(1, len(available), figsize=(6.5, 3.2))
    if len(available) == 1:
        axes = [axes]
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    for i, (ax, (col, title)) in enumerate(zip(axes, available)):
        pivot = df.pivot_table(values=col, index="bias", columns="human_bias",
                               aggfunc="mean").reindex(index=_BIAS_ORDER, columns=_BIAS_ORDER)
        pivot.index   = [_SHORT.get(b, b) for b in pivot.index]
        pivot.columns = [_SHORT.get(b, b) for b in pivot.columns]
        sns.heatmap(
            pivot.round(2), cmap=cmap, center=0, annot=False,
            vmin=-1, vmax=1, linewidths=0.3, linecolor="white",
            cbar=True, cbar_kws={"shrink": 0.9, "aspect": 20}, ax=ax,
        )
        ax.set_title(title, pad=6)
        ax.set_xlabel("Human Bias", labelpad=4)
        ax.set_ylabel("Target Bias" if i == 0 else "", labelpad=4)
        ax.xaxis.tick_top(); ax.xaxis.set_label_position("top")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="left")
        plt.setp(ax.get_yticklabels(), rotation=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout(w_pad=0.3)
    _savefig("heatmap_triptych_jp.pdf")
    print("    triptych combined figure saved")


def plot_heatmaps(df: pd.DataFrame):
    print("\n[1/4] Heatmaps …")

    # ── No-human bar chart (human_bias axis is meaningless here) ─────────────
    if "score_no_human" in df.columns:
        means = (
            df.groupby("bias")["score_no_human"]
            .mean()
            .reindex(_BIAS_ORDER)
        )
        sems = (
            df.groupby("bias")["score_no_human"]
            .sem()
            .reindex(_BIAS_ORDER)
        )
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        colors = ["tomato" if v > 0 else "steelblue" for v in means]
        ax.bar(range(len(_BIAS_ORDER)), means.values, yerr=sems.values,
               color=colors, alpha=0.85, edgecolor="black", capsize=4, width=0.6)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(len(_BIAS_ORDER)))
        ax.set_xticklabels([_SHORT.get(b, b) for b in _BIAS_ORDER])
        ax.set_ylabel("Mean Bias Score")
        ax.set_title("Intrinsic Bias Score — No Human Message\n(±1 SEM, jury-confirmed rows only)")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        plt.tight_layout()
        _savefig("barchart_no_human_jp.png")
        print(f"    score_no_human: global mean = {df['score_no_human'].mean():.4f}")

    # ── Raw score heatmaps (neutral, biased, delta) ───────────────────────────
    for score_col, label, fname in [
        ("score_neutral",  "Score: Neutral Human Message\n(ctrl vs treat, unbiased human)", "heatmap_neutral_jp.png"),
        ("score_biased",   "Score: Biased Human Message\n(ctrl vs treat, biased human, jury-confirmed)", "heatmap_biased_jp.png"),
        ("delta_score",    "Delta Score  =  biased − neutral\n(jury-confirmed rows only)", "heatmap_delta_jp.png"),
    ]:
        if score_col not in df.columns:
            print(f"    {score_col}: column missing, skipping")
            continue
        pivot = _impute_pivot(df.pivot_table(
            values=score_col, index="bias", columns="human_bias", aggfunc="mean"
        ).reindex(index=_BIAS_ORDER, columns=_BIAS_ORDER))

        pivot_with_avg = pivot.copy()
        pivot_with_avg["Average"] = pivot.mean(axis=1)
        avg_row = pivot_with_avg.mean(axis=0)
        pivot_with_avg.loc["Average"] = avg_row

        _heatmap(pivot_with_avg, label, fname)
        print(f"    {score_col}: global mean = {df[score_col].mean():.4f}")

    # ── Presence effect and total effect heatmaps ────────────────────────────
    # effect_neutral = score_neutral - score_no_human (presence effect)
    # effect_biased  = score_biased  - score_no_human (total effect)
    if {"effect_neutral", "effect_biased"}.issubset(df.columns):
        for score_col, label, fname in [
            ("effect_neutral",
             "Presence Effect  ($m_n - m_\\varnothing$)\n"
             "Positive = neutral human elevates LLM bias above no-human baseline",
             "heatmap_effect_neutral_jp.png"),
            ("effect_biased",
             "Total Effect  ($m_b - m_\\varnothing$)\n"
             "Positive = biased human elevates LLM bias above no-human baseline",
             "heatmap_effect_biased_jp.png"),
        ]:
            pivot = _impute_pivot(df.pivot_table(
                values=score_col, index="bias", columns="human_bias", aggfunc="mean"
            ).reindex(index=_BIAS_ORDER, columns=_BIAS_ORDER))
            pivot_with_avg = pivot.copy()
            pivot_with_avg["Average"] = pivot.mean(axis=1)
            pivot_with_avg.loc["Average"] = pivot_with_avg.mean(axis=0)
            _heatmap(pivot_with_avg, label, fname)
            print(f"    {score_col}: global mean = {df[score_col].mean():.4f}")

        # Combined triptych figure for main text
        plot_triptych_combined(df)
        # Bar chart: global per-bias means for all three effects side by side
        _plot_effect_comparison_bar(df)

    # ── Amplification / Suppression rate heatmap ─────────────────────────────
    if "delta_score" in df.columns:
        _plot_amplification_rates(df)

    # ── Absolute-difference comparison heatmaps ───────────────────────────────
    # Shows how much the absolute bias magnitude changes when a human is added.
    # Positive = human message increases the LLM's absolute bias expression.
    if {"score_no_human", "score_neutral", "score_biased"}.issubset(df.columns):
        df2 = df.copy()
        df2["abs_no_human"] = df2["score_no_human"].abs()
        df2["abs_neutral"]  = df2["score_neutral"].abs()
        df2["abs_biased"]   = df2["score_biased"].abs()
        df2["absdiff_neutral_vs_no_human"] = df2["abs_neutral"] - df2["abs_no_human"]
        df2["absdiff_biased_vs_no_human"]  = df2["abs_biased"]  - df2["abs_no_human"]

        # |score_no_human| as bar chart — no meaningful human_bias axis
        abs_means = df2.groupby("bias")["abs_no_human"].mean().reindex(_BIAS_ORDER)
        abs_sems  = df2.groupby("bias")["abs_no_human"].sem().reindex(_BIAS_ORDER)
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        ax.bar(range(len(_BIAS_ORDER)), abs_means.values, yerr=abs_sems.values,
               color="steelblue", alpha=0.85, edgecolor="black", capsize=4, width=0.6)
        ax.set_xticks(range(len(_BIAS_ORDER)))
        ax.set_xticklabels([_SHORT.get(b, b) for b in _BIAS_ORDER])
        ax.set_ylabel("|Bias Score| (ctrl vs treat)")
        ax.set_title("|Score|: No Human Message — Intrinsic Absolute Bias Strength\n(error bars = ±1 SEM)")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        plt.tight_layout()
        _savefig("barchart_abs_no_human_jp.png")
        print(f"    abs_no_human: global mean = {df2['abs_no_human'].mean():.4f}")

        for score_col, label, fname, vmin, vmax in [
            ("abs_neutral",               "|Score|: Neutral Human Message",                    "heatmap_abs_neutral_jp.png",   0,  1),
            ("abs_biased",                "|Score|: Biased Human Message",                     "heatmap_abs_biased_jp.png",    0,  1),
            ("absdiff_neutral_vs_no_human", "|Score| Δ: Neutral Human − No Human\n(positive = neutral human increases absolute bias)", "heatmap_absdiff_neutral_jp.png", -1, 1),
            ("absdiff_biased_vs_no_human",  "|Score| Δ: Biased Human − No Human\n(positive = biased human increases absolute bias)",  "heatmap_absdiff_biased_jp.png",  -1, 1),
        ]:
            pivot = _impute_pivot(df2.pivot_table(
                values=score_col, index="bias", columns="human_bias", aggfunc="mean"
            ).reindex(index=_BIAS_ORDER, columns=_BIAS_ORDER))

            pivot_with_avg = pivot.copy()
            pivot_with_avg["Average"] = pivot.mean(axis=1)
            pivot_with_avg.loc["Average"] = pivot_with_avg.mean(axis=0)

            _heatmap(pivot_with_avg, label, fname, vmin=vmin, vmax=vmax)
            print(f"    {score_col}: global mean = {df2[score_col].mean():.4f}")


def plot_score_distributions(df: pd.DataFrame):
    cols = [
        ("score_no_human", "#9bc4e8", "score_no_human"),
        ("score_neutral",  "#70a0e0", "score_neutral"),
        ("score_biased",   "#e07070", "score_biased"),
        ("delta_score",    "#70c070", "delta_score"),
    ]
    cols = [(c, col, lbl) for c, col, lbl in cols if c in df.columns]
    fig, axes = plt.subplots(1, len(cols), figsize=(6.5, 2.5), sharey=False)
    if len(cols) == 1:
        axes = [axes]
    for ax, (col, color, label) in zip(axes, cols):
        ax.hist(df[col].dropna(), bins=30, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(df[col].mean(), color="black", lw=1.5, linestyle="--",
                   label=f"mean={df[col].mean():.3f}")
        ax.set_title(label, fontsize=7)
        ax.set_xlabel("Score")
        ax.legend(fontsize=7)
    fig.suptitle("Score Distributions (jury_passed=True rows only)", fontsize=7)
    plt.tight_layout()
    _savefig("score_distributions_jp.png")


def _plot_effect_comparison_bar(df: pd.DataFrame):
    """Grouped bar chart: presence effect, content effect, and total effect per target bias."""
    bias_means = {}
    for col, label in [
        ("effect_neutral", "Presence\n$(m_n - m_\\varnothing)$"),
        ("delta_score",    "Content\n$(m_b - m_n)$"),
        ("effect_biased",  "Total\n$(m_b - m_\\varnothing)$"),
    ]:
        if col not in df.columns:
            continue
        bias_means[label] = df.groupby("bias")[col].mean().reindex(_BIAS_ORDER)

    if not bias_means:
        return

    labels = list(bias_means.keys())
    x = np.arange(len(_BIAS_ORDER))
    width = 0.25
    colors = ["#4393c3", "#d6604d", "#74c476"]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for i, (label, vals) in enumerate(bias_means.items()):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals.values, width, label=label,
                      color=colors[i], alpha=0.85, edgecolor="white")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([_SHORT.get(b, b) for b in _BIAS_ORDER])
    ax.set_ylabel("Mean Score Difference")
    ax.set_title(
        "Decomposition of Human Interlocutor Effects on LLM Bias\n"
        "Presence effect = $m_n-m_\\varnothing$  |  "
        "Content effect = $m_b-m_n$  |  "
        "Total effect = $m_b-m_\\varnothing$",
        fontsize=7,
    )
    ax.legend(fontsize=7, loc="upper right")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    _savefig("barchart_effect_decomposition_jp.png")
    print("    effect decomposition bar chart saved")


def _plot_amplification_rates(df: pd.DataFrame):
    """
    Heatmap of per-cell amplification rate (fraction of rows where delta_score > 0)
    and a paired suppression-rate heatmap.
    """
    amp_rows, sup_rows = [], []
    for bias in _BIAS_ORDER:
        for hbias in _BIAS_ORDER:
            cell = df[(df["bias"] == bias) & (df["human_bias"] == hbias)]["delta_score"].dropna()
            n = len(cell)
            amp = (cell > 0).sum() / n if n > 0 else np.nan
            sup = (cell < 0).sum() / n if n > 0 else np.nan
            amp_rows.append({"target_bias": bias, "human_bias": hbias, "amp_rate": amp})
            sup_rows.append({"target_bias": bias, "human_bias": hbias, "sup_rate": sup})

    amp_df = pd.DataFrame(amp_rows)
    sup_df = pd.DataFrame(sup_rows)

    for data, col, title, fname in [
        (amp_df, "amp_rate",
         "Amplification Rate per Cell\n(fraction of rows where $m_b > m_n$)",
         "heatmap_amplification_rate_jp.png"),
        (sup_df, "sup_rate",
         "Suppression Rate per Cell\n(fraction of rows where $m_b < m_n$)",
         "heatmap_suppression_rate_jp.png"),
    ]:
        pivot = data.pivot_table(
            values=col, index="target_bias", columns="human_bias", aggfunc="mean"
        ).reindex(index=_BIAS_ORDER, columns=_BIAS_ORDER)
        short_idx  = [_SHORT.get(b, b) for b in _BIAS_ORDER]
        short_cols = [_SHORT.get(b, b) for b in _BIAS_ORDER]
        fig, ax = plt.subplots(figsize=(5.7, 4.5))
        sns.heatmap(
            pivot.round(2).rename(index=_SHORT, columns=_SHORT),
            cmap="YlOrRd" if "amp" in col else "Blues",
            annot=True, fmt=".2f",
            vmin=0, vmax=1,
            linewidths=0.4, linecolor="white",
            cbar_kws={"shrink": 0.7, "label": "Fraction"},
            ax=ax,
        )
        ax.set_title(title, pad=10)
        ax.set_xlabel("Human Bias", labelpad=8)
        ax.set_ylabel("Target Bias", labelpad=8)
        ax.xaxis.tick_top(); ax.xaxis.set_label_position("top")
        plt.xticks(rotation=45, ha="left")
        plt.yticks(rotation=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.tight_layout()
        _savefig(fname)
        print(f"    {col}: global mean = {data[col].mean():.3f}")


def plot_diagonal_distribution(df: pd.DataFrame):
    df2 = df.copy()
    df2["pair_type"] = np.where(
        df2["bias"] == df2["human_bias"],
        "On-diagonal\n(target = human)",
        "Off-diagonal\n(target ≠ human)",
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))
    for ax, col in zip(axes, ["score_biased", "delta_score"]):
        sns.violinplot(
            data=df2, x="pair_type", y=col,
            palette={"On-diagonal\n(target = human)": "#4393c3",
                     "Off-diagonal\n(target ≠ human)": "#d6604d"},
            inner="box", ax=ax,
        )
        ax.axhline(0, color="black", lw=0.8, linestyle="--", alpha=0.6)
        ax.set_title(f"{col} — diagonal vs off-diagonal", fontsize=7)
        ax.set_xlabel("")
        for ptype in ["On-diagonal\n(target = human)", "Off-diagonal\n(target ≠ human)"]:
            mean_val = df2[df2["pair_type"] == ptype][col].mean()
            xpos = 0 if "On" in ptype else 1
            ax.text(xpos, mean_val + 0.02, f"μ={mean_val:.3f}",
                    ha="center", fontsize=7)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.suptitle("Diagonal vs Off-diagonal (jury_passed=True)", fontsize=7)
    plt.tight_layout()
    _savefig("diagonal_distribution_jp.png")


# ══════════════════════════════════════════════════════════════════════════════
# 2. STATISTICAL SIGNIFICANCE
# ══════════════════════════════════════════════════════════════════════════════

def run_significance(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every (target_bias, human_bias) cell:
      H0: mean delta_score = 0
      Test: one-sample t-test on delta_score values in that cell.
      Also report: n, mean_delta, Cohen's d, LMM cell p-value.

    Global tests:
      - Overall t-test: is mean delta_score != 0 across all jury-confirmed pairs?
      - Diagonal vs off-diagonal t-test
      - Per-target-bias LMM with scenario random effect
    """
    print("\n[2/4] Statistical significance …")

    rows = []
    for bias in _BIAS_ORDER:
        for hbias in _BIAS_ORDER:
            cell = df[(df["bias"] == bias) & (df["human_bias"] == hbias)]["delta_score"].dropna()
            n = len(cell)
            if n < 2:
                rows.append({"target_bias": bias, "human_bias": hbias, "n": n,
                             "mean_delta": np.nan, "std_delta": np.nan,
                             "t_stat": np.nan, "p_value": np.nan,
                             "cohens_d": np.nan, "significant_05": False})
                continue
            t, p = stats.ttest_1samp(cell, 0)
            d = cell.mean() / (cell.std() + 1e-9)
            rows.append({
                "target_bias": bias, "human_bias": hbias, "n": n,
                "mean_delta": cell.mean(), "std_delta": cell.std(),
                "t_stat": t, "p_value": p, "cohens_d": d,
                "significant_05": p < 0.05,
            })

    cell_df = pd.DataFrame(rows)

    # Bonferroni correction
    valid = cell_df["p_value"].dropna()
    from statsmodels.stats.multitest import multipletests
    reject, p_adj, _, _ = multipletests(valid.values, method="bonferroni")
    cell_df.loc[cell_df["p_value"].notna(), "p_bonferroni"] = p_adj
    cell_df.loc[cell_df["p_value"].notna(), "significant_bonferroni"] = reject

    # Global test
    all_delta = df["delta_score"].dropna()
    g_t, g_p = stats.ttest_1samp(all_delta, 0)
    print(f"  Global test (all jury-confirmed pairs): mean_delta={all_delta.mean():.4f}  t={g_t:.3f}  p={g_p:.4f}")

    # Diagonal vs off-diagonal
    diag    = df[df["is_diagonal"] == 1]["delta_score"].dropna()
    offdiag = df[df["is_diagonal"] == 0]["delta_score"].dropna()
    d_t, d_p = stats.ttest_ind(diag, offdiag)
    d_cohens = (diag.mean() - offdiag.mean()) / (
        np.sqrt((diag.std()**2 + offdiag.std()**2) / 2) + 1e-9
    )
    print(f"  Diagonal  mean={diag.mean():.4f}  n={len(diag)}")
    print(f"  Off-diag  mean={offdiag.mean():.4f}  n={len(offdiag)}")
    print(f"  Diag vs off-diag: t={d_t:.3f}  p={d_p:.4f}  Cohen's d={d_cohens:.3f}")

    # Per-target-bias LMM (scenario random intercept)
    lmm_rows = []
    for bias in _BIAS_ORDER:
        sub = df[df["bias"] == bias][["delta_score", "human_bias", "scenario_id"]].dropna()
        if len(sub) < 5:
            lmm_rows.append({"target_bias": bias, "lmm_p": np.nan, "lmm_coef": np.nan})
            continue
        try:
            # Fixed: human_bias as categorical predictor; random: scenario_id
            sub = sub.copy()
            sub["hb_code"] = pd.Categorical(sub["human_bias"]).codes.astype(float)
            md = smf.mixedlm("delta_score ~ hb_code", sub, groups=sub["scenario_id"])
            mdf = md.fit(reml=True, method="lbfgs")
            lmm_rows.append({
                "target_bias": bias,
                "lmm_coef":   mdf.params.get("hb_code", np.nan),
                "lmm_p":      mdf.pvalues.get("hb_code", np.nan),
            })
        except Exception as e:
            lmm_rows.append({"target_bias": bias, "lmm_p": np.nan, "lmm_coef": np.nan})

    lmm_df = pd.DataFrame(lmm_rows)
    print("\n  Per-target-bias LMM (scenario random effect):")
    for _, r in lmm_df.iterrows():
        sig = "**" if (not np.isnan(r["lmm_p"]) and r["lmm_p"] < 0.05) else ""
        print(f"    {r['target_bias']:30s}  coef={r['lmm_coef']:.4f}  p={r['lmm_p']:.4f} {sig}")

    # Save
    os.makedirs(os.path.dirname(_SIG_CSV), exist_ok=True)
    cell_df.to_csv(_SIG_CSV, index=False)
    lmm_df.to_csv(_SIG_CSV.replace(".csv", "_lmm.csv"), index=False)
    print(f"\n  Saved → {_SIG_CSV}")

    # Significance heatmaps
    _plot_significance_heatmaps(cell_df)

    return cell_df, lmm_df, {
        "global_mean_delta": float(all_delta.mean()),
        "global_t": float(g_t), "global_p": float(g_p),
        "diag_mean": float(diag.mean()), "offdiag_mean": float(offdiag.mean()),
        "diag_vs_offdiag_t": float(d_t), "diag_vs_offdiag_p": float(d_p),
        "diag_cohens_d": float(d_cohens),
    }


def _plot_significance_heatmaps(cell_df: pd.DataFrame):
    for col, label, fname, fmt in [
        ("mean_delta",    "Mean Δ-score per cell\n(jury-confirmed only)", "sig_mean_delta_jp.png", ".3f"),
        ("cohens_d",      "Cohen's d per cell",                           "sig_cohens_d_jp.png",   ".2f"),
        ("p_value",       "p-value (one-sample t-test vs 0)",             "sig_pvalue_jp.png",     ".3f"),
        ("p_bonferroni",  "Bonferroni-corrected p-value",                 "sig_bonferroni_jp.png", ".3f"),
    ]:
        pivot = cell_df.pivot_table(values=col, index="target_bias", columns="human_bias", aggfunc="mean")
        pivot = pivot.reindex(index=_BIAS_ORDER, columns=_BIAS_ORDER)

        if "p_value" in col or "bonferroni" in col:
            cmap = "YlOrRd_r"
            center = None
            vmin, vmax = 0, 1
        else:
            cmap = sns.diverging_palette(220, 20, as_cmap=True)
            center = 0
            vmin = vmax = None

        fig, ax = plt.subplots(figsize=(5.7, 4.5))
        kw = dict(center=center) if center is not None else {}
        if vmin is not None:
            kw["vmin"] = vmin
            kw["vmax"] = vmax
        sns.heatmap(
            pivot.round(3), cmap=cmap, annot=True, fmt=fmt,
            linewidths=0.4, linecolor="white",
            cbar_kws={"shrink": 0.7}, ax=ax, **kw,
        )
        ax.set_title(label, pad=14, fontsize=7)
        ax.set_xlabel("Human Bias", labelpad=10)
        ax.set_ylabel("Target Bias", labelpad=10)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        plt.xticks(rotation=45, ha="left", fontsize=7)
        plt.yticks(rotation=0, fontsize=7)
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.tight_layout()
        _savefig(fname)


# ══════════════════════════════════════════════════════════════════════════════
# 3. CAUSAL INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def run_did(df: pd.DataFrame) -> dict:
    print("\n  [DiD] Difference-in-Differences …")
    rows = []
    for _, r in df.iterrows():
        base = {"scenario_id": r["scenario_id"], "bias": r["bias"],
                "human_bias": r["human_bias"], "is_diagonal": r["is_diagonal"]}
        rows += [
            {**base, "score": 0.0,                "group": 0, "message": 0},
            {**base, "score": 0.0,                "group": 1, "message": 0},
            {**base, "score": r["score_neutral"],  "group": 0, "message": 1},
            {**base, "score": r["score_biased"],   "group": 1, "message": 1},
        ]
    long = pd.DataFrame(rows)
    long["bias_fe"] = pd.Categorical(long["bias"],       categories=_BIAS_ORDER).codes.astype(str)
    long["hb_fe"]   = pd.Categorical(long["human_bias"], categories=_BIAS_ORDER).codes.astype(str)
    long["sid_code"] = pd.Categorical(long["scenario_id"]).codes
    long = long.dropna(subset=["score"]).reset_index(drop=True)

    model = smf.ols(
        "score ~ group * message + C(bias_fe) + C(hb_fe)", data=long,
    ).fit(cov_type="cluster", cov_kwds={"groups": long["sid_code"]})

    att  = model.params.get("group:message", np.nan)
    se   = model.bse.get("group:message",   np.nan)
    pval = model.pvalues.get("group:message", np.nan)
    ci   = model.conf_int().loc["group:message"] if "group:message" in model.conf_int().index else [np.nan, np.nan]
    print(f"    Global ATT={att:.4f}  SE={se:.4f}  p={pval:.4f}  95%CI[{ci[0]:.4f},{ci[1]:.4f}]")

    # Per-bias
    per_bias = []
    for bias in _BIAS_ORDER:
        sub = long[long["bias"] == bias]
        try:
            m = smf.ols("score ~ group * message + C(hb_fe)", data=sub).fit(
                cov_type="cluster", cov_kwds={"groups": sub["sid_code"]})
            b_att = m.params.get("group:message", np.nan)
            b_se  = m.bse.get("group:message",   np.nan)
            b_p   = m.pvalues.get("group:message", np.nan)
            b_ci  = m.conf_int().loc["group:message"] if "group:message" in m.conf_int().index else [np.nan, np.nan]
        except Exception:
            b_att = b_se = b_p = np.nan; b_ci = [np.nan, np.nan]
        per_bias.append({"bias": bias, "att": b_att, "se": b_se, "p": b_p, "ci_lo": b_ci[0], "ci_hi": b_ci[1]})
    pb = pd.DataFrame(per_bias)

    # Plot
    fig, ax = plt.subplots(figsize=(5.7, 3.5))
    y = np.arange(len(pb))
    colors = ["tomato" if (not np.isnan(p) and p < 0.05) else "steelblue" for p in pb["p"]]
    ax.barh(y, pb["att"], xerr=[pb["att"] - pb["ci_lo"], pb["ci_hi"] - pb["att"]],
            color=colors, alpha=0.8, height=0.6, capsize=4)
    ax.axvline(0, color="black", lw=1, linestyle="--")
    ax.set_yticks(y); ax.set_yticklabels(pb["bias"], fontsize=7)
    ax.set_xlabel("DiD ATT (jury-confirmed rows only)")
    ax.set_title("DiD: Causal Effect of Biased Human Message per Target Bias\n(jury_passed=True, 95% CI, cluster-robust SE)")
    ax.legend(handles=[mpatches.Patch(color="tomato", label="p<0.05"),
                        mpatches.Patch(color="steelblue", label="p≥0.05")], loc="lower right")
    plt.tight_layout()
    _savefig("did_coefficients_jp.png", subdir="causal")

    # Parallel trends
    ncols = 5; nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5, 3.5), sharey=True)
    axes = axes.flatten()
    for i, bias in enumerate(_BIAS_ORDER):
        sub = df[df["bias"] == bias]
        ax  = axes[i]
        ax.plot([0, 1], [0, sub["score_biased"].mean()],  "o-",  color="tomato",    label="biased")
        ax.plot([0, 1], [0, sub["score_neutral"].mean()], "s--", color="steelblue", label="neutral")
        ax.axhline(0, color="gray", lw=0.5, linestyle=":")
        ax.set_title(bias.replace(" ", "\n"), fontsize=7)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pre\n(no msg)", "post\n(w/ msg)"], fontsize=7)
        ax.set_ylim(-0.6, 0.6)
    fig.legend(handles=[mpatches.Patch(color="tomato", label="biased human"),
                         mpatches.Patch(color="steelblue", label="neutral human")],
               loc="lower right", fontsize=7)
    fig.suptitle("Parallel Trends (jury_passed=True only)", fontsize=7, y=1.01)
    plt.tight_layout()
    _savefig("did_parallel_trends_jp.png", subdir="causal")

    return {"global_att": att, "global_se": se, "global_p": pval, "global_ci": list(ci), "per_bias": pb}


def run_psm(df: pd.DataFrame) -> dict:
    """
    PSM using only simulator features (pre-treatment covariates).
    Jury rating features are excluded — they determine jury_passed and cannot
    be used as matching covariates for the jury_passed treatment.
    Treatment: is_diagonal (same-bias human vs different-bias human).
    Outcome: delta_score.
    """
    print("\n  [PSM] Propensity Score Matching …")
    psm_features = ["sim_sentiment", "sim_assertiveness", "sim_emotional_intensity", "sim_word_count"]

    sub = df[psm_features + ["is_diagonal", "delta_score"]].dropna().copy()
    X_raw = sub[psm_features].values
    D     = sub["is_diagonal"].values
    Y     = sub["delta_score"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(X, D)
    pscore = lr.predict_proba(X)[:, 1]
    sub["pscore"] = pscore

    idx_t = np.where(D == 1)[0]
    idx_c = np.where(D == 0)[0]
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(pscore[idx_c].reshape(-1, 1))
    _, mi = nn.kneighbors(pscore[idx_t].reshape(-1, 1))
    mc = idx_c[mi.flatten()]

    Y_t = Y[idx_t]; Y_c = Y[mc]
    att = Y_t.mean() - Y_c.mean()
    t_stat, t_p = stats.ttest_rel(Y_t, Y_c)
    att_unmatched = Y[D == 1].mean() - Y[D == 0].mean()

    print(f"    Unmatched ATT={att_unmatched:.4f}  Matched ATT={att:.4f}  t={t_stat:.3f}  p={t_p:.4f}")
    print(f"    n diagonal={len(idx_t)}  n off-diagonal={len(idx_c)}")

    # Love plot
    matched_df = pd.concat([
        sub.iloc[idx_t].assign(role="diagonal"),
        sub.iloc[mc].assign(role="matched_offdiag"),
    ]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(5.7, 3.5))
    y = np.arange(len(psm_features))
    smd_b, smd_a = [], []
    for f in psm_features:
        t_all = sub.loc[sub["is_diagonal"]==1, f]; c_all = sub.loc[sub["is_diagonal"]==0, f]
        pooled = np.sqrt((t_all.std()**2 + c_all.std()**2) / 2) + 1e-9
        smd_b.append((t_all.mean() - c_all.mean()) / pooled)
        t_m = matched_df.loc[matched_df["role"]=="diagonal", f]
        c_m = matched_df.loc[matched_df["role"]=="matched_offdiag", f]
        smd_a.append((t_m.mean() - c_m.mean()) / pooled)

    ax.scatter(smd_b, y, color="tomato",   s=70, label="Before matching", zorder=3)
    ax.scatter(smd_a, y, color="seagreen", s=70, label="After matching",  zorder=3)
    for i in range(len(psm_features)):
        ax.plot([smd_b[i], smd_a[i]], [i, i], color="gray", lw=0.8)
    ax.axvline(0,    color="black",  lw=0.8)
    ax.axvline(0.1,  color="orange", lw=0.8, linestyle="--", label="|SMD|=0.1")
    ax.axvline(-0.1, color="orange", lw=0.8, linestyle="--")
    ax.set_yticks(y); ax.set_yticklabels([f.replace("sim_","") for f in psm_features], fontsize=7)
    ax.set_xlabel("Standardised Mean Difference")
    ax.set_title("Love Plot: Diagonal vs Off-Diagonal Balance\n(jury_passed=True, sim features only)")
    ax.legend(fontsize=7)
    plt.tight_layout()
    _savefig("psm_love_plot_jp.png", subdir="causal")

    # ATT bar
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.bar(["Unmatched\nATT", "Matched\nATT"], [att_unmatched, att],
           color=["#a8c8e8", "#f4a261"], width=0.5, edgecolor="black")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Δ-score difference (diagonal − off-diagonal)")
    ax.set_title(f"PSM: Is same-bias human more effective?\nMatched ATT p={t_p:.4f}")
    plt.tight_layout()
    _savefig("psm_att_jp.png", subdir="causal")

    return {"att_unmatched": att_unmatched, "att_matched": att, "t_stat": t_stat, "t_p": t_p,
            "n_diagonal": int(len(idx_t)), "n_offdiag": int(len(idx_c))}


def _optimise_scm_weights(X_treat, X_donors):
    n = len(X_donors)
    w0 = np.ones(n) / n
    rng = X_donors.max(axis=0) - X_donors.min(axis=0); rng[rng == 0] = 1.0
    Xn_t, Xn_d = X_treat / rng, X_donors / rng
    def obj(w): return np.sum((Xn_t - w @ Xn_d) ** 2)
    sol = minimize(obj, w0, method="SLSQP",
                   bounds=[(0,1)]*n, constraints=[{"type":"eq","fun":lambda w: np.sum(w)-1}],
                   options={"ftol":1e-9,"maxiter":1000})
    return sol.x if sol.success else w0


def run_scm(df: pd.DataFrame) -> dict:
    print("\n  [SCM] Synthetic Control …")
    scm_cov = ["sim_sentiment","sim_assertiveness","sim_emotional_intensity","sim_word_count",
                "jury_target_rating","jury_purity","jury_naturalness"]
    results = []; placebo_gaps = []

    for bias in _BIAS_ORDER:
        sub = df[df["bias"] == bias].copy()
        avail = [c for c in scm_cov if c in sub.columns]
        agg = sub.groupby("human_bias").agg(
            delta_score=("delta_score","mean"), **{c:(c,"mean") for c in avail}
        ).reset_index()
        tr = agg[agg["human_bias"]==bias]; do = agg[agg["human_bias"]!=bias]
        if len(tr)==0 or len(do)<2:
            results.append({"bias":bias,"actual":np.nan,"synthetic":np.nan,"gap":np.nan,"p_placebo":np.nan})
            continue
        X_t = tr[avail].values.flatten(); X_d = do[avail].values
        Y_t = float(tr["delta_score"].values[0]); Y_d = do["delta_score"].values
        w = _optimise_scm_weights(X_t, X_d)
        synth = float(np.dot(w, Y_d)); gap = Y_t - synth
        pg = []
        for k in range(len(Y_d)):
            X_kt = X_d[k]; X_kd = np.delete(X_d,k,0); Y_kd = np.delete(Y_d,k)
            if len(Y_kd)<2: continue
            wk = _optimise_scm_weights(X_kt, X_kd); pg.append(Y_d[k] - np.dot(wk, Y_kd))
        placebo_gaps.extend(pg)
        p_val = np.mean(np.abs(pg) >= np.abs(gap)) if pg else np.nan
        results.append({"bias":bias,"actual":Y_t,"synthetic":synth,"gap":gap,"p_placebo":p_val})
        print(f"    {bias:30s} actual={Y_t:.3f} synth={synth:.3f} gap={gap:+.3f} p={p_val:.3f}")

    res_df = pd.DataFrame(results)

    # Gap bar chart
    valid = res_df.dropna(subset=["gap"])
    colors = [("tomato" if p<0.05 else "orange" if p<0.10 else "steelblue") for p in valid["p_placebo"]]
    fig, ax = plt.subplots(figsize=(5.7, 3.5))
    ax.bar(np.arange(len(valid)), valid["gap"], color=colors, alpha=0.85, edgecolor="black", width=0.6)
    ax.axhline(0, color="black", lw=1)
    if placebo_gaps:
        null95 = np.percentile(np.abs(placebo_gaps), 95)
        ax.axhline(null95,  color="gray", lw=1, linestyle="--", label=f"Placebo 95th pct={null95:.3f}")
        ax.axhline(-null95, color="gray", lw=1, linestyle="--")
    ax.set_xticks(np.arange(len(valid)))
    ax.set_xticklabels(valid["bias"].str.replace(" ","\n"), fontsize=7)
    ax.set_ylabel("Gap = Actual − Synthetic Δ-score")
    ax.set_title("Synthetic Control: Diagonal Gap (jury_passed=True)")
    ax.legend(fontsize=7)
    plt.tight_layout()
    _savefig("scm_gap_jp.png", subdir="causal")

    return {"results": res_df}


def run_causal(df: pd.DataFrame) -> dict:
    print("\n[3/4] Causal inference (jury_passed=True) …")
    os.makedirs(_CAUSAL_PLOT_DIR, exist_ok=True)
    did = run_did(df)
    psm = run_psm(df)
    scm = run_scm(df)

    # Summary CSV
    rows = []
    pb = did["per_bias"]; scm_df = scm["results"]
    for bias in _BIAS_ORDER:
        dr = pb[pb["bias"]==bias].iloc[0] if len(pb[pb["bias"]==bias])>0 else {}
        sr = scm_df[scm_df["bias"]==bias].iloc[0] if len(scm_df[scm_df["bias"]==bias])>0 else {}
        rows.append({
            "bias": bias,
            "did_att": dr.get("att",np.nan), "did_se": dr.get("se",np.nan),
            "did_p": dr.get("p",np.nan),
            "did_ci_lo": dr.get("ci_lo",np.nan), "did_ci_hi": dr.get("ci_hi",np.nan),
            "scm_actual": sr.get("actual",np.nan), "scm_synthetic": sr.get("synthetic",np.nan),
            "scm_gap": sr.get("gap",np.nan),       "scm_p_placebo": sr.get("p_placebo",np.nan),
        })
    rows.append({
        "bias": "__GLOBAL__",
        "did_att": did["global_att"], "did_se": did["global_se"], "did_p": did["global_p"],
        "did_ci_lo": did["global_ci"][0], "did_ci_hi": did["global_ci"][1],
        "scm_actual": np.nan, "scm_synthetic": np.nan, "scm_gap": np.nan, "scm_p_placebo": np.nan,
    })
    causal_df = pd.DataFrame(rows)
    for k, v in psm.items():
        causal_df[f"psm_{k}"] = np.where(causal_df["bias"]=="__GLOBAL__", v, np.nan)
    causal_df.to_csv(_CAUSAL_CSV, index=False)
    print(f"  Causal summary → {_CAUSAL_CSV}")
    return {"did": did, "psm": psm, "scm": scm}


# ══════════════════════════════════════════════════════════════════════════════
# 4. SUMMARY PRINT
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(df, sig_globals, causal):
    print("\n" + "=" * 65)
    print("RESULTS SUMMARY  (jury_passed=True rows only)")
    print("=" * 65)
    print(f"  Rows used: {len(df)}")
    print(f"  Bias pairs: {len(df.groupby(['bias','human_bias']))}")
    print(f"  Diagonal rows: {df['is_diagonal'].sum()}")
    print()
    print("── Score means ────────────────────────────────────────────")
    print(f"  score_no_human  (m∅) : {df['score_no_human'].mean():.4f}" if "score_no_human" in df.columns else "  score_no_human : n/a")
    print(f"  score_neutral   (mn) : {df['score_neutral'].mean():.4f}"  if "score_neutral"  in df.columns else "  score_neutral  : n/a")
    print(f"  score_biased    (mb) : {df['score_biased'].mean():.4f}"   if "score_biased"   in df.columns else "  score_biased   : n/a")
    print("── Effect decomposition ────────────────────────────────────")
    print(f"  effect_neutral  (mn−m∅): {df['effect_neutral'].mean():.4f}" if "effect_neutral" in df.columns else "  effect_neutral : n/a (needs score_no_human)")
    print(f"  delta_score     (mb−mn): {df['delta_score'].mean():.4f}"    if "delta_score"    in df.columns else "  delta_score    : n/a")
    print(f"  effect_biased   (mb−m∅): {df['effect_biased'].mean():.4f}"  if "effect_biased"  in df.columns else "  effect_biased  : n/a (needs score_no_human)")
    print()
    print("── Global significance ─────────────────────────────────────")
    g = sig_globals
    print(f"  H0: delta=0  →  t={g['global_t']:.3f}  p={g['global_p']:.4f}")
    print(f"  Diagonal  mean={g['diag_mean']:.4f}  Off-diag mean={g['offdiag_mean']:.4f}")
    print(f"  Diag vs off-diag: t={g['diag_vs_offdiag_t']:.3f}  p={g['diag_vs_offdiag_p']:.4f}  d={g['diag_cohens_d']:.3f}")
    print()
    print("── DiD ─────────────────────────────────────────────────────")
    did = causal["did"]
    print(f"  Global ATT={did['global_att']:.4f}  SE={did['global_se']:.4f}  p={did['global_p']:.4f}")
    print()
    print("── PSM ─────────────────────────────────────────────────────")
    psm = causal["psm"]
    print(f"  Diagonal vs off-diagonal matched ATT={psm['att_matched']:.4f}  p={psm['t_p']:.4f}")
    print()
    print("── SCM ─────────────────────────────────────────────────────")
    scm_res = causal["scm"]["results"].dropna(subset=["gap"])
    sig_scm = scm_res[scm_res["p_placebo"] < 0.05]
    print(f"  Significant diagonal gaps (p<0.05): {sig_scm['bias'].tolist()}")
    print(f"  Mean |gap| = {scm_res['gap'].abs().mean():.4f}")
    print()
    print(f"Plots  → {_PLOT_DIR}")
    print(f"Causal → {_CAUSAL_PLOT_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. CROSS-MODEL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

_MODEL_CSVS = {
    "Llama-3.1-8B":    "data/conditional_decision_results/Llama-3.1-8B_pilot.csv",
    "Llama-3.1-70B":   "data/conditional_decision_results/Llama-3.1-70B_pilot.csv",
    "Claude-3.5-Haiku":"data/conditional_decision_results/Claude-3.5-Haiku_pilot.csv",
    "GPT-4o":          "data/conditional_decision_results/GPT-4o_pilot.csv",
    "DeepSeek-V3":     "data/conditional_decision_results/DeepSeek-V3_pilot.csv",
    "Qwen-2.5-72B":    "data/conditional_decision_results/Qwen-2.5-72B-Instruct_pilot.csv",
    "Phi-4":           "data/conditional_decision_results/Phi-3.5_pilot.csv",
    "Gemma-2-9B-IT":   "data/conditional_decision_results/Gemma-2-9B-IT_pilot.csv",
}


def plot_cross_model_delta_grid():
    """
    3×3 grid of content-effect (δ = |m_b|−|m_n|) heatmaps for all 8 models.
    Each panel is a 9×9 bias coupling matrix with square cells and annotated values.
    Shared colour scale. Saved to data/results/cross_model/cross_model_delta_grid.pdf
    """
    from scipy import stats as _stats

    out_dir = os.path.join(_RUN_DIR, "data", "results", "cross_model")
    os.makedirs(out_dir, exist_ok=True)

    ncols, nrows = 4, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 10))
    axes = axes.flatten()
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    for idx, (model_name, rel_path) in enumerate(_MODEL_CSVS.items()):
        ax = axes[idx]
        full_path = os.path.join(_RUN_DIR, rel_path)
        if not os.path.exists(full_path):
            ax.set_visible(False)
            continue

        df = pd.read_csv(full_path)
        ok = df[df["status"] == "OK"].copy()
        if "individual_score" in ok.columns and "score_biased" not in ok.columns:
            ok = ok.rename(columns={"individual_score": "score_biased",
                                    "neutral_score": "score_neutral"})
        if "score_biased" not in ok.columns:
            ax.text(0.5, 0.5, "data\npending", ha="center", va="center",
                    transform=ax.transAxes, fontsize=7, color="gray")
            ax.set_title(model_name, pad=3)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_visible(False)
            continue
        ok["delta_score"] = ok["score_biased"].abs() - ok["score_neutral"].abs()

        pivot = _impute_pivot(ok.pivot_table(
            values="delta_score", index="bias", columns="human_bias", aggfunc="mean"
        ).reindex(index=_BIAS_ORDER, columns=_BIAS_ORDER))
        pivot.index   = [_SHORT.get(b, b) for b in pivot.index]
        pivot.columns = [_SHORT.get(b, b) for b in pivot.columns]

        vals = ok["delta_score"].dropna()
        mean_d = vals.mean()
        _, p = _stats.ttest_1samp(vals, 0)
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))

        sns.heatmap(
            pivot, cmap=cmap, center=0,
            annot=True, fmt=".2f", annot_kws={"size": 6},
            square=True,
            vmin=-0.3, vmax=0.3,
            linewidths=0.2, linecolor="white",
            cbar=False,
            ax=ax,
        )
        ax.set_title(f"{model_name}\n$\\bar{{\\delta}}$={mean_d:+.3f} {sig}", pad=3)
        ax.set_xlabel("Human bias" if idx < ncols else "", labelpad=2)
        ax.set_ylabel("Target bias" if idx % ncols == 0 else "", labelpad=2)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="left")
        plt.setp(ax.get_yticklabels(), rotation=0)
        if idx % ncols != 0:
            ax.set_yticklabels([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    for idx in range(len(_MODEL_CSVS), len(axes)):
        axes[idx].set_visible(False)

    from matplotlib.colors import TwoSlopeNorm
    import matplotlib.cm as _cm
    norm = TwoSlopeNorm(vmin=-0.3, vcenter=0, vmax=0.3)
    sm = _cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.subplots_adjust(right=0.87, hspace=0.55, wspace=0.4)
    cbar_ax = fig.add_axes([0.89, 0.10, 0.015, 0.78])
    fig.colorbar(sm, cax=cbar_ax, label="δ")

    out_path = os.path.join(out_dir, "cross_model_delta_grid.pdf")
    plt.savefig(out_path, format="pdf", dpi=600, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out_path}")


def plot_cross_model_presence_effect_grid():
    """
    3×3 grid of presence-effect (Δ_n = |m_n|−|m_∅|) heatmaps for all 8 models.
    Same layout and colour scale as plot_cross_model_delta_grid. Square cells with values.
    Models without a zero-shot condition (Claude-3.5-Haiku) are shown as blank panels.
    Saved to data/results/cross_model/cross_model_presence_effect_grid.pdf
    """
    from scipy import stats as _stats

    out_dir = os.path.join(_RUN_DIR, "data", "results", "cross_model")
    os.makedirs(out_dir, exist_ok=True)

    ncols, nrows = 4, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 10))
    axes = axes.flatten()
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    for idx, (model_name, rel_path) in enumerate(_MODEL_CSVS.items()):
        ax = axes[idx]
        full_path = os.path.join(_RUN_DIR, rel_path)
        if not os.path.exists(full_path):
            ax.set_visible(False)
            continue

        df = pd.read_csv(full_path)
        ok = df[df["status"] == "OK"].copy()
        if "individual_score" in ok.columns and "score_biased" not in ok.columns:
            ok = ok.rename(columns={"individual_score": "score_biased",
                                    "neutral_score": "score_neutral"})
        if "score_biased" not in ok.columns:
            ax.text(0.5, 0.5, "data\npending", ha="center", va="center",
                    transform=ax.transAxes, fontsize=7, color="gray")
            ax.set_title(model_name, pad=3)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_visible(False)
            continue

        if "score_no_human" not in ok.columns or ok["score_no_human"].isna().all():
            ax.text(0.5, 0.5, "no zero-shot\ncondition",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=7, color="gray")
            ax.set_title(model_name, pad=3)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            continue

        ok["presence_effect"] = ok["score_neutral"].abs() - ok["score_no_human"].abs()

        pivot = _impute_pivot(ok.pivot_table(
            values="presence_effect", index="bias", columns="human_bias", aggfunc="mean"
        ).reindex(index=_BIAS_ORDER, columns=_BIAS_ORDER))
        pivot.index   = [_SHORT.get(b, b) for b in pivot.index]
        pivot.columns = [_SHORT.get(b, b) for b in pivot.columns]

        vals = ok["presence_effect"].dropna()
        mean_d = vals.mean()
        _, p = _stats.ttest_1samp(vals, 0)
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))

        sns.heatmap(
            pivot, cmap=cmap, center=0,
            annot=True, fmt=".2f", annot_kws={"size": 6},
            square=True,
            vmin=-0.3, vmax=0.3,
            linewidths=0.2, linecolor="white",
            cbar=False,
            ax=ax,
        )
        ax.set_title(f"{model_name}\n$\\bar{{\\Delta}}_n$={mean_d:+.3f} {sig}", pad=3)
        ax.set_xlabel("Human bias" if idx < ncols else "", labelpad=2)
        ax.set_ylabel("Target bias" if idx % ncols == 0 else "", labelpad=2)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="left")
        plt.setp(ax.get_yticklabels(), rotation=0)
        if idx % ncols != 0:
            ax.set_yticklabels([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    for idx in range(len(_MODEL_CSVS), len(axes)):
        axes[idx].set_visible(False)

    from matplotlib.colors import TwoSlopeNorm
    import matplotlib.cm as _cm
    norm = TwoSlopeNorm(vmin=-0.3, vcenter=0, vmax=0.3)
    sm = _cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.subplots_adjust(right=0.87, hspace=0.55, wspace=0.4)
    cbar_ax = fig.add_axes([0.89, 0.10, 0.015, 0.78])
    fig.colorbar(sm, cax=cbar_ax, label="$\\Delta_n$")

    out_path = os.path.join(out_dir, "cross_model_presence_effect_grid.pdf")
    plt.savefig(out_path, format="pdf", dpi=600, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out_path}")


def plot_cross_model_total_effect_grid():
    """
    3×3 grid of total-effect (Δ_b = |m_b|−|m_∅|) heatmaps for all 8 models.
    Same layout and colour scale as plot_cross_model_delta_grid. Square cells with values.
    Models without a zero-shot condition (Claude-3.5-Haiku) are shown as blank panels.
    Saved to data/results/cross_model/cross_model_total_effect_grid.pdf
    """
    from scipy import stats as _stats

    out_dir = os.path.join(_RUN_DIR, "data", "results", "cross_model")
    os.makedirs(out_dir, exist_ok=True)

    ncols, nrows = 4, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 10))
    axes = axes.flatten()
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    for idx, (model_name, rel_path) in enumerate(_MODEL_CSVS.items()):
        ax = axes[idx]
        full_path = os.path.join(_RUN_DIR, rel_path)
        if not os.path.exists(full_path):
            ax.set_visible(False)
            continue

        df = pd.read_csv(full_path)
        ok = df[df["status"] == "OK"].copy()
        if "individual_score" in ok.columns and "score_biased" not in ok.columns:
            ok = ok.rename(columns={"individual_score": "score_biased",
                                    "neutral_score": "score_neutral"})

        if "score_biased" not in ok.columns:
            ax.text(0.5, 0.5, "data\npending", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="gray")
            ax.set_title(model_name, pad=3, fontsize=13)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_visible(False)
            continue

        if "score_no_human" not in ok.columns or ok["score_no_human"].isna().all():
            ax.text(0.5, 0.5, "no zero-shot\ncondition",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=12, color="gray")
            ax.set_title(model_name, pad=3, fontsize=13)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            continue

        ok["total_effect"] = ok["score_biased"].abs() - ok["score_no_human"].abs()

        pivot = _impute_pivot(ok.pivot_table(
            values="total_effect", index="bias", columns="human_bias", aggfunc="mean"
        ).reindex(index=_BIAS_ORDER, columns=_BIAS_ORDER))
        pivot.index   = [_SHORT.get(b, b) for b in pivot.index]
        pivot.columns = [_SHORT.get(b, b) for b in pivot.columns]

        vals = ok["total_effect"].dropna()
        mean_d = vals.mean()
        _, p = _stats.ttest_1samp(vals, 0)
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))

        sns.heatmap(
            pivot, cmap=cmap, center=0,
            annot=True, fmt=".2f", annot_kws={"size": 9},
            square=True,
            vmin=-0.3, vmax=0.3,
            linewidths=0.2, linecolor="white",
            cbar=False,
            ax=ax,
        )
        ax.set_title(f"{model_name}\n$\\bar{{\\Delta}}_b$={mean_d:+.3f} {sig}", pad=3, fontsize=13)
        ax.set_xlabel("Human bias" if idx < ncols else "", labelpad=2, fontsize=13)
        ax.set_ylabel("Target bias" if idx % ncols == 0 else "", labelpad=2, fontsize=13)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="left", fontsize=12)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=12)
        if idx % ncols != 0:
            ax.set_yticklabels([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    for idx in range(len(_MODEL_CSVS), len(axes)):
        axes[idx].set_visible(False)

    from matplotlib.colors import TwoSlopeNorm
    import matplotlib.cm as _cm
    norm = TwoSlopeNorm(vmin=-0.3, vcenter=0, vmax=0.3)
    sm = _cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.subplots_adjust(right=0.87, hspace=0.55, wspace=0.4)
    cbar_ax = fig.add_axes([0.89, 0.10, 0.015, 0.78])
    cb = fig.colorbar(sm, cax=cbar_ax, label="$\\Delta_b$")
    cb.ax.tick_params(labelsize=12)
    cb.set_label("$\\Delta_b$", fontsize=13)

    out_path = os.path.join(out_dir, "cross_model_total_effect_grid.pdf")
    plt.savefig(out_path, format="pdf", dpi=600, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out_path}")


def plot_cross_model_summary():
    """
    Grouped bar chart of presence, content, and total effects for all 8 models.
    Saved to data/results/cross_model/cross_model_effects.pdf
    """
    from scipy import stats as _stats

    out_dir = os.path.join(_RUN_DIR, "data", "results", "cross_model")
    os.makedirs(out_dir, exist_ok=True)

    records = []
    for model_name, path in _MODEL_CSVS.items():
        full_path = os.path.join(_RUN_DIR, path)
        if not os.path.exists(full_path):
            print(f"  skipping {model_name} — CSV not found")
            continue
        df = pd.read_csv(full_path)
        ok = df[df["status"] == "OK"].copy()
        if "individual_score" in ok.columns and "score_biased" not in ok.columns:
            ok = ok.rename(columns={"individual_score": "score_biased",
                                    "neutral_score": "score_neutral"})
        if "score_biased" in ok.columns and "score_neutral" in ok.columns:
            ok["delta_score"] = ok["score_biased"].abs() - ok["score_neutral"].abs()
        if "score_no_human" in ok.columns:
            ok["effect_neutral"] = ok["score_neutral"].abs() - ok["score_no_human"].abs()
            ok["effect_biased"]  = ok["score_biased"].abs()  - ok["score_no_human"].abs()

        for col, label in [("effect_neutral", "Presence"), ("delta_score", "Content"),
                           ("effect_biased", "Total")]:
            if col not in ok.columns:
                continue
            vals = ok[col].dropna()
            mean = vals.mean()
            se   = vals.sem()
            t, p = _stats.ttest_1samp(vals, 0)
            records.append({"model": model_name, "effect": label,
                            "mean": mean, "se": se, "t": t, "p": p})

    if not records:
        print("  cross-model summary: no data found")
        return

    data = pd.DataFrame(records)

    models  = list(_MODEL_CSVS.keys())
    models  = [m for m in models if m in data["model"].unique()]
    effects = ["Presence", "Content", "Total"]
    colors  = {"Presence": "#4393c3", "Content": "#d6604d", "Total": "#74c476"}

    x      = np.arange(len(models))
    width  = 0.25
    fig, ax = plt.subplots(figsize=(6.5, 3.0))

    for i, eff in enumerate(effects):
        sub = data[data["effect"] == eff].set_index("model").reindex(models)
        offset = (i - 1) * width
        bars = ax.bar(x + offset, sub["mean"].values,
                      yerr=sub["se"].values,
                      width=width, label=eff,
                      color=colors[eff], alpha=0.85,
                      edgecolor="white", capsize=2)
        # star for significance
        for j, (mean, p) in enumerate(zip(sub["mean"].values, sub["p"].values)):
            if not np.isnan(p) and p < 0.05:
                ypos = mean + sub["se"].values[j] + 0.008
                ax.text(x[j] + offset, ypos, "*", ha="center", va="bottom", fontsize=6)

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("Mean effect (|m| difference)")
    ax.set_title("Presence, Content, and Total Effects Across 8 LLMs\n(* p<0.05; error bars = ±1 SE)")
    ax.legend(loc="upper right")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "cross_model_effects.pdf")
    plt.savefig(out_path, format="pdf", dpi=600, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out_path}")


def plot_cross_model_causal_forest():
    """
    Forest-plot style causal summary: one dot per model per bias.
    Left panel: DiD ATT with 95% CI error bars.
    Right panel: SCM gap with placebo significance markers.
    Saved to data/results/cross_model/cross_model_causal_forest.pdf
    """
    MODEL_DIRS = {
        "Llama-8B":    "Llama-3.1-8B",
        "Llama-70B":   "Llama-3.1-70B",
        "Claude-Haiku":"Claude-3.5-Haiku",
        "GPT-4o":      "GPT-4o",
        "DeepSeek-V3": "DeepSeek-V3",
        "Qwen-72B":    "Qwen-2.5-72B-Instruct",
        "Phi-4":       "Phi-3.5",
        "Gemma-9B":    "Gemma-2-9B-IT",
    }
    MODEL_COLORS = {
        "Llama-8B":    "#4393c3",
        "Llama-70B":   "#2166ac",
        "Claude-Haiku":"#d6604d",
        "GPT-4o":      "#f4a582",
        "DeepSeek-V3": "#1a9850",
        "Qwen-72B":    "#66bd63",
        "Phi-4":       "#9970ab",
        "Gemma-9B":    "#c2a5cf",
    }

    out_dir = os.path.join(_RUN_DIR, "data", "results", "cross_model")
    os.makedirs(out_dir, exist_ok=True)

    biases   = _BIAS_ORDER
    short_b  = [_SHORT.get(b, b) for b in biases]
    models   = list(MODEL_DIRS.keys())
    n_bias   = len(biases)
    n_model  = len(models)

    # Collect records per (bias, model)
    records = []
    for disp_name, result_dir in MODEL_DIRS.items():
        csv_path = os.path.join(_RUN_DIR, "data", "results", result_dir, "causal_summary.csv")
        if not os.path.exists(csv_path):
            continue
        raw = pd.read_csv(csv_path)
        raw = raw[raw["bias"] != "bias"].copy()
        raw = raw[raw["bias"].isin(biases)].copy()
        raw.set_index("bias", inplace=True)
        for col in ["did_att", "did_se", "did_ci_lo", "did_ci_hi",
                    "scm_gap", "scm_p_placebo"]:
            if col in raw.columns:
                raw[col] = pd.to_numeric(raw[col], errors="coerce")
        for b in biases:
            if b not in raw.index:
                continue
            row = raw.loc[b]
            records.append({
                "model":       disp_name,
                "bias":        b,
                "did_att":     row.get("did_att",   np.nan),
                "did_ci_lo":   row.get("did_ci_lo", np.nan),
                "did_ci_hi":   row.get("did_ci_hi", np.nan),
                "scm_gap":     row.get("scm_gap",   np.nan),
                "scm_p":       row.get("scm_p_placebo", np.nan),
            })

    df = pd.DataFrame(records)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.5),
                                   sharey=True, gridspec_kw={"wspace": 0.08})

    jitter_step = 0.08   # vertical jitter per model within a bias row
    y_mid       = np.arange(n_bias)

    for panel_idx, (ax, val_col, lo_col, hi_col, sig_col, xlabel, title) in enumerate([
        (ax1, "did_att", "did_ci_lo", "did_ci_hi", None,    "DiD ATT",  "Difference-in-Differences"),
        (ax2, "scm_gap", None,        None,         "scm_p", "SCM gap",  "Synthetic Control"),
    ]):
        ax.axvline(0, color="black", lw=0.8, zorder=1)

        # Highlight Planning Fallacy row
        pf_idx = biases.index("Planning Fallacy")
        ax.axhspan(pf_idx - 0.45, pf_idx + 0.45,
                   color="#fff3cd", alpha=0.5, zorder=0)

        for m_idx, model in enumerate(models):
            sub = df[(df["model"] == model)]
            color = MODEL_COLORS[model]
            y_off = (m_idx - (n_model - 1) / 2) * jitter_step

            for b_idx, bias in enumerate(biases):
                row = sub[sub["bias"] == bias]
                if row.empty:
                    continue
                v   = row[val_col].values[0]
                y   = y_mid[b_idx] + y_off

                if pd.isna(v):
                    continue

                # Error bars for DiD
                if lo_col and hi_col:
                    lo = row[lo_col].values[0]
                    hi = row[hi_col].values[0]
                    if not (pd.isna(lo) or pd.isna(hi)):
                        ax.plot([lo, hi], [y, y], color=color, lw=0.8,
                                alpha=0.6, zorder=2)

                # Significance marker for SCM
                marker = "D" if sig_col else "o"
                size   = 22
                if sig_col:
                    p = row[sig_col].values[0]
                    marker = "D" if (not pd.isna(p) and p < 0.05) else "o"

                ax.scatter(v, y, color=color, s=size, marker=marker,
                           zorder=3, linewidths=0.4,
                           edgecolors="white" if marker == "D" else color)

        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_title(title, fontsize=13, pad=5)
        ax.set_xlim(-0.35, 0.35)
        ax.tick_params(axis="both", labelsize=12)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.grid(axis="x", lw=0.4, color="gray", alpha=0.4, zorder=0)

    ax1.set_yticks(y_mid)
    ax1.set_yticklabels(short_b, fontsize=12)
    ax1.set_ylabel("Target bias", fontsize=13)
    ax1.invert_yaxis()
    ax2.invert_yaxis()

    # Legend
    handles = [plt.scatter([], [], color=MODEL_COLORS[m], s=22, label=m)
               for m in models]
    sig_handle = plt.scatter([], [], color="gray", s=22, marker="D",
                             label="SCM p<0.05")
    fig.legend(handles=handles + [sig_handle],
               loc="lower center", ncol=5, fontsize=11,
               frameon=False, bbox_to_anchor=(0.5, -0.10))

    fig.suptitle("Causal identification: DiD ATT and SCM gap per bias × model\n"
                 "(error bars = 95% CI; ◆ = SCM p<0.05)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "cross_model_causal_forest.pdf")
    plt.savefig(out_path, format="pdf", dpi=600, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out_path}")


def plot_cross_model_causal_heatmaps():
    """
    2-panel heatmap: DiD ATT (left) and SCM gap (right) for all 8 models × 9 biases.
    Cells are square, annotated with value + significance stars.
    Saved to data/results/cross_model/cross_model_causal_heatmaps.pdf
    """
    # Map display model names → causal_summary.csv result directories
    MODEL_DIRS = {
        "Llama-8B":    "Llama-3.1-8B",
        "Llama-70B":   "Llama-3.1-70B",
        "Claude-Haiku":"Claude-3.5-Haiku",
        "GPT-4o":      "GPT-4o",
        "DeepSeek-V3": "DeepSeek-V3",
        "Qwen-72B":    "Qwen-2.5-72B-Instruct",
        "Phi-4":       "Phi-3.5",
        "Gemma-9B":    "Gemma-2-9B-IT",
    }

    out_dir = os.path.join(_RUN_DIR, "data", "results", "cross_model")
    os.makedirs(out_dir, exist_ok=True)

    models      = list(MODEL_DIRS.keys())
    biases      = _BIAS_ORDER
    short_b     = [_SHORT.get(b, b) for b in biases]

    did_att = pd.DataFrame(index=biases, columns=models, dtype=float)
    did_p   = pd.DataFrame(index=biases, columns=models, dtype=float)
    scm_gap = pd.DataFrame(index=biases, columns=models, dtype=float)
    scm_p   = pd.DataFrame(index=biases, columns=models, dtype=float)

    for disp_name, result_dir in MODEL_DIRS.items():
        csv_path = os.path.join(_RUN_DIR, "data", "results", result_dir, "causal_summary.csv")
        if not os.path.exists(csv_path):
            continue
        raw = pd.read_csv(csv_path)
        raw = raw[raw["bias"] != "bias"].copy()   # drop duplicate header rows
        raw = raw[raw["bias"].isin(biases)].copy()
        raw.set_index("bias", inplace=True)
        for col in ["did_att", "did_p", "scm_gap", "scm_p_placebo"]:
            if col in raw.columns:
                raw[col] = pd.to_numeric(raw[col], errors="coerce")
        if "did_att" in raw.columns:
            did_att[disp_name] = raw["did_att"].reindex(biases)
        if "did_p" in raw.columns:
            did_p[disp_name]   = raw["did_p"].reindex(biases)
        if "scm_gap" in raw.columns:
            scm_gap[disp_name] = raw["scm_gap"].reindex(biases)
        if "scm_p_placebo" in raw.columns:
            scm_p[disp_name]   = raw["scm_p_placebo"].reindex(biases)

    def _stars(p):
        if pd.isna(p): return ""
        return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))

    def _annot(val_df, p_df):
        ann = val_df.copy().astype(object)
        for b in biases:
            for m in models:
                v = val_df.loc[b, m]
                p = p_df.loc[b, m]
                ann.loc[b, m] = "—" if pd.isna(v) else f"{v:+.2f}{_stars(p)}"
        ann.index = short_b
        return ann

    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── Panel 1: DiD ATT ────────────────────────────────────────────────────
    did_plot = did_att.copy()
    did_plot.index = short_b
    sns.heatmap(
        did_plot.astype(float), ax=ax1,
        cmap=cmap, center=0, vmin=-0.25, vmax=0.25,
        annot=_annot(did_att, did_p), fmt="",
        annot_kws={"size": 7},
        square=True, linewidths=0.3, linecolor="white",
        cbar_kws={"shrink": 0.6, "label": "ATT"},
    )
    ax1.set_title("Difference-in-Differences ATT\n($^*$p<0.05, $^{**}$p<0.01, $^{***}$p<0.001)", pad=6)
    ax1.set_xlabel("Model"); ax1.set_ylabel("Target bias")
    ax1.xaxis.tick_top(); ax1.xaxis.set_label_position("top")
    plt.setp(ax1.get_xticklabels(), rotation=30, ha="left", fontsize=8)
    plt.setp(ax1.get_yticklabels(), rotation=0, fontsize=8)
    for sp in ax1.spines.values(): sp.set_visible(False)

    # ── Panel 2: SCM gap ────────────────────────────────────────────────────
    scm_plot = scm_gap.copy()
    scm_plot.index = short_b
    sns.heatmap(
        scm_plot.astype(float), ax=ax2,
        cmap=cmap, center=0, vmin=-0.25, vmax=0.25,
        annot=_annot(scm_gap, scm_p), fmt="",
        annot_kws={"size": 7},
        square=True, linewidths=0.3, linecolor="white",
        cbar_kws={"shrink": 0.6, "label": "Gap"},
    )
    ax2.set_title("Synthetic Control Gap\n($^*$p<0.05, $^{**}$p<0.01, $^{***}$p<0.001)", pad=6)
    ax2.set_xlabel("Model"); ax2.set_ylabel("")
    ax2.set_yticklabels([])
    ax2.xaxis.tick_top(); ax2.xaxis.set_label_position("top")
    plt.setp(ax2.get_xticklabels(), rotation=30, ha="left", fontsize=8)
    for sp in ax2.spines.values(): sp.set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "cross_model_causal_heatmaps.pdf")
    plt.savefig(out_path, format="pdf", dpi=600, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(pilot_csv: str = _PILOT_CSV):
    global _PLOT_DIR, _CAUSAL_PLOT_DIR, _SIG_CSV, _CAUSAL_CSV

    # Derive model name from filename, e.g. "Claude-3.5-Haiku_pilot.csv" → "Claude-3.5-Haiku"
    basename = os.path.basename(pilot_csv)
    model_name = basename.replace("_pilot.csv", "").replace(".csv", "")

    # All outputs go under data/results/{model_name}/
    _model_dir       = os.path.join(_RUN_DIR, "data", "results", model_name)
    _PLOT_DIR        = os.path.join(_model_dir, "plots")
    _CAUSAL_PLOT_DIR = os.path.join(_PLOT_DIR, "causal")
    _SIG_CSV         = os.path.join(_model_dir, "significance_results.csv")
    _CAUSAL_CSV      = os.path.join(_model_dir, "causal_summary.csv")

    os.makedirs(_PLOT_DIR, exist_ok=True)
    os.makedirs(_CAUSAL_PLOT_DIR, exist_ok=True)
    print(f"Loading {pilot_csv} …")
    print(f"Saving results → {_model_dir}")
    df = load_data(pilot_csv)

    plot_heatmaps(df)
    plot_score_distributions(df)
    plot_diagonal_distribution(df)

    cell_df, lmm_df, sig_globals = run_significance(df)
    causal = run_causal(df)

    print_summary(df, sig_globals, causal)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=_PILOT_CSV)
    args = parser.parse_args()
    main(args.input)
