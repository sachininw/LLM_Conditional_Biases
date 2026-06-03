"""
Causal inference analysis for the conditional cognitive bias experiment.

Three methods applied to GPT-4o_pilot.csv:

1. Difference-in-Differences (DiD)
   - Formalises delta_score as a DiD estimator
   - OLS with scenario fixed effects and clustered standard errors
   - Coefficient plot + parallel-trends visualisation per target bias

2. Propensity Score Matching (PSM)
   - Asks: do rows where the jury confirmed the bias (jury_passed=True) produce
     higher individual_score than comparable jury-failed rows?
   - Logistic propensity model on simulator & jury features
   - 1:1 nearest-neighbour matching; Love plot of covariate balance; ATT estimate

3. Synthetic Control Method (SCM)
   - For each target bias B, the diagonal cell (human_bias == B) is the treated unit
   - Donor pool: off-diagonal cells with the same target bias
   - Weights minimise distance on covariates → synthetic counterfactual delta_score
   - Placebo permutation test for inference; gap + heatmap visualisations

Outputs:
  run/data/plots/causal/  — all figures (PNG, 300 dpi)
  run/data/conditional_decision_results/causal_summary.csv
"""

import os
import sys
import ast
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
import statsmodels.api as sm

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
_RUN_DIR = os.path.dirname(os.path.abspath(__file__))
_PILOT_CSV = os.path.join(_RUN_DIR, "data", "conditional_decision_results", "GPT-4o_pilot.csv")
_PLOT_DIR  = os.path.join(_RUN_DIR, "data", "plots", "causal")
_SUMMARY_CSV = os.path.join(_RUN_DIR, "data", "conditional_decision_results", "causal_summary.csv")

_PALETTE = sns.color_palette("Set2", 9)
_BIAS_ORDER = [
    "Anchoring", "Availability Heuristic", "Bandwagon Effect",
    "Confirmation Bias", "Framing Effect", "In Group Bias",
    "Loss Aversion", "Planning Fallacy", "Status Quo Bias",
]
_SHORT = {b: b.replace(" ", "\n") for b in _BIAS_ORDER}

def _savefig(name: str):
    pdf_name = os.path.splitext(name)[0] + ".pdf"
    path = os.path.join(_PLOT_DIR, pdf_name)
    plt.savefig(path, format="pdf", dpi=600, bbox_inches="tight")
    plt.close()
    print(f"  saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data(path: str = _PILOT_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["status"] == "OK"].copy()
    df["scenario_id"] = df["id"].astype(str) + "_" + df["bias"]
    df["is_diagonal"]  = (df["bias"] == df["human_bias"]).astype(int)
    # ordinal encoding for bias names (used as fixed effects)
    for col in ["bias", "human_bias"]:
        df[col + "_code"] = pd.Categorical(df[col], categories=_BIAS_ORDER).codes
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 1. DIFFERENCE-IN-DIFFERENCES
# ══════════════════════════════════════════════════════════════════════════════

def run_did(df: pd.DataFrame) -> dict:
    """
    DiD design:
      unit    = each (scenario_id, human_bias) observation
      outcome = score
      pre     = control condition (no human message, score = 0 by construction,
                 but we use actual control_decision numeric as the raw anchor)
      post    = with human message (either biased → individual_score,
                 or neutral → neutral_score)
      group   = biased (1) vs neutral (0)

    We reshape to long format:
      score  ~ D_biased * D_message + scenario_FE + human_bias_FE

    The interaction coefficient β₃ is the DiD ATT.
    """
    print("\n[1/3] Difference-in-Differences …")

    rows = []
    for _, r in df.iterrows():
        base = {
            "scenario_id": r["scenario_id"],
            "bias": r["bias"],
            "human_bias": r["human_bias"],
            "is_diagonal": r["is_diagonal"],
            "jury_passed": int(r.get("jury_passed", False)),
        }
        # control arm (no message, pre-treatment baseline for both groups)
        rows.append({**base, "score": 0.0, "group": 0, "message": 0})  # neutral, no msg
        rows.append({**base, "score": 0.0, "group": 1, "message": 0})  # biased,  no msg
        # post-message observations
        rows.append({**base, "score": r["neutral_score"],    "group": 0, "message": 1})
        rows.append({**base, "score": r["individual_score"], "group": 1, "message": 1})

    long = pd.DataFrame(rows)
    long["bias_fe"]      = pd.Categorical(long["bias"],      categories=_BIAS_ORDER).codes.astype(str)
    long["hb_fe"]        = pd.Categorical(long["human_bias"], categories=_BIAS_ORDER).codes.astype(str)
    long["scenario_fe"]  = long["scenario_id"]

    # Global DiD regression with scenario + human_bias fixed effects
    model = smf.ols(
        "score ~ group * message + C(bias_fe) + C(hb_fe)",
        data=long,
    ).fit(cov_type="cluster", cov_kwds={"groups": long["scenario_id"]})

    att = model.params.get("group:message", np.nan)
    se  = model.bse.get("group:message",  np.nan)
    pval = model.pvalues.get("group:message", np.nan)
    ci   = model.conf_int().loc["group:message"] if "group:message" in model.conf_int().index else [np.nan, np.nan]

    print(f"  Global DiD ATT = {att:.4f}  SE={se:.4f}  p={pval:.4f}  95% CI [{ci[0]:.4f}, {ci[1]:.4f}]")

    # Per-target-bias DiD
    per_bias = []
    for bias in _BIAS_ORDER:
        sub = long[long["bias"] == bias]
        try:
            m = smf.ols(
                "score ~ group * message + C(hb_fe)",
                data=sub,
            ).fit(cov_type="cluster", cov_kwds={"groups": sub["scenario_id"]})
            b_att  = m.params.get("group:message", np.nan)
            b_se   = m.bse.get("group:message",   np.nan)
            b_p    = m.pvalues.get("group:message", np.nan)
            b_ci   = m.conf_int().loc["group:message"] if "group:message" in m.conf_int().index else [np.nan, np.nan]
        except Exception:
            b_att = b_se = b_p = np.nan
            b_ci  = [np.nan, np.nan]
        per_bias.append({"bias": bias, "att": b_att, "se": b_se, "p": b_p,
                         "ci_lo": b_ci[0], "ci_hi": b_ci[1]})
    pb = pd.DataFrame(per_bias)

    _plot_did_coefs(pb)
    _plot_did_parallel(df)

    return {"global_att": att, "global_se": se, "global_p": pval,
            "global_ci": list(ci), "per_bias": pb}


def _plot_did_coefs(pb: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5.7, 3.5))
    y = np.arange(len(pb))
    colors = [("tomato" if p < 0.05 else "steelblue") for p in pb["p"]]
    ax.barh(y, pb["att"], xerr=[pb["att"] - pb["ci_lo"], pb["ci_hi"] - pb["att"]],
            color=colors, alpha=0.8, height=0.6, capsize=4)
    ax.axvline(0, color="black", lw=1, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(pb["bias"], fontsize=7)
    ax.set_xlabel("DiD ATT  (individual_score − neutral_score  |  biased vs neutral group)")
    ax.set_title("DiD: Causal Effect of Biased Human Message\nper Target Bias (95% CI, cluster-robust SE)")
    red_p  = mpatches.Patch(color="tomato",    label="p < 0.05")
    blue_p = mpatches.Patch(color="steelblue", label="p ≥ 0.05")
    ax.legend(handles=[red_p, blue_p], loc="lower right")
    fig.tight_layout()
    _savefig("did_coefficients.png")


def _plot_did_parallel(df: pd.DataFrame):
    """
    Parallel-trends visualisation: mean score in each condition per target bias.
    X-axis: 0 = no message (control), 1 = with message
    Two lines per panel: biased group (individual) and neutral group.
    """
    biases = _BIAS_ORDER
    ncols  = 5
    nrows  = int(np.ceil(len(biases) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5, 3.5), sharey=True)
    axes = axes.flatten()

    for i, bias in enumerate(biases):
        sub = df[df["bias"] == bias]
        ax  = axes[i]
        # biased group: (0 → control=0), (1 → individual_score)
        ax.plot([0, 1], [0, sub["individual_score"].mean()], "o-", color="tomato",   label="biased")
        # neutral group: (0 → control=0), (1 → neutral_score)
        ax.plot([0, 1], [0, sub["neutral_score"].mean()],    "s--", color="steelblue", label="neutral")
        ax.axhline(0, color="gray", lw=0.5, linestyle=":")
        ax.set_title(bias.replace(" ", "\n"), fontsize=7)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["pre\n(no msg)", "post\n(w/ msg)"], fontsize=7)
        ax.set_ylim(-0.6, 0.6)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    handles = [
        mpatches.Patch(color="tomato",    label="biased human"),
        mpatches.Patch(color="steelblue", label="neutral human"),
    ]
    fig.legend(handles=handles, loc="lower right", fontsize=7)
    fig.suptitle("Parallel Trends: Mean Score Pre/Post Human Message by Target Bias", fontsize=7, y=1.01)
    fig.tight_layout()
    _savefig("did_parallel_trends.png")


# ══════════════════════════════════════════════════════════════════════════════
# 2. PROPENSITY SCORE MATCHING
# ══════════════════════════════════════════════════════════════════════════════

_PSM_FEATURES = [
    "sim_sentiment", "sim_assertiveness", "sim_emotional_intensity", "sim_word_count",
    "jury_target_rating", "jury_purity", "jury_naturalness",
]

def run_psm(df: pd.DataFrame) -> dict:
    """
    PSM question: among all observations, does having a jury-confirmed biased
    human message (jury_passed=True) causally increase individual_score,
    relative to comparable jury-failed observations?

    Treatment D = jury_passed (1/0)
    Outcome   Y = individual_score
    """
    print("\n[2/3] Propensity Score Matching …")

    sub = df[_PSM_FEATURES + ["jury_passed", "individual_score", "bias", "human_bias"]].dropna()
    sub = sub.copy()
    sub["D"] = sub["jury_passed"].astype(int)
    X_raw = sub[_PSM_FEATURES].values
    Y     = sub["individual_score"].values
    D     = sub["D"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # Propensity model
    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(X, D)
    pscore = lr.predict_proba(X)[:, 1]
    sub["pscore"] = pscore

    # 1:1 Nearest-neighbour matching (treated → control)
    idx_treated = np.where(D == 1)[0]
    idx_control = np.where(D == 0)[0]

    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(pscore[idx_control].reshape(-1, 1))
    dists, matched_idx = nn.kneighbors(pscore[idx_treated].reshape(-1, 1))
    matched_control = idx_control[matched_idx.flatten()]

    matched_df = pd.concat([
        sub.iloc[idx_treated].assign(match_role="treated"),
        sub.iloc[matched_control].assign(match_role="control"),
    ]).reset_index(drop=True)

    # ATT: mean difference in Y on matched sample
    Y_t = matched_df.loc[matched_df["match_role"] == "treated", "individual_score"].values
    Y_c = matched_df.loc[matched_df["match_role"] == "control", "individual_score"].values
    att_val = Y_t.mean() - Y_c.mean()
    t_stat, t_p = stats.ttest_rel(Y_t, Y_c)

    # Unmatched ATT for comparison
    att_unmatched = Y[D == 1].mean() - Y[D == 0].mean()

    print(f"  Unmatched ATT = {att_unmatched:.4f}")
    print(f"  Matched ATT   = {att_val:.4f}  t={t_stat:.3f}  p={t_p:.4f}")
    print(f"  n treated={len(idx_treated)}  n matched-control={len(idx_control)}")

    # Standardised mean differences (Love plot)
    smd_before, smd_after = _compute_smd(sub, matched_df, _PSM_FEATURES)
    _plot_love(smd_before, smd_after, _PSM_FEATURES)
    _plot_pscore_dist(sub)
    _plot_att_comparison(att_unmatched, att_val, t_p)

    return {
        "att_unmatched": att_unmatched,
        "att_matched":   att_val,
        "t_stat": t_stat,
        "t_p":    t_p,
        "n_treated": int(len(idx_treated)),
        "n_control": int(len(idx_control)),
    }


def _compute_smd(full: pd.DataFrame, matched: pd.DataFrame, features: list):
    smd_before = {}
    smd_after  = {}
    for f in features:
        t_all = full.loc[full["D"] == 1, f]
        c_all = full.loc[full["D"] == 0, f]
        pooled_std = np.sqrt((t_all.std() ** 2 + c_all.std() ** 2) / 2)
        smd_before[f] = (t_all.mean() - c_all.mean()) / (pooled_std + 1e-9)

        t_m = matched.loc[matched["match_role"] == "treated", f]
        c_m = matched.loc[matched["match_role"] == "control", f]
        smd_after[f]  = (t_m.mean() - c_m.mean()) / (pooled_std + 1e-9)
    return smd_before, smd_after


def _plot_love(smd_before: dict, smd_after: dict, features: list):
    fig, ax = plt.subplots(figsize=(5.7, 3.5))
    y = np.arange(len(features))
    ax.scatter([smd_before[f] for f in features], y, color="tomato",   s=80, label="Before matching", zorder=3)
    ax.scatter([smd_after[f]  for f in features], y, color="seagreen", s=80, label="After matching",  zorder=3)
    for i, f in enumerate(features):
        ax.plot([smd_before[f], smd_after[f]], [i, i], color="gray", lw=0.8, zorder=2)
    ax.axvline(0,    color="black",  lw=0.8)
    ax.axvline(0.1,  color="orange", lw=0.8, linestyle="--", label="|SMD| = 0.1 threshold")
    ax.axvline(-0.1, color="orange", lw=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels([f.replace("sim_", "sim ").replace("jury_", "jury ") for f in features], fontsize=7)
    ax.set_xlabel("Standardised Mean Difference (SMD)")
    ax.set_title("Love Plot: Covariate Balance Before & After PSM\n(|SMD| < 0.1 indicates good balance)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    _savefig("psm_love_plot.png")


def _plot_pscore_dist(sub: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.5))
    for ax, (title, mask) in zip(axes, [("Treated (jury_passed=True)",  sub["D"] == 1),
                                         ("Control (jury_passed=False)", sub["D"] == 0)]):
        ax.hist(sub.loc[mask, "pscore"], bins=20, color="steelblue" if "Control" in title else "tomato",
                alpha=0.75, edgecolor="white")
        ax.set_title(title, fontsize=7)
        ax.set_xlabel("Propensity Score")
        ax.set_ylabel("Count")
    fig.suptitle("Propensity Score Distributions by Treatment Status", fontsize=7)
    fig.tight_layout()
    _savefig("psm_score_dist.png")


def _plot_att_comparison(att_u: float, att_m: float, p: float):
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    bars = ax.bar(["Unmatched\nATT", "Matched\nATT"], [att_u, att_m],
                  color=["#a8c8e8", "#f4a261"], width=0.5, edgecolor="black")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Average Treatment Effect (individual_score difference)")
    ax.set_title(f"PSM: Effect of Jury-Confirmed Bias\nMatched ATT p = {p:.4f}")
    for bar, val in zip(bars, [att_u, att_m]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005 * np.sign(val),
                f"{val:.3f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=7)
    fig.tight_layout()
    _savefig("psm_att.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3. SYNTHETIC CONTROL
# ══════════════════════════════════════════════════════════════════════════════

_SCM_COVARIATES = [
    "sim_sentiment", "sim_assertiveness", "sim_emotional_intensity", "sim_word_count",
    "jury_target_rating", "jury_purity", "jury_naturalness",
]

def run_scm(df: pd.DataFrame) -> dict:
    """
    Synthetic Control Method.

    For each target bias B:
      - Treated unit  : diagonal cell (human_bias == B); outcome = mean delta_score
      - Donor pool    : off-diagonal cells (human_bias != B, same target bias B)
      - Covariates    : mean sim_ & jury_ features per cell
      - Weight optimisation: minimise ||X_treated − Σ wⱼ X_donor_j||²  s.t. wⱼ≥0, Σwⱼ=1
      - Gap           : delta_actual − delta_synthetic
      - Placebo test  : permute which cell is "treated" across all donors; build null
    """
    print("\n[3/3] Synthetic Control Method …")

    results = []
    placebo_gaps = []

    for bias in _BIAS_ORDER:
        sub = df[df["bias"] == bias].copy()
        # aggregate to cell level (human_bias × covariates)
        agg = sub.groupby("human_bias").agg(
            delta_score=("delta_score", "mean"),
            **{c: (c, "mean") for c in _SCM_COVARIATES if c in sub.columns},
        ).reset_index()

        treat_row = agg[agg["human_bias"] == bias]
        donor_rows = agg[agg["human_bias"] != bias]

        if len(treat_row) == 0 or len(donor_rows) < 2:
            results.append({"bias": bias, "actual": np.nan, "synthetic": np.nan, "gap": np.nan, "p_placebo": np.nan})
            continue

        avail_cov = [c for c in _SCM_COVARIATES if c in agg.columns]
        X_treat = treat_row[avail_cov].values.flatten()
        X_donors = donor_rows[avail_cov].values          # shape (n_donors, n_cov)
        Y_treat  = treat_row["delta_score"].values[0]
        Y_donors = donor_rows["delta_score"].values      # shape (n_donors,)

        # Optimise weights
        weights = _optimise_scm_weights(X_treat, X_donors)
        Y_synth  = float(np.dot(weights, Y_donors))
        gap      = float(Y_treat - Y_synth)

        # Placebo: each donor takes the "treated" role in turn
        donor_names = donor_rows["human_bias"].tolist()
        pg = []
        for k, (dk, yk) in enumerate(zip(donor_names, Y_donors)):
            X_k_treat  = X_donors[k]
            X_k_donors = np.delete(X_donors, k, axis=0)
            Y_k_donors = np.delete(Y_donors, k)
            if len(Y_k_donors) < 2:
                continue
            w_k = _optimise_scm_weights(X_k_treat, X_k_donors)
            synth_k = float(np.dot(w_k, Y_k_donors))
            pg.append(yk - synth_k)
        placebo_gaps.extend(pg)

        # p-value: fraction of placebo gaps >= |gap|
        p_val = np.mean(np.abs(pg) >= np.abs(gap)) if pg else np.nan

        results.append({
            "bias": bias, "actual": Y_treat, "synthetic": Y_synth,
            "gap": gap, "p_placebo": p_val, "weights": weights,
        })
        print(f"  {bias:30s}  actual={Y_treat:.3f}  synthetic={Y_synth:.3f}  gap={gap:+.3f}  p_placebo={p_val:.3f}")

    res_df = pd.DataFrame([{k: v for k, v in r.items() if k != "weights"} for r in results])
    _plot_scm_gap(res_df, placebo_gaps)
    _plot_scm_heatmap(res_df)
    _plot_scm_placebo(results, df)

    return {"results": res_df}


def _optimise_scm_weights(X_treat: np.ndarray, X_donors: np.ndarray) -> np.ndarray:
    n = len(X_donors)
    w0 = np.ones(n) / n

    # Normalise covariates by range to avoid scale dominance
    rng = X_donors.max(axis=0) - X_donors.min(axis=0)
    rng[rng == 0] = 1.0
    Xn_treat  = X_treat  / rng
    Xn_donors = X_donors / rng

    def objective(w):
        synth = w @ Xn_donors
        return np.sum((Xn_treat - synth) ** 2)

    constraints = [{"type": "eq",   "fun": lambda w: np.sum(w) - 1}]
    bounds      = [(0, 1)] * n
    sol = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints,
                   options={"ftol": 1e-9, "maxiter": 1000})
    return sol.x if sol.success else w0


def _plot_scm_gap(res_df: pd.DataFrame, placebo_gaps: list):
    valid = res_df.dropna(subset=["gap"])
    fig, ax = plt.subplots(figsize=(5.7, 3.5))
    colors = []
    for _, row in valid.iterrows():
        p = row["p_placebo"]
        if p < 0.05:
            colors.append("tomato")
        elif p < 0.10:
            colors.append("orange")
        else:
            colors.append("steelblue")

    x = np.arange(len(valid))
    ax.bar(x, valid["gap"], color=colors, alpha=0.85, edgecolor="black", width=0.6)
    ax.axhline(0, color="black", lw=1)

    # Placebo null distribution reference line (95th percentile of |gap|)
    if placebo_gaps:
        null_95 = np.percentile(np.abs(placebo_gaps), 95)
        ax.axhline( null_95, color="gray", lw=1, linestyle="--", label=f"Placebo 95th pct = {null_95:.3f}")
        ax.axhline(-null_95, color="gray", lw=1, linestyle="--")

    ax.set_xticks(x)
    ax.set_xticklabels(valid["bias"].str.replace(" ", "\n"), fontsize=7)
    ax.set_ylabel("Gap = Actual − Synthetic Δ-score")
    ax.set_title("Synthetic Control: Diagonal Gap per Target Bias\n(colour: p_placebo < 0.05 red, < 0.10 orange, ≥ 0.10 blue)")
    patches = [
        mpatches.Patch(color="tomato",    label="p < 0.05"),
        mpatches.Patch(color="orange",    label="0.05 ≤ p < 0.10"),
        mpatches.Patch(color="steelblue", label="p ≥ 0.10"),
    ]
    ax.legend(handles=patches + ([mpatches.Patch(color="gray", label=f"Placebo 95th pct")] if placebo_gaps else []),
              fontsize=7)
    fig.tight_layout()
    _savefig("scm_gap.png")


def _plot_scm_heatmap(res_df: pd.DataFrame):
    valid = res_df.dropna(subset=["actual", "synthetic"])
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.5))

    for ax, col, title in zip(
        axes,
        ["actual", "synthetic", "gap"],
        ["Actual Δ-score\n(diagonal)", "Synthetic Δ-score\n(donor weighted)", "Gap\n(actual − synthetic)"],
    ):
        data = valid.set_index("bias")[col].reindex(_BIAS_ORDER).to_frame().T
        vmax = valid[["actual", "synthetic", "gap"]].abs().max().max()
        sns.heatmap(data, ax=ax, annot=True, fmt=".3f", cmap="RdBu_r",
                    center=0, vmin=-vmax, vmax=vmax,
                    linewidths=0.5, cbar_kws={"shrink": 0.7})
        ax.set_title(title, fontsize=7)
        ax.set_xticklabels([b.replace(" ", "\n") for b in data.columns], fontsize=7, rotation=45, ha="right")
        ax.set_yticklabels([])

    fig.suptitle("Synthetic Control: Actual vs Synthetic Diagonal Δ-scores", fontsize=7)
    fig.tight_layout()
    _savefig("scm_heatmap.png")


def _plot_scm_placebo(results: list, df: pd.DataFrame):
    """
    For each target bias: bar chart of actual vs synthetic delta overlaid on
    placebo null distribution (violin).
    """
    valid = [r for r in results if not np.isnan(r.get("gap", np.nan))]
    if not valid:
        return

    n = len(valid)
    ncols = min(5, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.0, nrows * 2.5), squeeze=False)
    axes = axes.flatten()

    for i, r in enumerate(valid):
        ax   = axes[i]
        bias = r["bias"]
        sub  = df[df["bias"] == bias]
        agg  = sub.groupby("human_bias")["delta_score"].mean()
        donor_vals = agg[agg.index != bias].values

        ax.violinplot(donor_vals, positions=[0], showmedians=True,
                      widths=0.5)
        ax.bar([1], [r["actual"]],    color="tomato",   width=0.4, label="Actual",    alpha=0.9)
        ax.bar([2], [r["synthetic"]], color="seagreen", width=0.4, label="Synthetic", alpha=0.9)
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Donor\ndist.", "Actual\ndiag.", "Synthetic"], fontsize=7)
        ax.set_title(bias.replace(" ", "\n"), fontsize=7)
        ax.set_ylim(-1.1, 1.1)
        p = r.get("p_placebo", np.nan)
        if not np.isnan(p):
            ax.set_xlabel(f"p={p:.2f}", fontsize=7)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("SCM Placebo: Donor Distribution vs Actual & Synthetic Diagonal Δ-score", fontsize=7)
    fig.tight_layout()
    _savefig("scm_placebo_panels.png")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

def build_summary(did: dict, psm: dict, scm: dict) -> pd.DataFrame:
    rows = []
    pb = did["per_bias"]
    scm_df = scm["results"]
    for bias in _BIAS_ORDER:
        did_row = pb[pb["bias"] == bias].iloc[0] if len(pb[pb["bias"] == bias]) > 0 else {}
        scm_row = scm_df[scm_df["bias"] == bias].iloc[0] if len(scm_df[scm_df["bias"] == bias]) > 0 else {}
        rows.append({
            "bias":            bias,
            "did_att":         did_row.get("att",        np.nan),
            "did_se":          did_row.get("se",         np.nan),
            "did_p":           did_row.get("p",          np.nan),
            "did_ci_lo":       did_row.get("ci_lo",      np.nan),
            "did_ci_hi":       did_row.get("ci_hi",      np.nan),
            "scm_actual":      scm_row.get("actual",     np.nan),
            "scm_synthetic":   scm_row.get("synthetic",  np.nan),
            "scm_gap":         scm_row.get("gap",        np.nan),
            "scm_p_placebo":   scm_row.get("p_placebo",  np.nan),
        })
    rows.append({
        "bias":       "__GLOBAL__",
        "did_att":    did["global_att"],
        "did_se":     did["global_se"],
        "did_p":      did["global_p"],
        "did_ci_lo":  did["global_ci"][0],
        "did_ci_hi":  did["global_ci"][1],
        "scm_actual": np.nan, "scm_synthetic": np.nan,
        "scm_gap": np.nan,    "scm_p_placebo": np.nan,
    })
    # PSM is global
    psm_row = {k: v for k, v in psm.items() if not isinstance(v, np.ndarray)}
    summary = pd.DataFrame(rows)
    for k, v in psm_row.items():
        summary[f"psm_{k}"] = np.where(summary["bias"] == "__GLOBAL__", v, np.nan)

    os.makedirs(os.path.dirname(_SUMMARY_CSV), exist_ok=True)
    summary.to_csv(_SUMMARY_CSV, index=False)
    print(f"\nSummary saved → {_SUMMARY_CSV}")
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(pilot_csv: str = _PILOT_CSV):
    os.makedirs(_PLOT_DIR, exist_ok=True)
    print(f"Loading data from {pilot_csv} …")
    df = load_data(pilot_csv)
    print(f"  {len(df)} OK rows | {df['bias'].nunique()} target biases | {df['human_bias'].nunique()} human biases")

    did = run_did(df)
    psm = run_psm(df)
    scm = run_scm(df)

    summary = build_summary(did, psm, scm)

    print("\n" + "=" * 60)
    print("CAUSAL ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Plots  → {_PLOT_DIR}")
    print(f"Summary→ {_SUMMARY_CSV}")
    print()
    print("── DiD (global) ──────────────────────────────────────────")
    print(f"  ATT = {did['global_att']:.4f}  SE={did['global_se']:.4f}  p={did['global_p']:.4f}")
    print(f"  95% CI [{did['global_ci'][0]:.4f}, {did['global_ci'][1]:.4f}]")
    print()
    print("── PSM ───────────────────────────────────────────────────")
    print(f"  Unmatched ATT = {psm['att_unmatched']:.4f}")
    print(f"  Matched ATT   = {psm['att_matched']:.4f}  (p={psm['t_p']:.4f})")
    print()
    print("── SCM (diagonal gaps) ───────────────────────────────────")
    scm_res = scm["results"].dropna(subset=["gap"])
    sig = scm_res[scm_res["p_placebo"] < 0.05]
    print(f"  Biases with significant diagonal gap (p<0.05): {sig['bias'].tolist()}")
    print(f"  Mean |gap| = {scm_res['gap'].abs().mean():.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Causal inference analysis for conditional bias experiment.")
    parser.add_argument("--input", default=_PILOT_CSV, help="Path to pilot CSV file.")
    args = parser.parse_args()
    main(args.input)
