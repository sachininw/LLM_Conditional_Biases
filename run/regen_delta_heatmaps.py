"""
Regenerate per-model delta (content-effect) heatmaps with square=True.
Copies output PDFs to the latex figures directories.
"""
import os, sys, shutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── Paths ─────────────────────────────────────────────────────────────────────
_RUN_DIR   = os.path.dirname(os.path.abspath(__file__))
_LATEX_DIR = os.path.join(_RUN_DIR, "..", "..", "emnlp26_working", "latex", "figures")

_BIAS_ORDER = [
    "Anchoring", "Framing Effect", "Availability Heuristic",
    "Confirmation Bias", "Loss Aversion", "Planning Fallacy",
    "In Group Bias", "Bandwagon Effect", "Status Quo Bias",
]
_SHORT = {
    "Anchoring":             "Anchor.",
    "Availability Heuristic":"Avail.",
    "Bandwagon Effect":      "Bandwgn",
    "Confirmation Bias":     "Confirm.",
    "Framing Effect":        "Framing",
    "In Group Bias":         "InGroup",
    "Loss Aversion":         "Loss Av.",
    "Planning Fallacy":      "Plan.",
    "Status Quo Bias":       "Stat.Quo",
}

# model_name → (csv_rel_path, latex_fig_dir)
MODELS = {
    "Llama-3.1-8B":         ("data/conditional_decision_results/Llama-3.1-8B_pilot.csv",           "llama"),
    "Llama-3.1-70B":        ("data/conditional_decision_results/Llama-3.1-70B_pilot.csv",          "llama-3-1-70b"),
    "Claude-3.5-Haiku":     ("data/conditional_decision_results/Claude-3.5-Haiku_pilot.csv",       "claude"),
    "GPT-4o":               ("data/conditional_decision_results/GPT-4o_pilot.csv",                 "gpt-4o"),
    "DeepSeek-V3":          ("data/conditional_decision_results/DeepSeek-V3_pilot.csv",            "deepseek-v3"),
    "Qwen-2.5-72B-Instruct":("data/conditional_decision_results/Qwen-2.5-72B-Instruct_pilot.csv", "qwen-2-5-72b-instruct"),
    "Phi-3.5":              ("data/conditional_decision_results/Phi-3.5_pilot.csv",                "phi-3-5"),
    "Gemma-2-9B-IT":        ("data/conditional_decision_results/Gemma-2-9B-IT_pilot.csv",         "gemma-2-9b-it"),
}


def _load_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    ok = df[df["status"] == "OK"]
    jp = ok[ok["jury_passed"] == True].copy()
    if "individual_score" in jp.columns and "score_biased" not in jp.columns:
        jp = jp.rename(columns={"individual_score": "score_biased",
                                 "neutral_score":    "score_neutral"})
    if "score_biased" in jp.columns and "score_neutral" in jp.columns:
        jp["delta_score"] = jp["score_biased"].abs() - jp["score_neutral"].abs()
    return jp


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


def _make_delta_heatmap(df: pd.DataFrame, model_name: str, out_pdf: str):
    if "delta_score" not in df.columns:
        print(f"  {model_name}: delta_score column missing, skipping")
        return
    pivot = _impute_pivot(df.pivot_table(
        values="delta_score", index="bias", columns="human_bias", aggfunc="mean"
    ).reindex(index=_BIAS_ORDER, columns=_BIAS_ORDER))

    # Add row/column averages
    pivot_with_avg = pivot.copy()
    pivot_with_avg["Average"] = pivot.mean(axis=1)
    avg_row = pivot_with_avg.mean(axis=0)
    pivot_with_avg.loc["Average"] = avg_row

    short_rows = [_SHORT.get(b, b) for b in pivot_with_avg.index]
    short_cols = [_SHORT.get(b, b) for b in pivot_with_avg.columns]

    fig, ax = plt.subplots(figsize=(5.7, 4.5))
    sns.heatmap(
        pivot_with_avg.round(3),
        cmap=sns.diverging_palette(220, 20, as_cmap=True),
        center=0, annot=True, fmt=".2f",
        square=True,
        vmin=-1, vmax=1,
        linewidths=0.4, linecolor="white",
        cbar_kws={"shrink": 0.7, "label": "Score"},
        ax=ax,
        xticklabels=short_cols,
        yticklabels=short_rows,
    )
    ax.set_xticklabels(short_cols, rotation=45, ha="left")
    ax.set_yticklabels(short_rows, rotation=0)
    ax.set_title(
        f"Content Effect (δ = |m_b| − |m_n|)\n{model_name}",
        pad=10
    )
    ax.set_xlabel("Human Bias", labelpad=8)
    ax.set_ylabel("Target Bias", labelpad=8)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    plt.savefig(out_pdf, format="pdf", dpi=600, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out_pdf}")


def main():
    for model_name, (rel_csv, latex_subdir) in MODELS.items():
        csv_path = os.path.join(_RUN_DIR, rel_csv)
        if not os.path.exists(csv_path):
            print(f"[skip] {model_name}: CSV not found at {csv_path}")
            continue

        # Output path in results/
        results_dir = os.path.join(_RUN_DIR, "data", "results", model_name, "plots")
        os.makedirs(results_dir, exist_ok=True)
        out_pdf = os.path.join(results_dir, "heatmap_delta_jp.pdf")

        print(f"\n[{model_name}] Loading {csv_path} ...")
        df = _load_df(csv_path)
        _make_delta_heatmap(df, model_name, out_pdf)

        # Copy to latex figures directory
        latex_out = os.path.join(_LATEX_DIR, latex_subdir, "heatmap_delta_jp.pdf")
        os.makedirs(os.path.dirname(latex_out), exist_ok=True)
        shutil.copy2(out_pdf, latex_out)
        print(f"  copied → {latex_out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
