"""
Regenerate the model effect space scatter (bubble) plot.
x-axis: content effect (delta)
y-axis: presence effect (Delta_n)
bubble size: MMLU score
color: alignment recipe
"""
import os, shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from adjustText import adjust_text

_RUN_DIR   = os.path.dirname(os.path.abspath(__file__))
_LATEX_DIR = os.path.join(_RUN_DIR, "..", "..", "emnlp26_working", "latex", "figures", "cross_model")

# From Table 1
MODELS = {
    "Llama-3.1-8B":     {"delta_n": +0.086, "delta": -0.035, "mmlu": 66.7},
    "Llama-3.1-70B":    {"delta_n": +0.061, "delta": -0.039, "mmlu": 79.3},
    "Claude-3.5-Haiku": {"delta_n": -0.101, "delta": +0.037, "mmlu": 77.0},
    "GPT-4o":           {"delta_n": +0.019, "delta": -0.020, "mmlu": 88.0},
    "DeepSeek-V3":      {"delta_n": +0.025, "delta": +0.009, "mmlu": 87.1},
    "Qwen-2.5-72B":     {"delta_n": +0.046, "delta": -0.007, "mmlu": 83.0},
    "Phi-4":            {"delta_n": +0.052, "delta": -0.018, "mmlu": 84.8},
    "Gemma-2-9B-IT":    {"delta_n": +0.058, "delta": -0.028, "mmlu": 68.4},
}

RECIPE_COLOR = {
    "Llama-3.1-8B":     "#4393c3",
    "Llama-3.1-70B":    "#4393c3",
    "Claude-3.5-Haiku": "#d6604d",
    "GPT-4o":           "#74c476",
    "DeepSeek-V3":      "#f4a442",
    "Qwen-2.5-72B":     "#4393c3",
    "Phi-4":            "#74c476",
    "Gemma-2-9B-IT":    "#4393c3",
}

SHORT = {
    "Llama-3.1-8B":     "Llama-8B",
    "Llama-3.1-70B":    "Llama-70B",
    "Claude-3.5-Haiku": "Claude-Haiku",
    "GPT-4o":           "GPT-4o",
    "DeepSeek-V3":      "DeepSeek-V3",
    "Qwen-2.5-72B":     "Qwen-72B",
    "Phi-4":            "Phi-4",
    "Gemma-2-9B-IT":    "Gemma-9B",
}


def main():
    fig, ax = plt.subplots(figsize=(3.2, 2.6))

    texts = []
    for name, d in MODELS.items():
        mmlu_min, mmlu_max = 66.7, 88.0
        size = 80 + (d["mmlu"] - mmlu_min) / (mmlu_max - mmlu_min) * 80  # 80–160 pt²
        ax.scatter(d["delta"], d["delta_n"],
                   s=size, c=RECIPE_COLOR[name],
                   alpha=0.9, edgecolors="white", linewidths=0.5, zorder=3)
        texts.append(ax.text(d["delta"], d["delta_n"], SHORT[name],
                             fontsize=6.5, zorder=4))

    adjust_text(
        texts, ax=ax,
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, shrinkA=5),
        expand_points=(1.6, 1.6),
        expand_text=(1.4, 1.4),
        force_points=(0.4, 0.4),
        force_text=(0.6, 0.6),
    )

    ax.axhline(0, color="black", lw=0.7, ls="--", alpha=0.5)
    ax.axvline(0, color="black", lw=0.9, ls="-", alpha=0.7)

    ax.set_xlabel("Content effect $\\bar{\\delta}$", fontsize=6)
    ax.set_ylabel("Presence effect $\\bar{\\Delta}_n$", fontsize=6)
    ax.set_title("Model Effect Space (bubble size $\\propto$ MMLU)", fontsize=6, pad=4)
    ax.tick_params(labelsize=5)

    legend_patches = [
        mpatches.Patch(color="#4393c3", label="SFT/RLHF"),
        mpatches.Patch(color="#d6604d", label="Constitutional AI"),
        mpatches.Patch(color="#74c476", label="RLHF (GPT/Phi)"),
        mpatches.Patch(color="#f4a442", label="CoT RLFT"),
    ]
    ax.legend(handles=legend_patches, fontsize=6, loc="lower left",
              borderaxespad=0.5, framealpha=0.85, handlelength=1.0)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    out = os.path.join(_RUN_DIR, "data", "results", "cross_model", "model_effect_space.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, format="pdf", dpi=600, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out}")

    latex_out = os.path.join(_LATEX_DIR, "model_effect_space.pdf")
    shutil.copy2(out, latex_out)
    print(f"  copied → {latex_out}")


if __name__ == "__main__":
    main()
