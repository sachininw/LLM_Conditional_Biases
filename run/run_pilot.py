"""
Run the full conditional bias pilot: every bias × every bias, N samples each.

Two-phase workflow
------------------
Phase 1 — Message bank (run once, reuse across all LLMs):
    Build validated_message_bank.csv with jury-approved human messages for every
    (scenario × human_bias) pair. Jury calls are never repeated for the same bank.

Phase 2 — LLM evaluation (run once per model):
    Read from the bank and execute only the 3 LLM calls per row (control,
    treatment, neutral). No jury involved. Swap --model to replicate on any LLM.

Usage:
    # Full run: build bank (100 samples/bias) then evaluate GPT-4o
    python run_pilot.py --n_samples 100 --model GPT-4o

    # Provide a pre-built bank (skip Phase 1) and evaluate a different model
    python run_pilot.py --message_bank data/validated_message_bank.csv --model GPT-4o-Mini

    # Provide a pre-built scenario dataset (skip HF download but still run jury)
    python run_pilot.py --dataset data/pilot_dataset.csv --n_samples 100 --model GPT-4o
"""

import sys
import os
import ast
import datetime
import concurrent.futures
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from conditional_test_decision import (
    decide_conditional_batch,
    decide_from_bank_batch,
    evaluate_bank,
    CONDITIONAL_RESULTS_DIR,
    _RUN_DIR,
    _REPO_ROOT,
)
from generate_message_bank import generate_message_bank
from prepare_dataset import convert_hf_to_csv

_DEFAULT_BIASES_FILE = os.path.join(os.path.dirname(_REPO_ROOT), "Biases.txt")
_DEFAULT_DATASET_PATH = os.path.join(_RUN_DIR, "data", "pilot_dataset.csv")
_DEFAULT_BANK_PATH = os.path.join(_RUN_DIR, "data", "validated_message_bank.csv")

_BIAS_NAME_MAP = {
    "In-Group Bias": "In Group Bias",
    "Status-Quo Bias": "Status Quo Bias",
}


def _read_biases(path: str) -> list:
    with open(path) as f:
        raw = [line.strip() for line in f if line.strip()]
    return [_BIAS_NAME_MAP.get(b, b) for b in raw]


def _build_bias_direction_map(dataset: pd.DataFrame) -> dict[str, str]:
    """Return {bias_name: 'positive'|'negative'} by reading k from metric_params."""
    mapping: dict[str, str] = {}
    for _, row in dataset.iterrows():
        bias = row.get("bias", "")
        if bias in mapping:
            continue
        try:
            mp = row.get("metric_params", "{}")
            if isinstance(mp, str):
                mp = ast.literal_eval(mp)
            k = int(mp.get("k", 0))
            if k != 0:
                mapping[bias] = "negative" if k == -1 else "positive"
        except Exception:
            pass
    return mapping


def _run_phase1_legacy(
    biases, dataset, bias_direction, model_name, direction, intensity,
    temperature, seed, max_jury_retries, n_workers, batch_dir,
):
    """
    Legacy single-phase pipeline: jury + LLM evaluation interleaved.
    Used when --message_bank is not provided and the bank is not pre-built.
    Results go to the same batch directory as Phase 2.
    """
    batches = [
        b for b in np.array_split(dataset, min(n_workers, len(dataset))) if len(b) > 0
    ]
    n = len(biases)
    for idx, human_bias in enumerate(biases, 1):
        human_direction = bias_direction.get(human_bias, direction)
        t0 = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{t0}] ({idx}/{n}) human_bias = {human_bias!r}  direction = {human_direction!r}")
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            for _ in executor.map(
                partial(
                    decide_conditional_batch,
                    model_name=model_name,
                    human_bias=human_bias,
                    direction=human_direction,
                    intensity=intensity,
                    randomly_flip_options=True,
                    shuffle_answer_options=False,
                    temperature=temperature,
                    seed=seed,
                    max_jury_retries=max_jury_retries,
                ),
                batches,
            ):
                pass


def _run_phase2_from_bank(
    bank: pd.DataFrame,
    biases: list,
    model_name: str,
    temperature: float,
    seed: int,
    n_workers: int,
    row_workers: int = 20,
):
    """
    Phase 2: LLM evaluation from a pre-validated message bank.
    Uses a flat ThreadPoolExecutor so the progress bar advances one row at a time.
    Filters to biases listed in Biases.txt.
    """
    bank = bank[bank["bias"].isin(biases) & bank["human_bias"].isin(biases)].reset_index(drop=True)
    total_workers = n_workers * row_workers
    n_unique = bank["scenario_id"].nunique() if "scenario_id" in bank.columns else len(bank)
    n_rows = len(bank)
    print(f"\n{'='*60}")
    print(f"  Model      : {model_name}")
    print(f"  Bank rows  : {n_rows:,}  (after bias filter)")
    print(f"  Unique scen: {n_unique:,}")
    print(f"  Workers    : {total_workers}  ({n_workers} processes × {row_workers} threads)")
    print(f"  API calls  : ~{n_unique * 4 + n_rows * 2:,}  "
          f"(cache: {n_unique * 4:,}  +  biased: {n_rows * 2:,})")
    print(f"{'='*60}\n")

    evaluate_bank(
        bank_df=bank,
        model_name=model_name,
        randomly_flip_options=True,
        shuffle_answer_options=False,
        temperature=temperature,
        seed=seed,
        total_workers=total_workers,
        on_row_done=None,
    )


def run_pilot(
    biases_file: str = _DEFAULT_BIASES_FILE,
    model_name: str = "GPT-4o",
    n_samples: int = 5,
    n_workers: int = 5,
    row_workers: int = 20,
    direction: str = "positive",
    intensity: float = 0.7,
    temperature: float = 0.0,
    seed: int = 42,
    max_jury_retries: int = 3,
    dataset_path: str = None,
    message_bank_path: str = None,
):
    """
    Run the full conditional bias pilot.

    If message_bank_path points to an existing validated_message_bank.csv, Phase 1
    (jury validation) is skipped and evaluation runs directly from the bank.
    Otherwise, the bank is built first (or the legacy pipeline is used).

    Args:
        biases_file:       Path to Biases.txt.
        model_name:        LLM to evaluate (Phase 2 only).
        n_samples:         Scenarios per target bias (Phase 1 dataset build).
        n_workers:         Parallel worker processes.
        direction:         Fallback bias direction if not found in metric_params.
        intensity:         Human simulator bias intensity.
        temperature:       LLM sampling temperature.
        seed:              Base random seed.
        max_jury_retries:  Max jury attempts per (scenario, human_bias) pair.
        dataset_path:      Pre-built scenario CSV (skips HF download).
        message_bank_path: Pre-built validated_message_bank.csv (skips Phase 1).
    """
    biases = _read_biases(biases_file)
    n = len(biases)
    print(f"Biases ({n}): {biases}")

    # --- Prepare batch directory (clear stale batch files) ---
    batch_dir = os.path.join(CONDITIONAL_RESULTS_DIR, model_name)
    os.makedirs(batch_dir, exist_ok=True)
    stale = [f for f in os.listdir(batch_dir) if f.startswith("batch_") and f.endswith(".csv")]
    if stale:
        print(f"Removing {len(stale)} stale batch files from previous run...")
        for f in stale:
            os.remove(os.path.join(batch_dir, f))

    start = datetime.datetime.now()

    # ── Phase 1: build or load the validated message bank ────────────────────
    if message_bank_path and os.path.exists(message_bank_path):
        print(f"\n[Phase 1] Using existing message bank: {message_bank_path}")
        bank = pd.read_csv(message_bank_path)
        print(f"  Bank rows: {len(bank)}")
        use_bank = True

    else:
        bank_path = message_bank_path or _DEFAULT_BANK_PATH
        print(f"\n[Phase 1] Building validated message bank → {bank_path}")
        print(f"  Matrix: {n} × {n} = {n * n} cells | {n_samples} samples per target bias")
        print(f"  Estimated jury tasks: ~{n * n * n_samples}\n")

        if dataset_path is None:
            dataset_path = _DEFAULT_DATASET_PATH
        if not os.path.exists(dataset_path):
            print("  Downloading scenario dataset from HuggingFace...")
            convert_hf_to_csv(
                output_path=dataset_path,
                bias_filter=",".join(biases),
                n_samples=n_samples,
            )
        else:
            print(f"  Using existing dataset: {dataset_path}")

        bank = generate_message_bank(
            dataset_path=dataset_path,
            output_path=bank_path,
            biases_file=biases_file,
            n_samples=n_samples,
            intensity=intensity,
            seed=seed,
            max_jury_retries=max_jury_retries,
            n_workers=n_workers,
        )
        use_bank = True

    # ── Phase 2: LLM evaluation from bank ────────────────────────────────────
    print(f"\n[Phase 2] Evaluating {model_name} on {len(bank)} jury-passed rows...")
    print(f"  Estimated LLM calls: ~{len(bank) * 3} (control + treatment + neutral)\n")

    _run_phase2_from_bank(
        bank=bank,
        biases=biases,
        model_name=model_name,
        temperature=temperature,
        seed=seed,
        n_workers=n_workers,
        row_workers=row_workers,
    )

    # ── Merge batch CSVs into one pilot file ──────────────────────────────────
    batch_files = [
        os.path.join(batch_dir, f)
        for f in os.listdir(batch_dir)
        if f.startswith("batch_") and f.endswith(".csv")
    ]
    if not batch_files:
        print("No batch files found — nothing to merge.")
        return pd.DataFrame()

    merged = pd.concat([pd.read_csv(f) for f in batch_files], ignore_index=True)
    out_path = os.path.join(CONDITIONAL_RESULTS_DIR, f"{model_name}_pilot.csv")
    merged.to_csv(out_path, index=False)

    elapsed = datetime.datetime.now() - start
    print(f"\nDone in {elapsed}.")
    print(f"Total rows: {len(merged)} | saved to {out_path}")

    coverage = merged.groupby(["bias", "human_bias"]).size().unstack(fill_value=0)
    print("\nCoverage (rows per cell):")
    print(coverage.to_string())

    return merged


def retry_errors(
    pilot_csv: str,
    model_name: str,
    temperature: float = 0.0,
    seed: int = 42,
    n_workers: int = 5,
    row_workers: int = 20,
    message_bank_path: str = None,
):
    """
    Re-run only the ERROR rows in an existing pilot CSV.

    Reads pilot_csv, extracts ERROR rows, evaluates them with evaluate_bank,
    then merges the new results back into pilot_csv (replacing ERROR rows).

    Usage:
        python run_pilot.py --retry data/conditional_decision_results/Claude-3.5-Haiku_pilot.csv \\
            --model Claude-3.5-Haiku
    """
    pilot = pd.read_csv(pilot_csv)
    error_rows = pilot[pilot["status"] == "ERROR"]
    if error_rows.empty:
        print("No ERROR rows — nothing to retry.")
        return pilot

    print(f"Retrying {len(error_rows)} ERROR rows from {pilot_csv}")

    # ERROR rows don't have bank columns — load them from the message bank.
    # The pilot uses "id" as the scenario key; the bank uses "scenario_id".
    bank_path = message_bank_path or _DEFAULT_BANK_PATH
    bank = pd.read_csv(bank_path)

    # Match ERROR rows to bank by (scenario_id, bias, human_bias).
    # The pilot's "id" column is NaN (bank rows use "scenario_id" as primary key);
    # "scenario_id" is populated in both OK and ERROR rows.
    merge_keys = ["scenario_id", "bias", "human_bias"]
    error_with_bank = error_rows[merge_keys].merge(bank, on=merge_keys, how="left")
    missing = error_with_bank["raw_treatment"].isna().sum()
    if missing:
        print(f"  Warning: {missing} rows not found in bank — they will remain ERROR")
        error_with_bank = error_with_bank.dropna(subset=["raw_treatment"])

    if error_with_bank.empty:
        print("No matchable ERROR rows in bank.")
        return pilot

    total_workers = n_workers * row_workers
    print(f"Evaluating {len(error_with_bank)} rows with {total_workers} threads...")

    with tqdm(total=len(error_with_bank), unit="row", desc="Retry", dynamic_ncols=True) as pbar:
        evaluate_bank(
            bank_df=error_with_bank,
            model_name=model_name,
            randomly_flip_options=True,
            shuffle_answer_options=False,
            temperature=temperature,
            seed=seed,
            total_workers=total_workers,
            on_row_done=lambda: pbar.update(1),
        )

    # Load the just-written batch file(s) and merge back
    batch_dir = os.path.join(CONDITIONAL_RESULTS_DIR, model_name)
    batch_files = [
        os.path.join(batch_dir, f)
        for f in os.listdir(batch_dir)
        if f.startswith("batch_") and f.endswith(".csv")
    ]
    if not batch_files:
        print("No batch files written — retry may have failed silently.")
        return pilot

    retried = pd.concat([pd.read_csv(f) for f in batch_files], ignore_index=True)
    for f in batch_files:
        os.remove(f)

    # Drop ERROR rows from the original pilot and replace with retried results
    pilot_ok = pilot[pilot["status"] != "ERROR"]
    pilot = pd.concat([pilot_ok, retried], ignore_index=True)
    pilot.to_csv(pilot_csv, index=False)

    ok_count = (pilot["status"] == "OK").sum()
    err_count = (pilot["status"] == "ERROR").sum()
    print(f"\nDone. OK={ok_count} | ERROR={err_count} | saved to {pilot_csv}")
    return pilot


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the full conditional bias pilot (every bias × every bias)."
    )
    parser.add_argument(
        "--biases_file", type=str, default=_DEFAULT_BIASES_FILE,
        help="Path to Biases.txt.",
    )
    parser.add_argument("--model", type=str, default="GPT-4o")
    parser.add_argument(
        "--n_samples", type=int, default=5,
        help="Scenarios per target bias (used in Phase 1 dataset build).",
    )
    parser.add_argument(
        "--n_workers", type=int, default=5,
        help="Parallel worker processes.",
    )
    parser.add_argument(
        "--row_workers", type=int, default=20,
        help="Concurrent threads per worker process for row-level API calls.",
    )
    parser.add_argument(
        "--direction", type=str, default="positive",
        choices=["positive", "negative"],
        help="Fallback bias direction if not derivable from metric_params.",
    )
    parser.add_argument("--intensity", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_jury_retries", type=int, default=3)
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Path to a pre-built scenario dataset CSV (skips HuggingFace download).",
    )
    parser.add_argument(
        "--message_bank", type=str, default=None,
        help=(
            "Path to a pre-built validated_message_bank.csv. "
            "If provided and file exists, Phase 1 (jury validation) is skipped entirely. "
            "Reuse the same bank to evaluate multiple LLMs without re-running jury."
        ),
    )
    parser.add_argument(
        "--retry", type=str, default=None,
        help=(
            "Path to an existing pilot CSV. Re-evaluates only ERROR rows. "
            "Use after a run with rate-limit failures. Requires --model to match."
        ),
    )
    args = parser.parse_args()

    if args.retry:
        retry_errors(
            pilot_csv=args.retry,
            model_name=args.model,
            temperature=args.temperature,
            seed=args.seed,
            n_workers=args.n_workers,
            row_workers=args.row_workers,
            message_bank_path=args.message_bank,
        )
    else:
        run_pilot(
            biases_file=args.biases_file,
            model_name=args.model,
            n_samples=args.n_samples,
            n_workers=args.n_workers,
            row_workers=args.row_workers,
            direction=args.direction,
            intensity=args.intensity,
            temperature=args.temperature,
            seed=args.seed,
            max_jury_retries=args.max_jury_retries,
            dataset_path=args.dataset,
            message_bank_path=args.message_bank,
        )
