"""
Generate and jury-validate human messages for all (target_bias × human_bias) cells.

Processing order
----------------
Rounds: 1 → n_samples_target (e.g. 100)

Each round:
  For each target_bias (in rotation):
    For each human_bias (in parallel, n_workers threads):
      - Try the next available scenario for this (target_bias, human_bias) cell
      - If jury passes  → write the row to CSV immediately, cell is done for this round
      - If jury fails   → discard that scenario, try the next one for the same cell
      - After max_consecutive_fails (default 10) consecutive failures, skip the cell
        for this round and retry in the next round

At the end of round 1 every (target_bias × human_bias) cell has exactly 1 row.
After 100 rounds every cell has 100 rows (10 × 10 × 100 = 10 000 total rows).

Each cell maintains its own independent scenario pointer, so a jury failure for
(Anchoring, Framing Effect) does not affect (Anchoring, Anchoring).

Incremental runs
----------------
If the bank already exists, per-cell counts are read from it.
Scenarios whose scenario_id is already used by a cell are skipped for that cell.
Re-running after interruption continues from where it stopped.

Dataset size note
-----------------
Each cell needs n_samples_target unique passing scenarios. With an ~80% jury pass
rate you need ~125 scenarios per target_bias. Download at least 150:
    python prepare_dataset.py --n_samples 150 --output data/full_dataset.csv
"""

import sys
import os
import ast
import threading
import datetime
import argparse
import concurrent.futures

import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from human_simulator import HumanSimulator
from bias_jury import BiasJury
from conditional_test_decision import (
    extract_common_and_diverging,
    _build_jury_feedback,
    _RUN_DIR,
    _REPO_ROOT,
)
from core.testing import Template
from prepare_dataset import convert_hf_to_csv

_DEFAULT_BIASES_FILE = os.path.join(os.path.dirname(_REPO_ROOT), "Biases.txt")
_DEFAULT_DATASET     = os.path.join(_RUN_DIR, "data", "full_dataset.csv")
_DEFAULT_BANK        = os.path.join(_RUN_DIR, "data", "validated_message_bank.csv")

_BIAS_NAME_MAP = {
    "In-Group Bias":   "In Group Bias",
    "Status-Quo Bias": "Status Quo Bias",
}

_RELEASE_COLUMNS = [
    "scenario_id",
    "bias", "scenario", "control", "treatment", "metric_params",
    "human_bias", "direction", "intensity",
    "common_situation", "diverging_situation",
    "human_message", "neutral_message",
    "jury_target_rating", "jury_purity", "jury_naturalness",
    "jury_blind_accuracy", "jury_krippendorff_alpha",
    "sim_sentiment", "sim_assertiveness", "sim_emotional_intensity", "sim_word_count",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_biases(path: str) -> list:
    with open(path) as f:
        raw = [line.strip() for line in f if line.strip()]
    return [_BIAS_NAME_MAP.get(b, b) for b in raw]


def _build_bias_direction_map(dataset: pd.DataFrame) -> dict:
    mapping = {}
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


def _write_row(result: dict, output_path: str, release_path: str, lock: threading.Lock):
    """Append one jury-passed row to both CSVs immediately under the write lock."""
    row_df = pd.DataFrame([result])
    with lock:
        write_header = not os.path.exists(output_path)
        row_df.to_csv(output_path, mode="a", header=write_header, index=False)
        release_cols = [c for c in _RELEASE_COLUMNS if c in row_df.columns]
        row_df[release_cols].to_csv(
            release_path, mode="a",
            header=not os.path.exists(release_path),
            index=False,
        )


# ── Per-cell worker ───────────────────────────────────────────────────────────

def _fill_cell_one_round(
    target_bias: str,
    human_bias: str,
    direction: str,
    cell_scenarios: list,       # this cell's remaining scenarios (mutated in place)
    intensity: float,
    seed: int,
    max_jury_retries: int,
    output_path: str,
    release_path: str,
    write_lock: threading.Lock,
    max_consecutive_fails: int = 2,
    neutral_cache: dict = None,      # shared across cells: scenario_id → neutral_message
    neutral_cache_lock: threading.Lock = None,
) -> bool:
    """
    Attempt to get one jury-passed row for a (target_bias, human_bias) cell.

    Pops scenarios from cell_scenarios until jury passes, the list is empty,
    or max_consecutive_fails consecutive scenario-level failures are hit.

    Returns True if a row was written, False otherwise.
    Skipped cells are retried in the next round.
    """
    os.chdir(_REPO_ROOT)
    simulator = HumanSimulator()
    jury      = BiasJury()

    consecutive_fails = 0

    while cell_scenarios:
        row_dict = cell_scenarios.pop(0)
        row      = pd.Series(row_dict)

        try:
            ctrl_tmpl  = Template(row["raw_control"])
            treat_tmpl = Template(row["raw_treatment"])
            common_text, _, treat_diverging = extract_common_and_diverging(
                ctrl_tmpl, treat_tmpl, str(row.get("scenario", ""))
            )
        except Exception:
            consecutive_fails += 1
            if consecutive_fails >= max_consecutive_fails:
                return False
            continue

        # Try to get a jury-passing biased message
        sim_message, jury_result, jury_passed = None, None, False
        feedback = ""
        for attempt in range(max_jury_retries):
            candidate  = simulator.generate(
                situation=common_text,
                bias_name=human_bias,
                direction=direction,
                intensity=intensity,
                seed=seed + attempt,
                feedback=feedback,
            )
            evaluation = jury.evaluate(candidate, common_text, human_bias, seed=seed + attempt)
            if jury.is_valid(evaluation):
                sim_message, jury_result, jury_passed = candidate, evaluation, True
                break
            feedback = _build_jury_feedback(evaluation, human_bias)

        if not jury_passed:
            consecutive_fails += 1
            if consecutive_fails >= max_consecutive_fails:
                tqdm.write(
                    f"  Skipping ({target_bias} × {human_bias}) this round — "
                    f"{max_consecutive_fails} consecutive jury failures."
                )
                return False
            continue

        consecutive_fails = 0

        # Neutral message: generate once per scenario and reuse across all
        # human_bias variants so the neutral baseline is identical for the same scenario.
        scenario_id = row.get("id")
        neutral_message = None
        if neutral_cache is not None:
            with neutral_cache_lock:
                neutral_message = neutral_cache.get(scenario_id)
        if neutral_message is None:
            neutral_message, _, _ = simulator.generate_neutral(
                situation=common_text, seed=seed + 100, jury=jury
            )
            if neutral_cache is not None:
                with neutral_cache_lock:
                    neutral_cache[scenario_id] = neutral_message

        jr = jury_result or {}
        lf = jr.get("linguistic_features", {})

        result = {
            "scenario_id":   row.get("id"),
            "bias":          target_bias,
            "scenario":      str(row.get("scenario", "")),
            "control":       row.get("control", ""),
            "treatment":     row.get("treatment", ""),
            "metric_params": row.get("metric_params", "{}"),
            "raw_control":   row.get("raw_control"),
            "raw_treatment": row.get("raw_treatment"),
            "seed":          seed,
            "common_situation":    common_text,
            "diverging_situation": treat_diverging,
            "human_bias":  human_bias,
            "direction":   direction,
            "intensity":   intensity,
            "human_message":           sim_message,
            "jury_passed":             True,
            "jury_target_rating":      jr.get("target_rating"),
            "jury_purity":             jr.get("purity"),
            "jury_naturalness":        jr.get("naturalness"),
            "jury_blind_accuracy":     jr.get("blind_accuracy"),
            "jury_krippendorff_alpha": jr.get("krippendorff_alpha"),
            "sim_sentiment":           lf.get("sentiment_polarity"),
            "sim_assertiveness":       lf.get("assertiveness"),
            "sim_emotional_intensity": lf.get("emotional_intensity"),
            "sim_word_count":          lf.get("word_count"),
            "neutral_message": neutral_message,
        }

        _write_row(result, output_path, release_path, write_lock)
        return True

    return False  # ran out of scenarios for this cell


# ── Main ──────────────────────────────────────────────────────────────────────

def generate_message_bank(
    dataset_path: str = _DEFAULT_DATASET,
    output_path: str  = _DEFAULT_BANK,
    biases_file: str  = _DEFAULT_BIASES_FILE,
    n_samples: int    = None,
    n_offset: int     = 0,
    intensity: float  = 0.7,
    seed: int         = 42,
    max_jury_retries: int = 3,
    max_consecutive_fails: int = 2,
    n_workers: int    = 5,
) -> pd.DataFrame:
    """
    Generate and jury-validate human messages using per-cell round-robin processing.

    n_samples controls the target number of rows per (target_bias × human_bias) cell.
    If a cell hits max_consecutive_fails consecutive jury failures in a round, it is
    skipped for that round and retried in the next. No rows are ever deleted.
    """
    biases = _read_biases(biases_file)

    if not os.path.exists(dataset_path):
        print("Dataset not found — downloading from HuggingFace...")
        convert_hf_to_csv(
            output_path=dataset_path,
            bias_filter=",".join(biases),
            n_samples=n_samples,
            n_offset=n_offset,
        )

    dataset        = pd.read_csv(dataset_path)
    dataset        = dataset[dataset["bias"].isin(biases)].reset_index(drop=True)
    bias_direction = _build_bias_direction_map(dataset)
    release_path   = output_path.replace(".csv", "_release.csv")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # ── Per-cell counts and used scenario tracking (for resume) ───────────────
    cell_counts: dict[str, dict[str, int]]   = {b: {hb: 0 for hb in biases} for b in biases}
    cell_used_ids: dict[str, dict[str, set]] = {b: {hb: set() for hb in biases} for b in biases}

    neutral_cache: dict        = {}   # scenario_id → neutral_message (shared across threads)
    neutral_cache_lock         = threading.Lock()

    if os.path.exists(output_path):
        existing = pd.read_csv(output_path)
        # Remove rows for any bias no longer in Biases.txt (e.g. after a bias is dropped)
        bias_set = set(biases)
        before   = len(existing)
        existing = existing[
            existing["bias"].isin(bias_set) & existing["human_bias"].isin(bias_set)
        ].reset_index(drop=True)
        if len(existing) < before:
            removed = before - len(existing)
            existing.to_csv(output_path, index=False)
            if os.path.exists(release_path):
                rel = pd.read_csv(release_path)
                rel = rel[rel["bias"].isin(bias_set) & rel["human_bias"].isin(bias_set)]
                rel.to_csv(release_path, index=False)
            print(f"Cleaned {removed} rows for biases no longer in Biases.txt.")
        for _, row in existing.iterrows():
            b   = row.get("bias", "")
            hb  = row.get("human_bias", "")
            sid = row.get("scenario_id")
            nm  = row.get("neutral_message")
            if b in cell_counts and hb in cell_counts[b]:
                cell_counts[b][hb] += 1
                if sid is not None:
                    cell_used_ids[b][hb].add(int(sid))
            # Pre-populate neutral cache from existing rows
            if sid is not None and nm is not None and not pd.isna(nm):
                neutral_cache.setdefault(int(sid), nm)
        total_existing = len(existing)
        print(f"Resuming — bank has {total_existing} rows across {len(biases)} biases.")
        print(f"  Neutral cache pre-populated with {len(neutral_cache)} scenarios.")
    else:
        total_existing = 0

    # ── Build per-cell scenario lists ─────────────────────────────────────────
    bias_scenarios: dict[str, list] = {}
    for bias in biases:
        rows = dataset[dataset["bias"] == bias].to_dict("records")
        bias_scenarios[bias] = rows

    cell_scenario_lists: dict[tuple, list] = {}
    for bias in biases:
        for hb in biases:
            used = cell_used_ids[bias][hb]
            cell_scenario_lists[(bias, hb)] = [
                r for r in bias_scenarios[bias]
                if r.get("id") not in used
            ]

    n_samples_target = n_samples or max(len(v) for v in bias_scenarios.values())
    total_target     = n_samples_target * len(biases) * len(biases)
    total_rows       = total_existing
    write_lock       = threading.Lock()

    print(f"Biases ({len(biases)}): {biases}")
    print(f"Target: {n_samples_target} rows/cell × {len(biases)}×{len(biases)} cells = {total_target} total rows")
    print(f"Bias directions: {bias_direction}\n")

    start     = datetime.datetime.now()
    round_num = 0

    with tqdm(
        total=total_target,
        initial=total_existing,
        unit="row",
        desc="Generating bank",
        dynamic_ncols=True,
    ) as pbar:
        for round_num in range(1, n_samples_target + 1):

            if all(cell_counts[b][hb] >= n_samples_target for b in biases for hb in biases):
                break

            for target_bias in biases:

                needed_hbs = [
                    hb for hb in biases
                    if cell_counts[target_bias][hb] < n_samples_target
                ]
                if not needed_hbs:
                    continue

                with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                    futures = {
                        executor.submit(
                            _fill_cell_one_round,
                            target_bias,
                            hb,
                            bias_direction.get(hb, "positive"),
                            cell_scenario_lists[(target_bias, hb)],
                            intensity,
                            seed,
                            max_jury_retries,
                            output_path,
                            release_path,
                            write_lock,
                            max_consecutive_fails,
                            neutral_cache,
                            neutral_cache_lock,
                        ): hb
                        for hb in needed_hbs
                    }
                    for future in concurrent.futures.as_completed(futures):
                        hb = futures[future]
                        try:
                            wrote = future.result()
                        except Exception as e:
                            tqdm.write(f"  Error ({target_bias} × {hb}): {e}")
                            wrote = False

                        if wrote:
                            cell_counts[target_bias][hb] += 1
                            total_rows += 1
                            pbar.update(1)
                            pbar.set_postfix(round=round_num, rows=total_rows)
                        else:
                            tqdm.write(
                                f"  WARNING: no row written for "
                                f"({target_bias} × {hb}) in round {round_num}"
                            )

    elapsed = datetime.datetime.now() - start
    print(f"\nDone in {elapsed}.")
    print(f"Total rows: {total_rows}  |  Rounds completed: {round_num}")
    print(f"Internal bank : {output_path}")
    print(f"Release CSV   : {release_path}")

    if os.path.exists(output_path):
        df = pd.read_csv(output_path)
        if not df.empty:
            coverage = df.groupby(["bias", "human_bias"]).size().unstack(fill_value=0)
            print("\nCoverage (rows per cell):")
            print(coverage.to_string())
        return df
    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(
        description="Generate jury-validated human messages (per-cell round-robin)."
    )
    parser.add_argument("--dataset",     type=str, default=_DEFAULT_DATASET)
    parser.add_argument("--output",      type=str, default=_DEFAULT_BANK)
    parser.add_argument("--biases_file", type=str, default=_DEFAULT_BIASES_FILE)
    parser.add_argument("--n_samples",   type=int, default=None,
                        help="Target rows per (target_bias × human_bias) cell.")
    parser.add_argument("--n_offset",    type=int, default=0)
    parser.add_argument("--intensity",   type=float, default=0.7)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--max_jury_retries", type=int, default=3)
    parser.add_argument("--max_consecutive_fails", type=int, default=2,
                        help="Skip a cell for the current round after this many consecutive "
                             "scenario-level jury failures. The cell is retried next round.")
    parser.add_argument("--n_workers",   type=int, default=5,
                        help="Parallel threads for human_bias cells within each target_bias.")
    args = parser.parse_args()

    generate_message_bank(
        dataset_path=args.dataset,
        output_path=args.output,
        biases_file=args.biases_file,
        n_samples=args.n_samples,
        n_offset=args.n_offset,
        intensity=args.intensity,
        seed=args.seed,
        max_jury_retries=args.max_jury_retries,
        max_consecutive_fails=args.max_consecutive_fails,
        n_workers=args.n_workers,
    )


if __name__ == "__main__":
    main()
