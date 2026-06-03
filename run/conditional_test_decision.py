import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.testing import TestCase, Template, DecisionResult
from core.utils import get_model, get_metric
from core.base import DecisionError
from human_simulator import HumanSimulator
from bias_jury import BiasJury
import ast
import json
import time
import random
import pandas as pd
import datetime
import numpy as np
from tqdm import tqdm
import argparse
import concurrent.futures
from functools import partial


_RUN_DIR   = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_RUN_DIR)


def _with_backoff(fn, max_retries=6):
    """Call fn(), retrying on HTTP 429 rate-limit errors with exponential backoff + jitter."""
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                if attempt == max_retries:
                    raise
                wait = min(60, (2 ** attempt) + random.uniform(0, 1))
                time.sleep(wait)
            else:
                raise


def _build_jury_feedback(evaluation: dict, bias_name: str) -> str:
    """
    Derive specific, actionable feedback from a failed jury evaluation.

    Identifies which threshold(s) were not met and what the simulator should
    change on the next attempt, so retries are guided rather than random.
    """
    if not evaluation:
        return ""

    target_rating  = evaluation.get("target_rating", 0.0)
    purity         = evaluation.get("purity", 0.0)
    naturalness    = evaluation.get("naturalness", 0.0)
    blind_accuracy = evaluation.get("blind_accuracy", 0.0)
    all_ratings    = evaluation.get("all_bias_ratings", {})
    ind_blind      = evaluation.get("individual_blind", [])

    issues, fixes = [], []

    if target_rating < 3.0:
        issues.append(
            f"The {bias_name} signal was too weak "
            f"(rated {target_rating:.1f}/5 — need ≥ 3.0)"
        )
        fixes.append(
            f"Make the {bias_name} mechanism much more explicit; "
            "use the example as a direct model"
        )

    if purity < 1.5:
        others = {k: v for k, v in all_ratings.items() if k != bias_name}
        if others:
            top_other       = max(others, key=lambda k: others[k])
            top_other_score = others[top_other]
            issues.append(
                f"Message was contaminated by {top_other} "
                f"(rated {top_other_score:.1f}, nearly as high as "
                f"{bias_name} at {target_rating:.1f})"
            )
            fixes.append(
                f"Remove any language that could suggest {top_other}; "
                f"focus the message exclusively on the {bias_name} mechanism"
            )

    if naturalness < 3.0:
        issues.append(
            f"Message sounded unnatural (naturalness {naturalness:.1f}/5 — need ≥ 3.0)"
        )
        fixes.append(
            "Rephrase as casual, direct speech a colleague would actually say; "
            "avoid formally constructed sentences"
        )

    if blind_accuracy < 2 / 3:
        top1s = [bl[0] for bl in ind_blind if bl]
        if top1s:
            identified_as = max(set(top1s), key=top1s.count)
            issues.append(
                f"Reviewers identified it as '{identified_as}' "
                f"instead of '{bias_name}' (blind accuracy {blind_accuracy:.0%})"
            )
            fixes.append(
                f"The {bias_name} mechanism must be unmistakably distinct from "
                f"{identified_as} — use more specific language tied to the "
                f"{bias_name} definition"
            )

    if not issues:
        return ""

    return (
        "Issues:\n" + "\n".join(f"  - {i}" for i in issues) + "\n"
        "Required corrections:\n" + "\n".join(f"  - {f}" for f in fixes)
    )

DATASET_FILE_PATH = os.path.join(_RUN_DIR, "data", "full_dataset.csv")
CONDITIONAL_RESULTS_DIR = os.path.join(_RUN_DIR, "data", "conditional_decision_results")


def extract_common_and_diverging(
    control: Template, treatment: Template, scenario: str
) -> tuple[str, str, str]:
    """
    Splits control and treatment templates into a shared situation prefix and diverging suffixes.

    Compares situation elements from both templates in order. The longest common prefix of
    identical situations becomes common_text; the remainder becomes each template's diverging text.
    If no situations are shared, common_text falls back to the scenario string.

    Args:
        control: Populated control Template.
        treatment: Populated treatment Template.
        scenario: Scenario string from the dataset row (fallback if no common situations).

    Returns:
        (common_text, ctrl_diverging_sits, treat_diverging_sits)
        where *_diverging_sits contains only the situation text unique to each template.
    """

    def get_situation_texts(template: Template) -> list[str]:
        return [
            template._apply_insertions(elem.text)
            for elem in template._data.findall("situation")
        ]

    ctrl_sits = get_situation_texts(control)
    treat_sits = get_situation_texts(treatment)

    common_count = 0
    for cs, ts in zip(ctrl_sits, treat_sits):
        if cs.strip() == ts.strip():
            common_count += 1
        else:
            break

    common_text = "\n".join(ctrl_sits[:common_count]) if common_count > 0 else scenario
    ctrl_diverging = "\n".join(ctrl_sits[common_count:])
    treat_diverging = "\n".join(treat_sits[common_count:])

    return common_text, ctrl_diverging, treat_diverging


def _run_conditional_decision(
    row: pd.Series,
    model,
    simulator: HumanSimulator,
    jury: BiasJury,
    human_bias: str,
    direction: str,
    intensity: float,
    temperature: float,
    seed: int,
    max_jury_retries: int,
) -> dict:
    """
    Run one conditional decision for a single test case row.

    Both the baseline (control) and conditional (treatment) runs use the treatment
    template, isolating the effect of the human message from the scenario framing.

    Conversation structure:

      Control (baseline — no human message):
        Single turn: full treatment scenario + prompt + options → LLM decision

      Treatment (conditional — with human message):
        Turn 1 (user):      common situation  (shared prefix of treatment template)
        Turn 2 (user):      human simulator's biased message
        Turn 3 (assistant): LLM's intermediate response
        Turn 4 (user):      treatment-diverging situation + prompt + answer options
        Turn 5 (assistant): LLM's final decision (scored)

    Score = shift in decision caused by the human message alone.

    Args:
        row: A row from the full dataset CSV.
        model: An LLM instance (the model being evaluated).
        simulator: A HumanSimulator instance.
        jury: A BiasJury instance.
        human_bias: The bias the human simulator should express.
        direction: Directional orientation of the human bias ("positive" or "negative").
                   Should be derived from the human bias's own k value, not the target bias's.
        intensity: Bias intensity in [0, 1].
        temperature: Temperature for the evaluated LLM.
        seed: Base seed for reproducibility.
        max_jury_retries: Maximum attempts to generate a jury-approved simulator message.

    Returns:
        A dict containing all fields for one row of the output CSV.
    """
    control_tmpl = Template(row["raw_control"])
    treatment_tmpl = Template(row["raw_treatment"])
    scenario = row.get("scenario", "")

    # Derive the common context (shown before the human message) and the
    # treatment-specific diverging text (shown in the decision turn).
    common_text, _, treat_diverging = extract_common_and_diverging(
        control_tmpl, treatment_tmpl, scenario
    )

    # ── Generate biased human message + jury validation ───────────────────────
    # "neutral" is a special condition: skip the jury and generate an unbiased message.
    if human_bias.lower() == "neutral":
        sim_message, _, _ = simulator.generate_neutral(
            situation=common_text, seed=seed, jury=jury
        )
        jury_result, jury_passed = None, True
    else:
        sim_message, jury_result, jury_passed = None, None, False
        last_candidate, last_eval = None, None
        feedback = ""
        for attempt in range(max_jury_retries):
            candidate = simulator.generate(
                situation=common_text,
                bias_name=human_bias,
                direction=direction,
                intensity=intensity,
                seed=seed + attempt,
                feedback=feedback,
            )
            evaluation = jury.evaluate(candidate, common_text, human_bias, seed=seed + attempt)
            last_candidate, last_eval = candidate, evaluation
            if jury.is_valid(evaluation):
                sim_message, jury_result, jury_passed = candidate, evaluation, True
                break
            # Build specific feedback so the next attempt corrects the actual failure
            feedback = _build_jury_feedback(evaluation, human_bias)
        if sim_message is None:
            sim_message, jury_result, jury_passed = last_candidate, last_eval, False

    # ── Extract jury scalar fields for CSV storage ────────────────────────────
    jr = jury_result or {}
    lf = jr.get("linguistic_features", {})
    jury_purity             = jr.get("purity")
    jury_target_rating      = jr.get("target_rating")
    jury_naturalness        = jr.get("naturalness")
    jury_blind_accuracy     = jr.get("blind_accuracy")
    jury_krippendorff_alpha = jr.get("krippendorff_alpha")
    sim_sentiment           = lf.get("sentiment_polarity")
    sim_assertiveness       = lf.get("assertiveness")
    sim_emotional_intensity = lf.get("emotional_intensity")
    sim_word_count          = lf.get("word_count")

    # ── Control: treatment scenario WITHOUT any human message ─────────────────
    (
        ctrl_answer, ctrl_extraction, ctrl_decision, ctrl_opts, ctrl_opt_order
    ) = model._decide(treatment_tmpl, temperature=temperature, seed=seed)

    # ── Treatment: treatment scenario WITH biased human message ───────────────
    (
        treat_intermediate, treat_answer, treat_extraction,
        treat_decision, treat_opts, treat_opt_order
    ) = model._decide_multiturn(
        common_situation=common_text,
        human_message=sim_message,
        diverging_situations=treat_diverging,
        template=treatment_tmpl,
        temperature=temperature,
        seed=seed,
    )

    # ── Neutral baseline: treatment scenario WITH an unbiased human message ───
    # Isolates the bias-specific effect from general linguistic properties
    # (length, sentiment, assertiveness) that any human message carries.
    # seed offset (+100) keeps this message consistent across human_bias iterations
    # for the same row, so comparisons are fair.
    neutral_message, neutral_jury_result, neutral_jury_passed = simulator.generate_neutral(
        situation=common_text, seed=seed + 100, jury=jury
    )
    (
        neutral_intermediate, neutral_answer, neutral_extraction,
        neutral_decision, neutral_opts, neutral_opt_order
    ) = model._decide_multiturn(
        common_situation=common_text,
        human_message=neutral_message,
        diverging_situations=treat_diverging,
        template=treatment_tmpl,
        temperature=temperature,
        seed=seed,
    )

    return {
        "id": row.get("id"),
        "bias": row.get("bias"),
        "model": model.NAME,
        "human_bias": human_bias,
        "direction": direction,
        "intensity": intensity,
        # ── Biased human message ──────────────────────────────────────────────
        "human_message": sim_message,
        "jury_result": json.dumps(jury_result, default=str) if jury_result else None,
        "jury_passed": jury_passed,
        # ── Jury scientific metrics ───────────────────────────────────────────
        "jury_purity":             jury_purity,
        "jury_target_rating":      jury_target_rating,
        "jury_naturalness":        jury_naturalness,
        "jury_blind_accuracy":     jury_blind_accuracy,
        "jury_krippendorff_alpha": jury_krippendorff_alpha,
        # ── Linguistic features of the biased message ─────────────────────────
        "sim_sentiment":           sim_sentiment,
        "sim_assertiveness":       sim_assertiveness,
        "sim_emotional_intensity": sim_emotional_intensity,
        "sim_word_count":          sim_word_count,
        # ── Conversation context ──────────────────────────────────────────────
        "common_situation": common_text,
        "diverging_situation": treat_diverging,
        # ── Control (no human message) ────────────────────────────────────────
        "control_intermediate": None,
        "control_answer": ctrl_answer,
        "control_extraction": ctrl_extraction,
        "control_decision": ctrl_decision,
        "control_options": ctrl_opts,
        "control_option_order": ctrl_opt_order,
        # ── Treatment (biased human message) ─────────────────────────────────
        "treatment_intermediate": treat_intermediate,
        "treatment_answer": treat_answer,
        "treatment_extraction": treat_extraction,
        "treatment_decision": treat_decision,
        "treatment_options": treat_opts,
        "treatment_option_order": treat_opt_order,
        # ── Neutral baseline (unbiased human message) ─────────────────────────
        "neutral_message": neutral_message,
        "neutral_jury_result": json.dumps(neutral_jury_result, default=str) if neutral_jury_result else None,
        "neutral_jury_passed": neutral_jury_passed,
        "neutral_intermediate": neutral_intermediate,
        "neutral_answer": neutral_answer,
        "neutral_extraction": neutral_extraction,
        "neutral_decision": neutral_decision,
        "neutral_options": neutral_opts,
        "neutral_option_order": neutral_opt_order,
        "status": "OK",
        "error_message": None,
    }


def _ratio_score_from_params(ctrl_ans: float, treat_ans: float, mp: dict,
                              n_treat_opts: int) -> float:
    """
    Compute RatioScaleMetric score directly from metric_params dict.

    Used as fallback when the bias-specific metric class cannot be instantiated
    (e.g. because template insertions are absent in HuggingFace-sourced data).
    Also used to score the neutral baseline condition (same formula, different treat_ans).
    """
    k = float(mp.get("k", -1))
    x_1 = float(mp.get("x_1", 0))
    x_2 = float(mp.get("x_2", 0))
    if mp.get("flip_treatment", False):
        treat_ans = float(n_treat_opts - 1) - treat_ans
    delta_ctrl = abs(ctrl_ans - x_1)
    delta_treat = abs(treat_ans - x_2)
    denom = max(delta_ctrl, delta_treat)
    return k * (delta_ctrl - delta_treat) / denom if denom > 0 else 0.0


def _compute_individual_scores(rows: list[dict], batch: pd.DataFrame) -> list[dict]:
    """
    Computes per-row bias scores for OK rows across all three human-message conditions.

    Each score pairs the original control-template response (unbiased framing) against
    the original treatment-template response (biased framing) within the same
    human-message condition:
      score_no_human  — metric(no_human_ctrl_decision, no_human_treat_decision)
      score_neutral   — metric(neutral_ctrl_decision,  neutral_treat_decision)
      score_biased    — metric(biased_ctrl_decision,   biased_treat_decision)
      delta_score     — |score_biased| − |score_neutral|   (content effect: bias magnitude change)
      effect_neutral  — |score_neutral| − |score_no_human| (presence effect: bias magnitude change)
      effect_biased   — |score_biased| − |score_no_human|  (total effect: bias magnitude change)

    Falls back to metric_params-based computation for biases whose metric class
    reads template insertions that are absent in HuggingFace-sourced templates.
    """
    id_to_batch_row = {row.get("id"): row for _, row in batch.iterrows()}

    bias_groups: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        if r.get("status") == "OK":
            bias_groups.setdefault(r.get("bias", ""), []).append(i)

    for bias_raw, indices in bias_groups.items():
        bias_fmt = "".join(
            " " + c if c.isupper() else c for c in bias_raw
        ).strip().title().replace(" ", "")

        test_cases_m = []
        drs_no_human, drs_neutral, drs_biased = [], [], []
        valid_indices = []
        metric_params_list = []

        for i in indices:
            r = rows[i]
            batch_row = id_to_batch_row.get(r.get("id"))
            if batch_row is None:
                continue
            try:
                tc = TestCase(
                    bias=batch_row["bias"],
                    control=Template(batch_row["raw_control"]),
                    treatment=Template(batch_row["raw_treatment"]),
                    generator=str(batch_row.get("generator", "")),
                    temperature=float(batch_row.get("temperature", 0.0)),
                    seed=int(batch_row.get("seed", 42)),
                    scenario=str(batch_row.get("scenario", "")),
                )

                def _parse_list(v):
                    if isinstance(v, str):
                        return ast.literal_eval(v)
                    return v

                def _make_dr(ctrl_prefix, treat_prefix):
                    return DecisionResult(
                        model=r["model"],
                        control_options=_parse_list(r[f"{ctrl_prefix}_options"]),
                        control_option_order=_parse_list(r[f"{ctrl_prefix}_option_order"]),
                        control_answer=str(r[f"{ctrl_prefix}_answer"]),
                        control_extraction=str(r[f"{ctrl_prefix}_extraction"]),
                        control_decision=int(r[f"{ctrl_prefix}_decision"]),
                        treatment_options=_parse_list(r[f"{treat_prefix}_options"]),
                        treatment_option_order=_parse_list(r[f"{treat_prefix}_option_order"]),
                        treatment_answer=str(r[f"{treat_prefix}_answer"]),
                        treatment_extraction=str(r[f"{treat_prefix}_extraction"]),
                        treatment_decision=int(r[f"{treat_prefix}_decision"]),
                    )

                dr_no_human = _make_dr("no_human_ctrl", "no_human_treat")
                dr_neutral  = _make_dr("neutral_ctrl",  "neutral_treat")
                dr_biased   = _make_dr("biased_ctrl",   "biased_treat")

                mp_raw = batch_row.get("metric_params", "{}")
                mp = ast.literal_eval(mp_raw) if isinstance(mp_raw, str) else dict(mp_raw)

                test_cases_m.append(tc)
                drs_no_human.append(dr_no_human)
                drs_neutral.append(dr_neutral)
                drs_biased.append(dr_biased)
                valid_indices.append(i)
                metric_params_list.append(mp)
            except Exception:
                continue

        if not test_cases_m:
            continue

        try:
            def _batch_scores(drs):
                m = get_metric(bias_fmt)(test_results=list(zip(test_cases_m, drs)))
                return m.compute().flatten(), m.test_weights.flatten()

            scores_no_human, weights = _batch_scores(drs_no_human)
            scores_neutral,  _       = _batch_scores(drs_neutral)
            scores_biased,   _       = _batch_scores(drs_biased)

            for j, idx in enumerate(valid_indices):
                s_nh = float(scores_no_human[j])
                s_n  = float(scores_neutral[j])
                s_b  = float(scores_biased[j])
                rows[idx]["score_no_human"]  = s_nh
                rows[idx]["score_neutral"]   = s_n
                rows[idx]["score_biased"]    = s_b
                rows[idx]["delta_score"]     = abs(s_b) - abs(s_n)    # content effect: |m_b| - |m_n|
                rows[idx]["effect_neutral"]  = abs(s_n) - abs(s_nh)   # presence effect: |m_n| - |m_∅|
                rows[idx]["effect_biased"]   = abs(s_b) - abs(s_nh)   # total effect: |m_b| - |m_∅|
                rows[idx]["weight"]          = float(weights[j])
        except Exception:
            # Fallback: metric_params-based computation for biases whose metric
            # inits read template insertions absent in HuggingFace-sourced templates.
            for j, idx in enumerate(valid_indices):
                try:
                    mp = metric_params_list[j]

                    def _ratio(dr):
                        return _ratio_score_from_params(
                            float(dr.CONTROL_DECISION),
                            float(dr.TREATMENT_DECISION),
                            mp,
                            len(dr.TREATMENT_OPTIONS),
                        )

                    s_no_human = _ratio(drs_no_human[j])
                    s_neutral  = _ratio(drs_neutral[j])
                    s_biased   = _ratio(drs_biased[j])

                    rows[idx]["score_no_human"]  = float(s_no_human)
                    rows[idx]["score_neutral"]   = float(s_neutral)
                    rows[idx]["score_biased"]    = float(s_biased)
                    rows[idx]["delta_score"]     = abs(float(s_biased)) - abs(float(s_neutral))
                    rows[idx]["effect_neutral"]  = abs(float(s_neutral)) - abs(float(s_no_human))
                    rows[idx]["effect_biased"]   = abs(float(s_biased))  - abs(float(s_no_human))
                    rows[idx]["weight"]          = 1.0
                except Exception:
                    pass

    return rows


def decide_conditional_batch(
    batch: pd.DataFrame,
    model_name: str,
    human_bias: str,
    direction: str,
    intensity: float,
    randomly_flip_options: bool,
    shuffle_answer_options: bool,
    temperature: float,
    seed: int,
    max_jury_retries: int,
):
    """
    Process a batch of test cases with multi-turn conditional decisions and save results.

    Each worker process calls this function independently; results are written to a
    per-process CSV file that is later merged by the caller.
    """
    # Model files open paths like "./models/OpenAI/prompts.yml" relative to the repo root.
    # Workers may inherit a different CWD, so pin to the repo root for model loading.
    os.chdir(_REPO_ROOT)

    model = get_model(
        model_name,
        randomly_flip_options=randomly_flip_options,
        shuffle_answer_options=shuffle_answer_options,
    )
    simulator = HumanSimulator()
    jury = BiasJury()
    rows = []

    for _, row in batch.iterrows():
        try:
            result = _run_conditional_decision(
                row=row,
                model=model,
                simulator=simulator,
                jury=jury,
                human_bias=human_bias,
                direction=direction,
                intensity=intensity,
                temperature=temperature,
                seed=seed,
                max_jury_retries=max_jury_retries,
            )
        except Exception as e:
            result = {
                "id": row.get("id"),
                "bias": row.get("bias"),
                "model": model_name,
                "human_bias": human_bias,
                "direction": direction,
                "intensity": intensity,
                "status": "ERROR",
                "error_message": str(e),
                **{k: None for k in [
                    "human_message", "jury_result", "jury_passed",
                    "jury_purity", "jury_target_rating", "jury_naturalness",
                    "jury_blind_accuracy", "jury_krippendorff_alpha",
                    "sim_sentiment", "sim_assertiveness", "sim_emotional_intensity",
                    "sim_word_count", "common_situation", "diverging_situation",
                    "control_intermediate", "control_answer", "control_extraction",
                    "control_decision", "control_options", "control_option_order",
                    "treatment_intermediate", "treatment_answer", "treatment_extraction",
                    "treatment_decision", "treatment_options", "treatment_option_order",
                    "neutral_message", "neutral_jury_result", "neutral_jury_passed",
                    "neutral_intermediate", "neutral_answer",
                    "neutral_extraction", "neutral_decision", "neutral_options",
                    "neutral_option_order",
                ]},
            }
        rows.append(result)

    rows = _compute_individual_scores(rows, batch)

    out_dir = os.path.join(CONDITIONAL_RESULTS_DIR, model_name)
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, f"batch_{os.getpid()}_{ts}.csv"), index=False
    )


def _evaluate_row(row, model, no_human_cache, neutral_cache, seed, temperature):
    """
    Evaluate one bank row with 6 LLM calls: original ctrl/treat templates × no-human/neutral/biased.

    no_human_cache and neutral_cache are keyed by ("ctrl"|"treat", scenario_id, seed).
    The biased condition is not cached because it varies per human_bias.
    Thread-safe — no shared mutable state.
    """
    row_seed = int(row.get("seed", seed))
    try:
        control_tmpl    = Template(row["raw_control"])
        treatment_tmpl  = Template(row["raw_treatment"])
        common_text     = str(row["common_situation"])
        treat_diverging = str(row["diverging_situation"])
        sim_message     = str(row["human_message"])
        neutral_message = str(row["neutral_message"])

        scenario    = str(row.get("scenario", ""))
        _, ctrl_diverging, _ = extract_common_and_diverging(
            control_tmpl, treatment_tmpl, scenario
        )

        scenario_id = row.get("scenario_id")
        cache_key   = (scenario_id, row_seed)

        # ── No human message (cached per scenario) ────────────────────────────
        nh_ctrl = no_human_cache.get(("ctrl",) + cache_key)
        nh_treat = no_human_cache.get(("treat",) + cache_key)
        if isinstance(nh_ctrl, Exception):
            raise nh_ctrl
        if isinstance(nh_treat, Exception):
            raise nh_treat
        (no_human_ctrl_answer, no_human_ctrl_extraction, no_human_ctrl_decision,
         no_human_ctrl_opts, no_human_ctrl_opt_order) = nh_ctrl
        (no_human_treat_answer, no_human_treat_extraction, no_human_treat_decision,
         no_human_treat_opts, no_human_treat_opt_order) = nh_treat

        # ── Neutral human message (cached per scenario) ───────────────────────
        n_ctrl = neutral_cache.get(("ctrl",) + cache_key)
        n_treat = neutral_cache.get(("treat",) + cache_key)
        if isinstance(n_ctrl, Exception):
            raise n_ctrl
        if isinstance(n_treat, Exception):
            raise n_treat
        (neutral_ctrl_intermediate, neutral_ctrl_answer, neutral_ctrl_extraction,
         neutral_ctrl_decision, neutral_ctrl_opts, neutral_ctrl_opt_order) = n_ctrl
        (neutral_treat_intermediate, neutral_treat_answer, neutral_treat_extraction,
         neutral_treat_decision, neutral_treat_opts, neutral_treat_opt_order) = n_treat

        # ── Biased human message (per row — varies with human_bias) ───────────
        (biased_ctrl_intermediate, biased_ctrl_answer, biased_ctrl_extraction,
         biased_ctrl_decision, biased_ctrl_opts, biased_ctrl_opt_order) = _with_backoff(
            lambda: model._decide_multiturn(
                common_situation=common_text,
                human_message=sim_message,
                diverging_situations=ctrl_diverging,
                template=control_tmpl,
                temperature=temperature,
                seed=row_seed,
            )
        )
        (biased_treat_intermediate, biased_treat_answer, biased_treat_extraction,
         biased_treat_decision, biased_treat_opts, biased_treat_opt_order) = _with_backoff(
            lambda: model._decide_multiturn(
                common_situation=common_text,
                human_message=sim_message,
                diverging_situations=treat_diverging,
                template=treatment_tmpl,
                temperature=temperature,
                seed=row_seed,
            )
        )

        return {
            "id":         row.get("id"),
            "scenario_id": scenario_id,
            "bias":        row.get("bias"),
            "model":       model.NAME,
            "human_bias":  row.get("human_bias"),
            "direction":   row.get("direction"),
            "intensity":   row.get("intensity"),
            "human_message":           sim_message,
            "jury_result":             None,
            "jury_passed":             row.get("jury_passed", True),
            "jury_purity":             row.get("jury_purity"),
            "jury_target_rating":      row.get("jury_target_rating"),
            "jury_naturalness":        row.get("jury_naturalness"),
            "jury_blind_accuracy":     row.get("jury_blind_accuracy"),
            "jury_krippendorff_alpha": row.get("jury_krippendorff_alpha"),
            "sim_sentiment":           row.get("sim_sentiment"),
            "sim_assertiveness":       row.get("sim_assertiveness"),
            "sim_emotional_intensity": row.get("sim_emotional_intensity"),
            "sim_word_count":          row.get("sim_word_count"),
            "common_situation":        common_text,
            "ctrl_diverging_situation":  ctrl_diverging,
            "treat_diverging_situation": treat_diverging,
            "neutral_message":         neutral_message,
            "neutral_jury_result":     None,
            "neutral_jury_passed":     True,
            # No human message — original ctrl and treat templates
            "no_human_ctrl_answer":       no_human_ctrl_answer,
            "no_human_ctrl_extraction":   no_human_ctrl_extraction,
            "no_human_ctrl_decision":     no_human_ctrl_decision,
            "no_human_ctrl_options":      no_human_ctrl_opts,
            "no_human_ctrl_option_order": no_human_ctrl_opt_order,
            "no_human_treat_answer":       no_human_treat_answer,
            "no_human_treat_extraction":   no_human_treat_extraction,
            "no_human_treat_decision":     no_human_treat_decision,
            "no_human_treat_options":      no_human_treat_opts,
            "no_human_treat_option_order": no_human_treat_opt_order,
            # Neutral human message — original ctrl and treat templates
            "neutral_ctrl_intermediate":   neutral_ctrl_intermediate,
            "neutral_ctrl_answer":         neutral_ctrl_answer,
            "neutral_ctrl_extraction":     neutral_ctrl_extraction,
            "neutral_ctrl_decision":       neutral_ctrl_decision,
            "neutral_ctrl_options":        neutral_ctrl_opts,
            "neutral_ctrl_option_order":   neutral_ctrl_opt_order,
            "neutral_treat_intermediate":  neutral_treat_intermediate,
            "neutral_treat_answer":        neutral_treat_answer,
            "neutral_treat_extraction":    neutral_treat_extraction,
            "neutral_treat_decision":      neutral_treat_decision,
            "neutral_treat_options":       neutral_treat_opts,
            "neutral_treat_option_order":  neutral_treat_opt_order,
            # Biased human message — original ctrl and treat templates
            "biased_ctrl_intermediate":   biased_ctrl_intermediate,
            "biased_ctrl_answer":         biased_ctrl_answer,
            "biased_ctrl_extraction":     biased_ctrl_extraction,
            "biased_ctrl_decision":       biased_ctrl_decision,
            "biased_ctrl_options":        biased_ctrl_opts,
            "biased_ctrl_option_order":   biased_ctrl_opt_order,
            "biased_treat_intermediate":  biased_treat_intermediate,
            "biased_treat_answer":        biased_treat_answer,
            "biased_treat_extraction":    biased_treat_extraction,
            "biased_treat_decision":      biased_treat_decision,
            "biased_treat_options":       biased_treat_opts,
            "biased_treat_option_order":  biased_treat_opt_order,
            "status":        "OK",
            "error_message": None,
        }
    except Exception as e:
        return {
            "id":         row.get("id"),
            "scenario_id": row.get("scenario_id"),
            "bias":        row.get("bias"),
            "model":       model.NAME,
            "human_bias":  row.get("human_bias"),
            "direction":   row.get("direction"),
            "intensity":   row.get("intensity"),
            "status":        "ERROR",
            "error_message": str(e),
            **{k: None for k in [
                "human_message", "jury_result", "jury_passed",
                "jury_purity", "jury_target_rating", "jury_naturalness",
                "jury_blind_accuracy", "jury_krippendorff_alpha",
                "sim_sentiment", "sim_assertiveness", "sim_emotional_intensity",
                "sim_word_count", "common_situation",
                "ctrl_diverging_situation", "treat_diverging_situation",
                "neutral_message", "neutral_jury_result", "neutral_jury_passed",
                "no_human_ctrl_answer", "no_human_ctrl_extraction", "no_human_ctrl_decision",
                "no_human_ctrl_options", "no_human_ctrl_option_order",
                "no_human_treat_answer", "no_human_treat_extraction", "no_human_treat_decision",
                "no_human_treat_options", "no_human_treat_option_order",
                "neutral_ctrl_intermediate", "neutral_ctrl_answer", "neutral_ctrl_extraction",
                "neutral_ctrl_decision", "neutral_ctrl_options", "neutral_ctrl_option_order",
                "neutral_treat_intermediate", "neutral_treat_answer", "neutral_treat_extraction",
                "neutral_treat_decision", "neutral_treat_options", "neutral_treat_option_order",
                "biased_ctrl_intermediate", "biased_ctrl_answer", "biased_ctrl_extraction",
                "biased_ctrl_decision", "biased_ctrl_options", "biased_ctrl_option_order",
                "biased_treat_intermediate", "biased_treat_answer", "biased_treat_extraction",
                "biased_treat_decision", "biased_treat_options", "biased_treat_option_order",
            ]},
        }


def decide_from_bank_batch(
    bank_batch: pd.DataFrame,
    model_name: str,
    randomly_flip_options: bool,
    shuffle_answer_options: bool,
    temperature: float,
    seed: int,
    row_workers: int = 20,
):
    """
    Run LLM evaluation on a batch of pre-validated message bank rows.

    Unlike decide_conditional_batch, this function skips jury generation entirely —
    human_message and neutral_message are already validated and stored in bank_batch.
    This makes it safe to reuse the same bank across different LLMs without
    incurring jury API costs again.

    Results are written to the same per-batch CSV format used by the standard
    pipeline so that downstream merging and analysis code works unchanged.

    Args:
        bank_batch:            Slice of validated_message_bank.csv (one row per
                               jury-approved (scenario, human_bias) pair).
        model_name:            LLM to evaluate.
        randomly_flip_options: Whether to reverse answer options in 50% of cases.
        shuffle_answer_options: Whether to randomly shuffle answer options.
        temperature:           LLM sampling temperature.
        seed:                  Fallback seed if row-level seed is missing.
    """
    os.chdir(_REPO_ROOT)
    model = get_model(
        model_name,
        randomly_flip_options=randomly_flip_options,
        shuffle_answer_options=shuffle_answer_options,
    )

    # No-human and neutral calls depend only on (scenario, seed) — not on human_bias.
    # Pre-compute once per unique (scenario_id, seed) for both ctrl and treat templates,
    # then reuse across all human_bias variants to avoid redundant API calls.
    # Keys: ("ctrl"|"treat", scenario_id, seed)
    no_human_cache: dict = {}
    neutral_cache: dict = {}
    for _, row in bank_batch.iterrows():
        s_id      = row.get("scenario_id")
        s_seed    = int(row.get("seed", seed))
        ctrl_tmpl = Template(row["raw_control"])
        treat_tmpl = Template(row["raw_treatment"])
        common_text     = str(row["common_situation"])
        treat_diverging = str(row["diverging_situation"])
        scenario        = str(row.get("scenario", ""))
        _, ctrl_diverging, _ = extract_common_and_diverging(ctrl_tmpl, treat_tmpl, scenario)
        neutral_msg = str(row["neutral_message"])

        for prefix, tmpl, diverg in [("ctrl", ctrl_tmpl, ctrl_diverging),
                                      ("treat", treat_tmpl, treat_diverging)]:
            nh_key = (prefix, s_id, s_seed)
            if nh_key not in no_human_cache:
                try:
                    no_human_cache[nh_key] = model._decide(
                        tmpl, temperature=temperature, seed=s_seed
                    )
                except Exception as e:
                    no_human_cache[nh_key] = e
            n_key = (prefix, s_id, s_seed)
            if n_key not in neutral_cache:
                try:
                    neutral_cache[n_key] = model._decide_multiturn(
                        common_situation=common_text,
                        human_message=neutral_msg,
                        diverging_situations=diverg,
                        template=tmpl,
                        temperature=temperature,
                        seed=s_seed,
                    )
                except Exception as e:
                    neutral_cache[n_key] = e

    with concurrent.futures.ThreadPoolExecutor(max_workers=row_workers) as thread_pool:
        future_to_idx = {
            thread_pool.submit(
                _evaluate_row, row, model, no_human_cache, neutral_cache, seed, temperature
            ): idx
            for idx, (_, row) in enumerate(bank_batch.iterrows())
        }
        rows = [None] * len(future_to_idx)
        for future in concurrent.futures.as_completed(future_to_idx):
            rows[future_to_idx[future]] = future.result()

    # _compute_individual_scores looks up raw_control/raw_treatment/metric_params
    # by row id. Bank rows use scenario_id as their stable HF identifier; create
    # a temporary "id" column so the lookup works without modifying the scorer.
    # Duplicate ids (one per human_bias) are safe — the looked-up fields are
    # identical across all rows sharing the same scenario_id.
    lookup_df = bank_batch.copy()
    if "scenario_id" in lookup_df.columns and "id" not in lookup_df.columns:
        lookup_df["id"] = lookup_df["scenario_id"]
    # Propagate scenario_id into result rows so the output CSV carries it
    sid_map = dict(zip(bank_batch.index, bank_batch.get("scenario_id", bank_batch.get("id"))))
    for i, r in enumerate(rows):
        if "scenario_id" not in r:
            r["scenario_id"] = sid_map.get(i)
    rows = _compute_individual_scores(rows, lookup_df)

    out_dir = os.path.join(CONDITIONAL_RESULTS_DIR, model_name)
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, f"batch_{os.getpid()}_{ts}.csv"), index=False
    )


def evaluate_bank(
    bank_df: pd.DataFrame,
    model_name: str,
    randomly_flip_options: bool,
    shuffle_answer_options: bool,
    temperature: float,
    seed: int,
    total_workers: int = 100,
    on_row_done=None,
):
    """
    Evaluate all rows in bank_df with a flat ThreadPoolExecutor.

    Replaces the ProcessPoolExecutor + batches approach so the caller can track
    progress at row granularity via the on_row_done callback.

    Flow:
      1. Pre-compute control + neutral caches concurrently (one call per unique scenario).
      2. Process every row concurrently (treatment call only — control + neutral are cached).
      3. Compute scores and write a single batch CSV.
    """
    os.chdir(_REPO_ROOT)

    # Normalise so both "id" and "scenario_id" always exist, pointing to the same key.
    # Bank CSV uses "scenario_id"; retry path may have already renamed it to "id".
    bank_df = bank_df.copy()
    if "scenario_id" in bank_df.columns and "id" not in bank_df.columns:
        bank_df["id"] = bank_df["scenario_id"]
    elif "id" in bank_df.columns and "scenario_id" not in bank_df.columns:
        bank_df["scenario_id"] = bank_df["id"]
    elif "id" not in bank_df.columns:
        bank_df["id"] = range(len(bank_df))
        bank_df["scenario_id"] = bank_df["id"]

    model = get_model(
        model_name,
        randomly_flip_options=randomly_flip_options,
        shuffle_answer_options=shuffle_answer_options,
    )

    # ── Step 1: build no-human + neutral caches concurrently ─────────────────
    # Keys: ("ctrl"|"treat", scenario_id, seed)
    # Both ctrl and treat templates are pre-prompted for each human condition
    # so they can be reused across all human_bias rows sharing the same scenario.
    unique_rows = {}
    for _, row in bank_df.iterrows():
        key = (row.get("scenario_id"), int(row.get("seed", seed)))
        if key not in unique_rows:
            unique_rows[key] = row

    def _build_cache_entry(item):
        key, row = item
        ctrl_tmpl  = Template(row["raw_control"])
        treat_tmpl = Template(row["raw_treatment"])
        common_text     = str(row["common_situation"])
        treat_diverging = str(row["diverging_situation"])
        scenario        = str(row.get("scenario", ""))
        _, ctrl_diverging, _ = extract_common_and_diverging(ctrl_tmpl, treat_tmpl, scenario)
        neutral_msg = str(row["neutral_message"])

        results = {}
        for prefix, tmpl, diverg in [("ctrl", ctrl_tmpl, ctrl_diverging),
                                      ("treat", treat_tmpl, treat_diverging)]:
            nh_key = (prefix,) + key
            try:
                results[("nh", nh_key)] = _with_backoff(
                    lambda t=tmpl: model._decide(t, temperature=temperature, seed=key[1])
                )
            except Exception as e:
                results[("nh", nh_key)] = e
            try:
                results[("n", nh_key)] = _with_backoff(
                    lambda t=tmpl, d=diverg: model._decide_multiturn(
                        common_situation=common_text,
                        human_message=neutral_msg,
                        diverging_situations=d,
                        template=t,
                        temperature=temperature,
                        seed=key[1],
                    )
                )
            except Exception as e:
                results[("n", nh_key)] = e
        return results

    no_human_cache: dict = {}
    neutral_cache: dict = {}
    n_unique = len(unique_rows)
    print(f"\n  Step 1/3 — Building zero-shot + neutral cache "
          f"({n_unique} unique scenarios × 4 calls = {n_unique * 4:,} API calls) ...")
    cache_start = datetime.datetime.now()
    with concurrent.futures.ThreadPoolExecutor(max_workers=total_workers) as pool:
        with tqdm(total=n_unique, unit="scenario", desc="  Cache", dynamic_ncols=True,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            for results in pool.map(_build_cache_entry, unique_rows.items()):
                for (kind, cache_key), val in results.items():
                    if kind == "nh":
                        no_human_cache[cache_key] = val
                    else:
                        neutral_cache[cache_key] = val
                pbar.update(1)
    cache_elapsed = datetime.datetime.now() - cache_start
    print(f"  Cache built in {cache_elapsed}. "
          f"({no_human_cache.__len__()} zero-shot + {neutral_cache.__len__()} neutral entries)")

    # ── Step 2: process all rows concurrently ────────────────────────────────
    n_rows = len(bank_df)
    print(f"\n  Step 2/3 — Biased-turn evaluation "
          f"({n_rows:,} rows × 2 calls = {n_rows * 2:,} API calls) ...")
    rows = [None] * n_rows
    with concurrent.futures.ThreadPoolExecutor(max_workers=total_workers) as pool:
        future_to_idx = {
            pool.submit(
                _evaluate_row, row, model, no_human_cache, neutral_cache, seed, temperature
            ): idx
            for idx, (_, row) in enumerate(bank_df.iterrows())
        }
        with tqdm(total=n_rows, unit="row", desc="  Eval",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                  dynamic_ncols=True) as biased_pbar:
            for future in concurrent.futures.as_completed(future_to_idx):
                rows[future_to_idx[future]] = future.result()
                biased_pbar.update(1)
                if on_row_done:
                    on_row_done()

    # ── Step 3: score + write ────────────────────────────────────────────────
    print(f"\n  Step 3/3 — Computing bias scores and writing batch CSV ...")
    lookup_df = bank_df  # already has "id" column from normalisation above
    sid_col = "scenario_id" if "scenario_id" in bank_df.columns else "id"
    sid_map = dict(zip(bank_df.index, bank_df[sid_col]))
    for i, r in enumerate(rows):
        if r is not None and "scenario_id" not in r:
            r["scenario_id"] = sid_map.get(i)
    rows = _compute_individual_scores(rows, lookup_df)

    out_dir = os.path.join(CONDITIONAL_RESULTS_DIR, model_name)
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, f"batch_{os.getpid()}_{ts}.csv"), index=False
    )


def decide_conditional_dataset(
    dataset: pd.DataFrame,
    model_name: str,
    human_bias: str,
    direction: str,
    intensity: float,
    n_batches: int,
    n_workers: int,
    randomly_flip_options: bool,
    shuffle_answer_options: bool,
    temperature: float,
    seed: int,
    max_jury_retries: int,
):
    """
    Orchestrate parallel conditional decision-making over the full dataset.

    Splits the dataset into n_batches equal chunks, distributes them across
    n_workers processes, then merges all per-batch CSVs into a single result file.

    Args:
        dataset: The full dataset DataFrame.
        model_name: Name of the LLM to evaluate.
        human_bias: Bias for the human simulator to express.
        direction: Directional orientation of the human bias.
        intensity: Bias intensity in [0, 1].
        n_batches: Number of equal-sized batches to split the dataset into.
        n_workers: Maximum number of parallel worker processes.
        randomly_flip_options: Whether to reverse answer options in 50% of cases.
        shuffle_answer_options: Whether to randomly shuffle answer options.
        temperature: Temperature for the evaluated LLM.
        seed: Base seed for reproducibility.
        max_jury_retries: Maximum jury retries per simulator message.
    """
    out_dir = os.path.join(CONDITIONAL_RESULTS_DIR, model_name)
    os.makedirs(out_dir, exist_ok=True)

    batches = np.array_split(dataset, n_batches)

    with tqdm(total=len(batches)) as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            for _ in executor.map(
                partial(
                    decide_conditional_batch,
                    model_name=model_name,
                    human_bias=human_bias,
                    direction=direction,
                    intensity=intensity,
                    randomly_flip_options=randomly_flip_options,
                    shuffle_answer_options=shuffle_answer_options,
                    temperature=temperature,
                    seed=seed,
                    max_jury_retries=max_jury_retries,
                ),
                batches,
            ):
                pbar.update()

    # Merge all per-batch CSVs into a single results file
    all_files = [
        os.path.join(out_dir, f)
        for f in os.listdir(out_dir)
        if f.startswith("batch_") and f.endswith(".csv")
    ]
    merged = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
    merged.to_csv(
        os.path.join(CONDITIONAL_RESULTS_DIR, f"{model_name}.csv"), index=False
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate LLMs under multi-turn conditional bias interactions. "
            "For each test case in the dataset, a human simulator injects a biased "
            "message (validated by a jury) into the conversation before the LLM "
            "makes its final decision."
        )
    )
    parser.add_argument(
        "--dataset", type=str, default=DATASET_FILE_PATH,
        help="Path to the full_dataset.csv generated by dataset_assembly.py.",
    )
    parser.add_argument(
        "--model", type=str, default="GPT-4o-Mini",
        help="Name of the LLM to evaluate (must be registered in core/utils.py).",
    )
    parser.add_argument(
        "--human_bias", type=str, default=None,
        help=(
            "Cognitive bias for the human simulator to express. "
            "If omitted, uses the bias of the first test case in the dataset."
        ),
    )
    parser.add_argument(
        "--direction", type=str, default="positive", choices=["positive", "negative"],
        help="Directional orientation of the human bias.",
    )
    parser.add_argument(
        "--intensity", type=float, default=0.7,
        help="Bias intensity in [0, 1] (0=none, 1=maximum).",
    )
    parser.add_argument(
        "--n_workers", type=int, default=10,
        help="Maximum number of parallel worker processes.",
    )
    parser.add_argument(
        "--n_batches", type=int, default=100,
        help="Number of equal-sized batches to split the dataset into.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Temperature for the evaluated LLM.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base seed for reproducibility.",
    )
    parser.add_argument(
        "--max_jury_retries", type=int, default=3,
        help="Maximum attempts to generate a jury-approved simulator message per test case.",
    )
    parser.add_argument(
        "--bias", type=str, default=None,
        help="Filter dataset to only rows with this target bias (e.g. 'Framing Effect').",
    )
    parser.add_argument(
        "--n_samples", type=int, default=None,
        help="Limit the dataset to this many rows (after bias filtering).",
    )
    args = parser.parse_args()

    dataset = pd.read_csv(args.dataset)
    if args.bias:
        dataset = dataset[dataset["bias"] == args.bias].reset_index(drop=True)
    if args.n_samples:
        dataset = dataset.iloc[: args.n_samples].reset_index(drop=True)
    human_bias = args.human_bias if args.human_bias else dataset["bias"].iloc[0]

    print(
        f"Starting conditional bias evaluation:\n"
        f"  model={args.model}, human_bias={human_bias}, "
        f"direction={args.direction}, intensity={args.intensity}\n"
        f"  dataset rows={len(dataset)}, workers={args.n_workers}, batches={args.n_batches}"
    )
    start = datetime.datetime.now()
    decide_conditional_dataset(
        dataset=dataset,
        model_name=args.model,
        human_bias=human_bias,
        direction=args.direction,
        intensity=args.intensity,
        n_batches=args.n_batches,
        n_workers=args.n_workers,
        randomly_flip_options=True,
        shuffle_answer_options=False,
        temperature=args.temperature,
        seed=args.seed,
        max_jury_retries=args.max_jury_retries,
    )
    print(f"Done in {datetime.datetime.now() - start}.")


if __name__ == "__main__":
    main()
