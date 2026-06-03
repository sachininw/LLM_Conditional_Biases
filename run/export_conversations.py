"""
Export pilot results as full conversations for testing another LLM.

Each record contains the complete multi-turn conversation up to (but not including)
GPT-4o's final decision, so you can replay it against any other model and compare.

Conversation structure per record:
  Turn 1  user      : common_situation
  Turn 2  user      : human_message          (biased human interlocutor)
  Turn 3  assistant : treatment_intermediate  (GPT-4o intermediate response)
  Turn 4  user      : diverging_situation + options  (decision prompt)
  → GPT-4o answer saved separately for comparison

Outputs:
  conversations_treatment.jsonl  — biased human message condition
  conversations_neutral.jsonl    — neutral human message condition
  conversations_control.jsonl    — no human message (single-turn baseline)

Usage:
    python export_conversations.py
    python export_conversations.py --input path/to/pilot.csv --output_dir path/to/out/
"""

import os
import ast
import json
import argparse
import pandas as pd

_RUN_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_INPUT  = os.path.join(_RUN_DIR, "data", "conditional_decision_results", "GPT-4o_pilot.csv")
_DEFAULT_OUTPUT = os.path.join(_RUN_DIR, "data", "conversations")


def _parse_options(opts_raw) -> list[str]:
    if isinstance(opts_raw, list):
        return opts_raw
    try:
        return ast.literal_eval(str(opts_raw))
    except Exception:
        return []


def _format_options(options: list[str]) -> str:
    return "\n".join(f"Option {i+1}: {opt}" for i, opt in enumerate(options))


def _build_decision_prompt(diverging_situation: str, options: list[str]) -> str:
    parts = []
    if diverging_situation and str(diverging_situation) not in ("nan", "None", ""):
        parts.append(diverging_situation.strip())
    if options:
        parts.append("Answer Options:\n" + _format_options(options))
    return "\n\n".join(parts)


def export_conversations(input_path: str, output_dir: str):
    df = pd.read_csv(input_path)
    ok = df[df["status"] == "OK"].copy()
    os.makedirs(output_dir, exist_ok=True)

    treatment_records = []
    neutral_records   = []
    control_records   = []

    for _, row in ok.iterrows():
        base = {
            "id":          row.get("id"),
            "target_bias": row.get("bias"),
            "human_bias":  row.get("human_bias"),
            "direction":   row.get("direction"),
            "intensity":   row.get("intensity"),
            "individual_score": row.get("individual_score"),
            "neutral_score":    row.get("neutral_score"),
            "delta_score":      row.get("delta_score"),
        }

        common_sit  = str(row.get("common_situation", "")).strip()
        diverging   = str(row.get("diverging_situation", "")).strip()
        treat_opts  = _parse_options(row.get("treatment_options", "[]"))
        neutral_opts = _parse_options(row.get("neutral_options", "[]"))
        ctrl_opts   = _parse_options(row.get("control_options", "[]"))

        # ── Treatment: biased human message ───────────────────────────────────
        treat_decision_prompt = _build_decision_prompt(diverging, treat_opts)
        treatment_records.append({
            **base,
            "condition": "treatment",
            "messages": [
                {"role": "user",      "content": common_sit},
                {"role": "user",      "content": str(row.get("human_message", "")).strip()},
                {"role": "assistant", "content": str(row.get("treatment_intermediate", "")).strip()},
                {"role": "user",      "content": treat_decision_prompt},
            ],
            "gpt4o_answer":   row.get("treatment_answer"),
            "gpt4o_decision": row.get("treatment_decision"),
            "jury_passed":    row.get("jury_passed"),
        })

        # ── Neutral: unbiased human message ───────────────────────────────────
        neutral_decision_prompt = _build_decision_prompt(diverging, neutral_opts)
        neutral_records.append({
            **base,
            "condition": "neutral",
            "messages": [
                {"role": "user",      "content": common_sit},
                {"role": "user",      "content": str(row.get("neutral_message", "")).strip()},
                {"role": "assistant", "content": str(row.get("neutral_intermediate", "")).strip()},
                {"role": "user",      "content": neutral_decision_prompt},
            ],
            "gpt4o_answer":         row.get("neutral_answer"),
            "gpt4o_decision":       row.get("neutral_decision"),
            "neutral_jury_passed":  row.get("neutral_jury_passed"),
        })

        # ── Control: no human message (single turn) ───────────────────────────
        ctrl_decision_prompt = _build_decision_prompt(diverging, ctrl_opts)
        control_records.append({
            **base,
            "condition": "control",
            "messages": [
                {"role": "user", "content": ctrl_decision_prompt},
            ],
            "gpt4o_answer":   row.get("control_answer"),
            "gpt4o_decision": row.get("control_decision"),
        })

    def write_jsonl(records, fname):
        path = os.path.join(output_dir, fname)
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r, default=str) + "\n")
        print(f"Saved {len(records)} records → {path}")

    write_jsonl(treatment_records, "conversations_treatment.jsonl")
    write_jsonl(neutral_records,   "conversations_neutral.jsonl")
    write_jsonl(control_records,   "conversations_control.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      default=_DEFAULT_INPUT)
    parser.add_argument("--output_dir", default=_DEFAULT_OUTPUT)
    args = parser.parse_args()
    export_conversations(args.input, args.output_dir)
