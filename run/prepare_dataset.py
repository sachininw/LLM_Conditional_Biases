"""
Convert the HuggingFace tum-nlp/cognitive-biases-in-llms dataset into the
full_dataset.csv format expected by test_decision.py and conditional_test_decision.py.

Each row in the HF dataset has plain-text `control` and `treatment` columns (output of
Template.format()). This script parses them back into XML <template> elements and stores
the XML strings in `raw_control` / `raw_treatment` columns, matching the schema produced
by test_generation.py.
"""

import sys
import os
import re
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
from datasets import load_dataset

_RUN_DIR     = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT   = os.path.dirname(_RUN_DIR)
_BIASES_FILE = os.path.join(os.path.dirname(_REPO_ROOT), "Biases.txt")

# Name normalisation: Biases.txt spelling → HuggingFace dataset spelling
_BIAS_NAME_MAP = {
    "In-Group Bias":  "In Group Bias",
    "Status-Quo Bias": "Status Quo Bias",
}


def _read_biases_from_file(path: str = _BIASES_FILE) -> list[str]:
    """Return the normalised bias list from Biases.txt."""
    with open(path) as f:
        return [_BIAS_NAME_MAP.get(l.strip(), l.strip()) for l in f if l.strip()]


def _parse_formatted_template(text: str, template_type: str) -> str:
    """
    Parse a Template.format() output string back into a <template> XML string.

    Each non-empty line in the Situation section becomes its own <situation> element.
    The Prompt section becomes a <prompt> element.
    Each option line becomes an <option> element.
    """
    template_elem = ET.Element("template")
    template_elem.set("type", template_type)

    # Split on section headers
    # Typical format:
    #   Situation:\n<line1>\n<line2>\n...\n\nPrompt:\n<text>\n\nAnswer Options:\nOption 1: ...\n...
    sit_match = re.search(r'Situation:\n(.*?)(?:\n\nPrompt:|\Z)', text, re.DOTALL)
    prompt_match = re.search(r'\nPrompt:\n(.*?)(?:\n\nAnswer Options:|\Z)', text, re.DOTALL)
    options_match = re.search(r'\nAnswer Options:\n(.*)\Z', text, re.DOTALL)

    if sit_match:
        for line in sit_match.group(1).split('\n'):
            line = line.rstrip()
            if line:
                elem = ET.SubElement(template_elem, "situation")
                elem.text = line

    if prompt_match:
        prompt_text = prompt_match.group(1).strip()
        if prompt_text:
            elem = ET.SubElement(template_elem, "prompt")
            elem.text = prompt_text

    if options_match:
        for line in options_match.group(1).split('\n'):
            line = line.rstrip()
            m = re.match(r'Option \d+: (.+)', line)
            if m:
                elem = ET.SubElement(template_elem, "option")
                elem.text = m.group(1)

    return ET.tostring(template_elem, encoding="unicode")


def convert_hf_to_csv(
    output_path: str,
    bias_filter: str = None,
    n_samples: int = None,
    n_offset: int = 0,
    split: str = "train",
):
    """
    Download the HF dataset, convert to full_dataset.csv format, and save.

    Args:
        output_path: Path to write the CSV file.
        bias_filter: Comma-separated bias name(s) to keep,
                     e.g. "Framing Effect,Anchoring".
                     Defaults to the biases listed in Biases.txt when None.
        n_samples: Rows to keep *per bias* (after filtering and offset).
        n_offset: Skip the first n_offset rows per bias before collecting.
                  Use this to get a non-overlapping second batch of scenarios.
        split: HuggingFace dataset split (default "train").
    """
    print("Loading HuggingFace dataset tum-nlp/cognitive-biases-in-llms …")
    ds = load_dataset("tum-nlp/cognitive-biases-in-llms")[split]

    if bias_filter is not None:
        bias_set = {b.strip() for b in bias_filter.split(",")}
    else:
        bias_set = set(_read_biases_from_file())
        print(f"No --bias supplied — using Biases.txt: {sorted(bias_set)}")

    seen: dict = {}    # rows seen per bias (including skipped)
    counts: dict = {}  # rows collected per bias
    rows = []
    for i, hf_row in enumerate(ds):
        bias = hf_row["bias"]
        if bias_set and bias not in bias_set:
            continue
        seen[bias] = seen.get(bias, 0) + 1
        if seen[bias] <= n_offset:
            continue
        if n_samples and counts.get(bias, 0) >= n_samples:
            continue
        raw_control = _parse_formatted_template(hf_row["control"], "control")
        raw_treatment = _parse_formatted_template(hf_row["treatment"], "treatment")
        rows.append({
            "id": i,
            "bias": bias,
            "scenario": hf_row.get("scenario", ""),
            "raw_control": raw_control,
            "raw_treatment": raw_treatment,
            "control": hf_row["control"],
            "treatment": hf_row["treatment"],
            "metric_params": str(hf_row.get("metric_params", {})),
            "generator": "GPT-4o",
            "temperature": 0.0,
            "seed": 42,
        })
        counts[bias] = counts.get(bias, 0) + 1

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows — {dict(sorted(counts.items()))} — to {output_path}")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace cognitive-biases-in-llms dataset to full_dataset.csv"
    )
    parser.add_argument(
        "--output", type=str, default=os.path.join(".", "data", "full_dataset.csv"),
        help="Output path for the CSV file.",
    )
    parser.add_argument(
        "--bias", type=str, default=None,
        help="Comma-separated bias name(s) to include, e.g. 'Framing Effect,Anchoring'. "
             "Defaults to all biases listed in Biases.txt.",
    )
    parser.add_argument(
        "--n_samples", type=int, default=None,
        help="Limit to this many rows per bias (after filtering and offset).",
    )
    parser.add_argument(
        "--n_offset", type=int, default=0,
        help="Skip the first N rows per bias (use to get a non-overlapping batch).",
    )
    args = parser.parse_args()
    convert_hf_to_csv(args.output, bias_filter=args.bias,
                      n_samples=args.n_samples, n_offset=args.n_offset)


if __name__ == "__main__":
    main()
