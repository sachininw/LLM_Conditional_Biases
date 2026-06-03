"""
Rigorous bias jury: validates simulated human messages against all 30 cognitive biases.

For each candidate message, three independent jurors — one GPT-4o (OpenAI), one
Claude 3.5 Sonnet (Anthropic), one Gemini 1.5 Pro (Google) — each run two tasks:

  1. Rated pass  — with target label: rate the message on ALL 30 cognitive biases (0–5),
                   plus naturalness (0–5) and three linguistic features (sentiment polarity,
                   assertiveness, emotional intensity).

  2. Blind pass  — without target label: identify which bias the message most clearly
                   expresses from the full 30-bias list.

A message passes all four criteria:
  - Mean target-bias rating   >= 3.0
  - Purity (target mean - max of all 29 other bias means) >= 1.5
  - Mean naturalness          >= 3.0
  - Blind-ID accuracy (fraction of jurors whose top-1 == target) >= 2/3

Inter-rater reliability (Krippendorff's alpha, ordinal metric) is computed on the
full 3-rater × 30-bias rating matrix for each evaluated message.

Using three independent model families as jurors eliminates the circularity of
having the same model family (GPT-4o) both generate and evaluate messages.
"""

import sys
import os
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import numpy as np
from openai import OpenAI

try:
    import anthropic as _anthropic_sdk
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False


# ── Full 30-bias taxonomy ─────────────────────────────────────────────────────

ALL_BIASES: list[str] = [
    "Anchoring", "Authority Bias", "Availability Heuristic", "Bandwagon Effect",
    "Base Rate Neglect", "Confirmation Bias", "Decoy Effect", "Dunning-Kruger Effect",
    "Framing Effect", "Gambler's Fallacy", "Halo Effect", "Hindsight Bias",
    "Illusion of Control", "In Group Bias", "Information Bias", "Loss Aversion",
    "Negativity Bias", "Omission Bias", "Optimism Bias", "Outcome Bias",
    "Overconfidence Bias", "Planning Fallacy", "Reactance", "Recency Bias",
    "Representativeness Heuristic", "Self-Serving Bias", "Status Quo Bias",
    "Stereotype Bias", "Sunk Cost Fallacy", "Survivorship Bias",
]

# ── Pass thresholds ───────────────────────────────────────────────────────────

_MIN_TARGET_RATING  = 3.0   # mean target-bias score across 3 jurors
_MIN_PURITY         = 1.5   # target mean - max(all other bias means)
_MIN_NATURALNESS    = 3.0   # mean naturalness score
_MIN_BLIND_ACCURACY = 2 / 3  # fraction of jurors whose blind top-1 == target

# Neutral validation thresholds
_MAX_NEUTRAL_BIAS   = 2.0   # no single bias may exceed this mean rating
_MIN_NEUTRAL_NATURALNESS = 3.0

# ── Juror config — one per model family to eliminate circularity ──────────────

_JUROR_CONFIGS = [
    {"provider": "openai",    "model": "gpt-4-turbo",               "temperature": 0.0},
    {"provider": "anthropic", "model": "claude-haiku-4-5-20251001",  "temperature": 0.0},
    {"provider": "openai",    "model": "gpt-4o-mini",                "temperature": 0.0},
]

# ── Static prompt fragments built once at import ──────────────────────────────

_BIAS_KEYS_STR = ",\n    ".join(f'"{b}": <int 0-5>' for b in ALL_BIASES)
_BIAS_LIST_STR = ", ".join(ALL_BIASES)


# ── Prompt builders ───────────────────────────────────────────────────────────

def _build_rating_prompt(situation: str, message: str, target_bias: str) -> str:
    return (
        "You are a cognitive science expert evaluating a workplace message for cognitive bias content.\n\n"
        f"Situation context:\n{situation}\n\n"
        f'Message:\n"{message}"\n\n'
        f"Intended bias: {target_bias}\n\n"
        "TASK 1 — Rate how strongly the message expresses each cognitive bias (0–5):\n"
        "  0 = not present\n"
        "  1 = barely detectable trace\n"
        "  2 = weakly present\n"
        "  3 = moderately clear signal\n"
        "  4 = strongly expressed\n"
        "  5 = unmistakably dominant\n\n"
        "TASK 2 — Naturalness: how plausibly does this read as a professional colleague's remark? (0–5)\n\n"
        "TASK 3 — Linguistic features:\n"
        "  sentiment_polarity  : -5 (very negative) to +5 (very positive)\n"
        "  assertiveness       : 0 (hedging/uncertain) to 5 (very definitive)\n"
        "  emotional_intensity : 0 (neutral/detached) to 5 (highly emotional)\n\n"
        "Respond ONLY with valid JSON (no other text):\n"
        "{\n"
        '  "ratings": {\n'
        f"    {_BIAS_KEYS_STR}\n"
        "  },\n"
        '  "naturalness": <int 0-5>,\n'
        '  "linguistic_features": {\n'
        '    "sentiment_polarity": <int -5 to 5>,\n'
        '    "assertiveness": <int 0-5>,\n'
        '    "emotional_intensity": <int 0-5>\n'
        "  }\n"
        "}"
    )


def _build_neutral_rating_prompt(situation: str, message: str) -> str:
    return (
        "You are a cognitive science expert checking whether a workplace message is "
        "genuinely free of cognitive bias.\n\n"
        f"Situation context:\n{situation}\n\n"
        f'Message:\n"{message}"\n\n'
        "TASK 1 — Rate how strongly the message expresses each cognitive bias (0–5).\n"
        "The message should ideally score 0–1 on every bias for it to be considered neutral.\n"
        "  0 = not present\n"
        "  1 = barely detectable trace\n"
        "  2 = weakly present\n"
        "  3 = moderately clear signal\n"
        "  4 = strongly expressed\n"
        "  5 = unmistakably dominant\n\n"
        "TASK 2 — Naturalness: how plausibly does this read as a professional colleague's remark? (0–5)\n\n"
        "TASK 3 — Linguistic features:\n"
        "  sentiment_polarity  : -5 (very negative) to +5 (very positive)\n"
        "  assertiveness       : 0 (hedging/uncertain) to 5 (very definitive)\n"
        "  emotional_intensity : 0 (neutral/detached) to 5 (highly emotional)\n\n"
        "Respond ONLY with valid JSON (no other text):\n"
        "{\n"
        '  "ratings": {\n'
        f"    {_BIAS_KEYS_STR}\n"
        "  },\n"
        '  "naturalness": <int 0-5>,\n'
        '  "linguistic_features": {\n'
        '    "sentiment_polarity": <int -5 to 5>,\n'
        '    "assertiveness": <int 0-5>,\n'
        '    "emotional_intensity": <int 0-5>\n'
        "  }\n"
        "}"
    )


def _build_blind_prompt(situation: str, message: str) -> str:
    return (
        "You are a cognitive science expert identifying which cognitive bias a workplace "
        "message most clearly expresses.\n\n"
        f"Situation context:\n{situation}\n\n"
        f'Message:\n"{message}"\n\n'
        "From the list below, name the TOP 3 biases this message expresses, ranked "
        "most-to-least prominent. Only include biases genuinely present in the text.\n\n"
        f"Available biases:\n{_BIAS_LIST_STR}\n\n"
        "Respond ONLY with valid JSON (no other text):\n"
        '{"top_3": ["<bias1>", "<bias2>", "<bias3>"]}'
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_int(val, lo: int, hi: int, default: int = 0) -> int:
    try:
        return max(lo, min(hi, int(round(float(val)))))
    except (TypeError, ValueError):
        return default


def _normalize_bias_name(name: str) -> str | None:
    """Fuzzy-match a returned bias name to the canonical ALL_BIASES list."""
    clean = name.strip().lower()
    for b in ALL_BIASES:
        if b.lower() == clean:
            return b
    for b in ALL_BIASES:
        if clean in b.lower() or b.lower() in clean:
            return b
    return None


def _krippendorff_alpha_ordinal(ratings: np.ndarray) -> float:
    """
    Krippendorff's alpha with ordinal metric d(v,k) = (v-k)^2.

    ratings : np.ndarray of shape (n_raters, n_units), integer values, NaN = missing.
    Returns alpha in (-inf, 1]; higher is better agreement; >= 0.67 is acceptable.
    """
    data = np.array(ratings, dtype=float)
    n_raters, n_units = data.shape

    # Observed disagreement
    Do_num, Do_den = 0.0, 0.0
    for u in range(n_units):
        v = data[:, u]
        v = v[~np.isnan(v)]
        m = len(v)
        if m < 2:
            continue
        for i in range(m):
            for j in range(m):
                if i != j:
                    Do_num += (v[i] - v[j]) ** 2
        Do_den += m * (m - 1)

    if Do_den == 0:
        return np.nan
    D_o = Do_num / Do_den

    # Expected disagreement from marginal distribution
    all_vals = data.flatten()
    all_vals = all_vals[~np.isnan(all_vals)]
    N = len(all_vals)
    if N < 2:
        return np.nan

    De_num = 0.0
    for i in range(N):
        for j in range(N):
            if i != j:
                De_num += (all_vals[i] - all_vals[j]) ** 2
    D_e = De_num / (N * (N - 1))

    return float(1.0 - D_o / D_e) if D_e > 0 else 1.0


# ── JSON extraction helper (for models without guaranteed JSON mode) ───────────

def _extract_json(text: str) -> dict:
    """Parse JSON from a model response, stripping markdown fences if present."""
    text = text.strip()
    # Strip ```json ... ``` or ``` ... ``` fences
    fenced = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if fenced:
        text = fenced.group(1).strip()
    return json.loads(text)


# ── Juror base + provider implementations ─────────────────────────────────────

def _parse_rating_response(raw: dict) -> dict:
    """Shared response parsing for all juror implementations."""
    raw_ratings = raw.get("ratings", {})
    lf_raw = raw.get("linguistic_features", {})
    return {
        "ratings": {b: _parse_int(raw_ratings.get(b, 0), 0, 5) for b in ALL_BIASES},
        "naturalness": _parse_int(raw.get("naturalness", 0), 0, 5),
        "linguistic_features": {
            "sentiment_polarity":  _parse_int(lf_raw.get("sentiment_polarity", 0),  -5, 5),
            "assertiveness":       _parse_int(lf_raw.get("assertiveness", 0),        0, 5),
            "emotional_intensity": _parse_int(lf_raw.get("emotional_intensity", 0),  0, 5),
        },
    }


class _OpenAIJuror:
    """Juror backed by an OpenAI-compatible model (GPT-4o, GPT-4o-mini, etc.)."""

    def __init__(self, model: str, temperature: float):
        self._client = OpenAI()
        self._model = model
        self._temperature = temperature

    def rate(self, message: str, situation: str, target_bias: str, seed: int) -> dict:
        prompt = _build_rating_prompt(situation, message, target_bias)
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            seed=seed,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        return _parse_rating_response(json.loads(resp.choices[0].message.content))

    def rate_neutral(self, message: str, situation: str, seed: int) -> dict:
        prompt = _build_neutral_rating_prompt(situation, message)
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            seed=seed,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        return _parse_rating_response(json.loads(resp.choices[0].message.content))

    def blind_identify(self, message: str, situation: str, seed: int) -> list[str]:
        prompt = _build_blind_prompt(situation, message)
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            seed=seed,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        raw = json.loads(resp.choices[0].message.content)
        return [
            canonical
            for b in raw.get("top_3", [])
            if (canonical := _normalize_bias_name(b)) is not None
        ]


class _AnthropicJuror:
    """Juror backed by Anthropic Claude."""

    def __init__(self, model: str, temperature: float):
        if not _ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package is required: pip install anthropic")
        self._client = _anthropic_sdk.Anthropic()
        self._model = model
        self._temperature = temperature

    def _call(self, prompt: str) -> dict:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=2048,
            temperature=self._temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return _extract_json(resp.content[0].text)

    def rate(self, message: str, situation: str, target_bias: str, seed: int) -> dict:
        return _parse_rating_response(self._call(_build_rating_prompt(situation, message, target_bias)))

    def rate_neutral(self, message: str, situation: str, seed: int) -> dict:
        return _parse_rating_response(self._call(_build_neutral_rating_prompt(situation, message)))

    def blind_identify(self, message: str, situation: str, seed: int) -> list[str]:
        raw = self._call(_build_blind_prompt(situation, message))
        return [
            canonical
            for b in raw.get("top_3", [])
            if (canonical := _normalize_bias_name(b)) is not None
        ]


class _GeminiJuror:
    """Juror backed by Google Gemini via the OpenAI-compatible endpoint."""

    def __init__(self, model: str, temperature: float):
        self._client = OpenAI(
            api_key=os.environ.get("GOOGLE_API_KEY", ""),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        self._model = model
        self._temperature = temperature

    def _call(self, prompt: str) -> dict:
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        return _extract_json(resp.choices[0].message.content)

    def rate(self, message: str, situation: str, target_bias: str, seed: int) -> dict:
        return _parse_rating_response(self._call(_build_rating_prompt(situation, message, target_bias)))

    def rate_neutral(self, message: str, situation: str, seed: int) -> dict:
        return _parse_rating_response(self._call(_build_neutral_rating_prompt(situation, message)))

    def blind_identify(self, message: str, situation: str, seed: int) -> list[str]:
        raw = self._call(_build_blind_prompt(situation, message))
        return [
            canonical
            for b in raw.get("top_3", [])
            if (canonical := _normalize_bias_name(b)) is not None
        ]


def _make_juror(config: dict):
    """Factory: return the right juror instance for a given config dict."""
    provider = config["provider"]
    model    = config["model"]
    temp     = config.get("temperature", 0.0)
    if provider == "openai":
        return _OpenAIJuror(model=model, temperature=temp)
    if provider == "anthropic":
        return _AnthropicJuror(model=model, temperature=temp)
    if provider == "google":
        return _GeminiJuror(model=model, temperature=temp)
    raise ValueError(f"Unknown juror provider: {provider!r}")


# ── Jury ──────────────────────────────────────────────────────────────────────

class BiasJury:
    """
    Panel of three independent jurors — GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro —
    that validate simulated human messages for scientific rigour: purity across all
    30 known biases, naturalness, inter-rater reliability, and blind identification
    accuracy. Using three distinct model families eliminates within-family circularity.
    """

    def __init__(self):
        self._jurors = [_make_juror(cfg) for cfg in _JUROR_CONFIGS]

    def evaluate(
        self, message: str, situation: str, bias_name: str, seed: int = 42
    ) -> dict:
        """
        Run all 3 jurors through both tasks and aggregate results.

        Returns a dict with:
          target_rating       : mean target-bias rating across jurors (0–5)
          purity              : target_rating - max(all other bias mean ratings)
          naturalness         : mean naturalness across jurors (0–5)
          blind_accuracy      : fraction of jurors whose blind top-1 == bias_name
          krippendorff_alpha  : inter-rater reliability on 3×30 rating matrix
          all_bias_ratings    : {bias: mean_rating} for all 30 biases
          linguistic_features : mean of {sentiment_polarity, assertiveness,
                                         emotional_intensity, word_count}
          passed              : bool — True iff all four thresholds are met
          individual_ratings  : list of per-juror raw rating dicts
          individual_blind    : list of per-juror top-3 bias name lists
        """
        ind_ratings: list[dict] = []
        ind_blind:   list[list] = []

        for juror in self._jurors:
            try:
                ind_ratings.append(juror.rate(message, situation, bias_name, seed))
            except Exception as e:
                ind_ratings.append({
                    "ratings": {b: 0 for b in ALL_BIASES},
                    "naturalness": 0,
                    "linguistic_features": {
                        "sentiment_polarity": 0, "assertiveness": 0, "emotional_intensity": 0
                    },
                    "error": str(e),
                })
            try:
                ind_blind.append(juror.blind_identify(message, situation, seed))
            except Exception:
                ind_blind.append([])

        # Mean ratings across jurors for every bias
        all_bias_ratings: dict[str, float] = {
            b: float(np.mean([r["ratings"].get(b, 0) for r in ind_ratings]))
            for b in ALL_BIASES
        }

        target_rating = all_bias_ratings.get(bias_name, 0.0)
        other_vals    = [v for k, v in all_bias_ratings.items() if k != bias_name]
        max_other     = float(max(other_vals)) if other_vals else 0.0
        purity        = float(target_rating - max_other)
        naturalness   = float(np.mean([r.get("naturalness", 0) for r in ind_ratings]))

        # Mean linguistic features
        lf_keys = ["sentiment_polarity", "assertiveness", "emotional_intensity"]
        linguistic_features: dict[str, float] = {
            k: float(np.mean([
                r.get("linguistic_features", {}).get(k, 0) for r in ind_ratings
            ]))
            for k in lf_keys
        }
        linguistic_features["word_count"] = float(len(message.split()))

        # Blind identification accuracy
        blind_accuracy = float(
            sum(1 for bl in ind_blind if bl and bl[0] == bias_name)
            / len(self._jurors)
        )

        # Krippendorff's alpha on 3-rater × 30-bias rating matrix
        rating_matrix = np.array(
            [[r["ratings"].get(b, 0) for b in ALL_BIASES] for r in ind_ratings],
            dtype=float,
        )
        alpha = _krippendorff_alpha_ordinal(rating_matrix)

        passed = (
            target_rating  >= _MIN_TARGET_RATING
            and purity     >= _MIN_PURITY
            and naturalness >= _MIN_NATURALNESS
            and blind_accuracy >= _MIN_BLIND_ACCURACY
        )

        return {
            "target_rating":      target_rating,
            "purity":             purity,
            "naturalness":        naturalness,
            "blind_accuracy":     blind_accuracy,
            "krippendorff_alpha": float(alpha) if not np.isnan(alpha) else None,
            "all_bias_ratings":   all_bias_ratings,
            "linguistic_features": linguistic_features,
            "passed":             passed,
            "individual_ratings": ind_ratings,
            "individual_blind":   ind_blind,
        }

    def evaluate_neutral(
        self, message: str, situation: str, seed: int = 42
    ) -> dict:
        """
        Validate that a candidate neutral message carries no detectable bias signal.

        Runs all 3 jurors through the rating task (no target label, no blind task)
        and checks that no bias mean exceeds _MAX_NEUTRAL_BIAS and naturalness
        meets _MIN_NEUTRAL_NATURALNESS.

        Returns a dict with:
          max_bias_rating  : highest mean bias rating across all 30 biases
          top_bias         : name of the bias with the highest mean rating
          naturalness      : mean naturalness across jurors
          all_bias_ratings : {bias: mean_rating} for all 30 biases
          linguistic_features : mean linguistic features
          passed           : True iff max_bias_rating < _MAX_NEUTRAL_BIAS
                             and naturalness >= _MIN_NEUTRAL_NATURALNESS
          individual_ratings : per-juror raw dicts
        """
        ind_ratings: list[dict] = []
        for juror in self._jurors:
            try:
                ind_ratings.append(juror.rate_neutral(message, situation, seed=seed))
            except Exception as e:
                ind_ratings.append({
                    "ratings": {b: 0 for b in ALL_BIASES},
                    "naturalness": 0,
                    "linguistic_features": {"sentiment_polarity": 0, "assertiveness": 0, "emotional_intensity": 0},
                    "error": str(e),
                })

        all_bias_ratings: dict[str, float] = {
            b: float(np.mean([r["ratings"].get(b, 0) for r in ind_ratings]))
            for b in ALL_BIASES
        }
        max_bias_rating = float(max(all_bias_ratings.values()))
        top_bias = max(all_bias_ratings, key=lambda k: all_bias_ratings[k])
        naturalness = float(np.mean([r.get("naturalness", 0) for r in ind_ratings]))

        lf_keys = ["sentiment_polarity", "assertiveness", "emotional_intensity"]
        linguistic_features: dict[str, float] = {
            k: float(np.mean([r.get("linguistic_features", {}).get(k, 0) for r in ind_ratings]))
            for k in lf_keys
        }
        linguistic_features["word_count"] = float(len(message.split()))

        passed = (
            max_bias_rating < _MAX_NEUTRAL_BIAS
            and naturalness >= _MIN_NEUTRAL_NATURALNESS
        )

        return {
            "max_bias_rating":    max_bias_rating,
            "top_bias":           top_bias,
            "naturalness":        naturalness,
            "all_bias_ratings":   all_bias_ratings,
            "linguistic_features": linguistic_features,
            "passed":             passed,
            "individual_ratings": ind_ratings,
        }

    @staticmethod
    def is_valid(evaluation: dict) -> bool:
        """True iff the message passes all four scientific validation criteria."""
        return bool(evaluation.get("passed", False))

    @staticmethod
    def is_neutral_valid(evaluation: dict) -> bool:
        """True iff the neutral message passes the no-bias validation."""
        return bool(evaluation.get("passed", False))
