<p align="center">
  <img src="figures/score_distributions_jp.png" width="80%" alt="Bias score distributions across three evaluation conditions for Llama-3.1-8B">
</p>

<h1 align="center">Conditional Cognitive Bias in Instruction-Tuned LLMs</h1>
<h3 align="center"><em>How Biased User Turns Modulate In-Context Reasoning</em></h3>

<p align="center">
  <a href="https://arxiv.org/"><img src="https://img.shields.io/badge/paper-arXiv-red" alt="Paper"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-CC%20BY--SA%204.0-blue" alt="License"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-brightgreen" alt="Python">
</p>

<p align="center">
  <strong>Sachini Weerasekara &nbsp;·&nbsp; Sagar Kamarthi &nbsp;·&nbsp; Jacqueline Isaacs</strong><br>
  Northeastern University
</p>

---

Prior work on LLM cognitive bias measured models under zero-shot, context-free prompts — a setting that does not reflect deployment, where every response is conditioned on a preceding user turn. This paper asks: *does the content of that user turn causally modulate bias expression?*

We introduce a three-condition design (zero-shot / neutral turn / biased turn) across 8 instruction-tuned LLMs and 9 cognitive biases, with a 24,300-message stimulus bank validated by a three-model LLM jury. Key findings:

- Biased user turns **elevate bias above zero-shot in 6 of 8 models** (Δb = +0.022–+0.051, p < 0.001).
- Two mechanisms operate in opposition: a **presence effect** (any conversational turn inflates bias) is partially cancelled by a **content effect** (explicit bias in the user turn triggers alignment-driven suppression).
- **Planning Fallacy** is the only bias causally confirmed as universally inducible across all 8 models.

---

## Contents

- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Reproducing the Experiments](#reproducing-the-experiments)
- [Project Structure](#project-structure)
- [Extending the Framework](#extending-the-framework)
- [Data](#data)
- [Citation](#citation)

---

## Results

### Effect decomposition across models

| Model | Zero-shot (\|m∅\|) | Neutral (\|mn\|) | Biased (\|mb\|) | Presence (Δn) | Content (δ) | Total (Δb) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Llama-3.1-8B     | 0.350 | 0.436 | 0.401 | +0.086*** | −0.035*** | +0.051*** |
| Llama-3.1-70B    | 0.424 | 0.485 | 0.446 | +0.061*** | −0.039*** | +0.022*** |
| Claude-3.5-Haiku | 0.393 | 0.293 | 0.330 | −0.101*** | +0.037*** | −0.064*** |
| GPT-4o           | 0.378 | 0.397 | 0.378 | +0.019*** | −0.020*** | ~0.000    |
| DeepSeek-V3      | 0.433 | 0.458 | 0.467 | +0.025*** | +0.009    | +0.034*** |
| Qwen-2.5-72B     | 0.367 | 0.414 | 0.407 | +0.046*** | −0.007    | +0.040*** |
| Phi-4            | 0.437 | 0.488 | 0.470 | +0.052*** | −0.018*** | +0.033*** |
| Gemma-2-9B-IT    | 0.342 | 0.400 | 0.372 | +0.058*** | −0.028*** | +0.030*** |

\*\*\* p < 0.001 · values are mean absolute bias magnitudes

### Bias coupling (Llama-3.1-8B)

Rows = target bias (β) · columns = human-turn bias (γ)

<table>
<tr>
<td align="center">
  <img src="figures/heatmap_biased_jp.png" width="100%" alt="Biased-condition heatmap"><br>
  <em>Biased condition (|mb|)</em>
</td>
<td align="center">
  <img src="figures/heatmap_neutral_jp.png" width="100%" alt="Neutral-condition heatmap"><br>
  <em>Neutral condition (|mn|)</em>
</td>
<td align="center">
  <img src="figures/heatmap_delta_jp.png" width="100%" alt="Content effect heatmap"><br>
  <em>Content effect (δ = |mb| − |mn|)</em>
</td>
</tr>
</table>

**Planning Fallacy** is the only target bias with a consistently positive content effect across all models. **Bandwagon Effect** and **Availability Heuristic** are most suppressible.

### Significance across bias pairs (Llama-3.1-8B)

<table>
<tr>
<td align="center">
  <img src="figures/sig_bonferroni_jp.png" width="100%" alt="Bonferroni-corrected p-values"><br>
  <em>Bonferroni-corrected p-values</em>
</td>
<td align="center">
  <img src="figures/sig_cohens_d_jp.png" width="100%" alt="Cohen's d effect sizes"><br>
  <em>Cohen's d effect sizes</em>
</td>
</tr>
</table>

---

## Installation

**Requirements:** Python 3.10+

```bash
git clone https://github.com/sachininw/LLM_Conditional_Biases.git
cd LLM_Conditional_Biases
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in the API keys for the providers you need:

| Provider | Variable | Models |
|:---|:---|:---|
| OpenAI | `OPENAI_API_KEY` | GPT-4o |
| Anthropic | `ANTHROPIC_API_KEY` | Claude-3.5-Haiku |
| Google | `GOOGLE_API` | Gemini-1.5-Pro, Gemini-2.5-Flash |
| DeepInfra | `DEEPINFRA_API` | Llama-3.1-8B/70B, Qwen-2.5-72B, DeepSeek-V3, Phi-4, Gemma-2-9B-IT |

---

## Usage

> **Note:** Pre-generated data (message bank and decision results) is available for download — see [Data](#data). Steps 1–5 below can be skipped if you use the pre-generated data.

Run the three-condition evaluation on a single model:

```bash
python run/conditional_test_decision.py --model "GPT-4o" --n_workers 50 --n_batches 1000
```

Evaluate all models and run causal analysis:

```bash
bash run/run_all_models.sh
python run/causal_analysis.py
```

---

## Reproducing the Experiments

### 1. Generate scenarios

```bash
python run/scenario_generation.py --model "GPT-4o" --n_positions 8
```

Produces 200 scenario strings (8 managerial positions × 25 GICS sectors) in `data/scenarios.txt`. The file used in the paper is included.

### 2. Build the message bank

```bash
python run/generate_message_bank.py --generator_model "GPT-4o" --n_messages 300
```

For each of the 81 cells (9 target biases × 9 human-turn biases), generates user-turn candidates via persona-injection prompting and filters them through a three-model LLM jury (GPT-4-Turbo, Claude Haiku, GPT-4o-Mini) using four quality criteria: target rating ≥ 3.0, purity, naturalness, and blind-identification accuracy. Failed turns are regenerated with diagnostic feedback up to 3 times. Final bank: 24,300 validated pairs.

### 3. Generate test cases

```bash
python run/test_generation.py [--bias "FramingEffect,PlanningFallacy"] [--num_instances 2]
```

Generates test instances for each of the 9 target biases. Output: `data/generated_tests/`, `data/generation_logs/`, `data/generated_datasets/`.

### 4. (Optional) Spot-check generated tests

```bash
python run/test_check.py --bias "PlanningFallacy" --n_sample 10
```

### 5. Assemble the dataset

```bash
python run/dataset_assembly.py
```

Merges per-bias CSVs into `data/full_dataset.csv`.

### 6. Run the three-condition evaluation

```bash
python run/conditional_test_decision.py --model "Llama-3.1-8B" --n_workers 50 --n_batches 1000
```

Evaluates the model under zero-shot, neutral-turn, and biased-turn conditions. Results written to `run/data/conditional_decision_results/{model}/`.

### 7. Causal analysis

```bash
python run/causal_analysis.py
```

Runs Difference-in-Differences, Synthetic Control, and Propensity Score Matching. Produces per-model figures in `figures/`.

---

## Project Structure

```
LLM_Conditional_Biases/
├── core/
│   ├── base.py          # Abstract LLM, TestGenerator, Metric base classes
│   ├── testing.py       # TestCase, Template, DecisionResult data classes
│   ├── utils.py         # Model/bias registry
│   └── add_test.py      # Scaffold for adding a new bias
│
├── models/              # One directory per provider: model.py + prompts.yml
│   ├── Anthropic/
│   ├── DeepSeek/
│   ├── Google/
│   ├── Meta/
│   ├── Microsoft/
│   ├── Alibaba/
│   └── OpenAI/
│
├── tests/               # Test definitions for 9 target biases
│   ├── Anchoring/       # config.xml (templates) + test.py (generator + metric)
│   ├── AvailabilityHeuristic/
│   ├── BandwagonEffect/
│   ├── ConfirmationBias/
│   ├── FramingEffect/
│   ├── InGroupBias/
│   ├── LossAversion/
│   ├── PlanningFallacy/
│   └── StatusQuoBias/
│
├── run/
│   ├── scenario_generation.py
│   ├── generate_message_bank.py
│   ├── human_simulator.py       # LLM-driven persona for biased user turns
│   ├── bias_jury.py             # Three-model validation jury
│   ├── test_generation.py
│   ├── dataset_assembly.py
│   ├── conditional_test_decision.py
│   ├── causal_analysis.py
│   ├── evaluate_results.py
│   ├── run_all_models.sh
│   └── data/                    # Generated outputs (gitignored)
│
├── data/
│   ├── scenarios.txt            # 200 decision-making scenarios (committed)
│   ├── generated_tests/
│   ├── generated_datasets/
│   └── decision_results/
│
├── figures/             # Plots used in this README
├── demo.py              # Minimal end-to-end example
├── .env.example
├── requirements.txt
└── LICENSE
```

---

## Extending the Framework

### Add a new cognitive bias

```bash
python core/add_test.py
```

Creates `tests/<BiasName>/` with `config.xml` (control/treatment templates) and `test.py`. Implement `generate()` in the `TestGenerator` subclass and `compute()` in the `Metric` subclass, then register the bias in `core/utils.py`.

Minimal `test.py`:

```python
from core.base import TestGenerator, RatioScaleMetric
from core.testing import TestCase

class MyBiasTestGenerator(TestGenerator):
    def __init__(self):
        self.BIAS = "MyBias"
        self.config = super().load_config(self.BIAS)

    def generate(self, model, scenario, custom_values={}, temperature=0.0, seed=42):
        control, treatment = self.config.get_control_template(), self.config.get_treatment_template()
        control, treatment = super().populate(model, control, treatment, scenario, temperature, seed)
        return TestCase(bias=self.BIAS, control=control, treatment=treatment,
                        generator=model.NAME, temperature=temperature, seed=seed,
                        scenario=scenario, variant=None, remarks=None)

class MyBiasMetric(RatioScaleMetric):
    def __init__(self, test_results):
        super().__init__(test_results)
        self.k = 1
```

### Add a new model

Create `models/<Provider>/model.py` subclassing `LLM`, copy and adapt `prompts.yml` from an existing provider, then register in `core/utils.py`:

```python
from core.base import LLM

class MyModel(LLM):
    NAME = "provider/model-name"

    def __init__(self, randomly_flip_options=True, shuffle_answer_options=False):
        super().__init__(randomly_flip_options, shuffle_answer_options)

    def prompt(self, prompt: str, temperature: float = 0.0, seed: int = 42) -> str:
        ...

    def prompt_multiturn(self, messages: list[dict], temperature: float = 0.0, seed: int = 42) -> str:
        ...
```

---

## Data

| Resource | Location |
|:---|:---|
| Scenarios (`data/scenarios.txt`) | Included in this repository |
| Message bank (24,300 validated pairs) | [Dropbox](https://www.dropbox.com/scl/fo/a2c75wjso016f743fspvy/ALXH_sTUkvUDSfZCS-3Z3a8?rlkey=xg5wrfjj8207vhqk2ykxyqbn3&st=guv1u25w&dl=0) |
| Decision results (all models) | [Dropbox](https://www.dropbox.com/scl/fo/a2c75wjso016f743fspvy/ALXH_sTUkvUDSfZCS-3Z3a8?rlkey=xg5wrfjj8207vhqk2ykxyqbn3&st=guv1u25w&dl=0) |

---

## Citation

```bibtex
@article{weerasekara2025conditional,
  title   = {Conditional Cognitive Bias in Instruction-Tuned {LLM}s:
             How Biased User Turns Modulate In-Context Reasoning},
  author  = {Weerasekara, Sachini and Kamarthi, Sagar and Isaacs, Jacqueline},
  year    = {2025},
  note    = {Northeastern University},
}
```
