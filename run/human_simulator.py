import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from openai import OpenAI


# How each bias should concretely manifest in a spoken workplace message.
# Each entry describes the linguistic/rhetorical move the simulator should make,
# distinct from simply saying "be positive" or "be negative".
_BIAS_MECHANISMS: dict[str, str] = {
    "Framing Effect": (
        "Take ONE specific number or outcome and present it in a way that makes the framing direction "
        "unmistakably clear to any reader. "
        "Positive direction: state the fact using gain/success/opportunity language — "
        "e.g., '9 out of 10 complete on time' or 'a 90% success rate' rather than '10% fail'. "
        "Negative direction: state the exact same fact using loss/failure/risk language — "
        "e.g., '1 in 10 fail' or 'a 10% failure rate' rather than '90% succeed'. "
        "The underlying number must be IDENTICAL — only the verbal frame (success vs. failure, "
        "gain vs. loss, opportunity vs. risk) changes. "
        "Make the reframing explicit and salient — a reader should immediately recognise "
        "that you are presenting one side of the same coin, not hiding it behind neutral language."
    ),
    "Anchoring": (
        "Drop a specific number early in your message that will serve as a reference point. "
        "Positive direction: anchor high (e.g., cite an impressive benchmark, past peak, or aspirational target). "
        "Negative direction: anchor low (e.g., cite a disappointing prior result or a pessimistic baseline). "
        "State the anchor as a matter-of-fact observation, then let it implicitly pull the judgment."
    ),
    "Availability Heuristic": (
        "Make a vivid, memorable event feel more likely by describing it in concrete detail. "
        "Positive direction: vividly recall a recent striking success to make good outcomes feel probable. "
        "Negative direction: vividly recall a recent failure or near-miss to make bad outcomes feel probable. "
        "Use sensory or emotional language ('I still remember when…', 'Just last quarter…')."
    ),
    "Confirmation Bias": (
        "Selectively cite only the evidence that supports one conclusion while ignoring contradictory data. "
        "Positive direction: mention only the supporting metrics and dismiss or omit the counter-evidence. "
        "Negative direction: highlight only the data that supports a pessimistic conclusion. "
        "Present the cherry-picked evidence as if it is the full picture."
    ),
    "Loss Aversion": (
        "Frame the decision asymmetrically so that losses loom larger than equivalent gains. "
        "Negative direction (typical): emphasise what could be lost or put at risk rather than potential gains "
        "('we stand to lose X' rather than 'we could gain X'). "
        "Positive direction (reverse): downplay losses and emphasise certain gains to encourage action."
    ),
    "Optimism Bias": (
        "Express a specific, overconfident probability estimate or prediction that clearly exceeds "
        "what evidence supports. This is NOT general enthusiasm — it is a miscalibrated belief "
        "about how likely a good (or bad) outcome is. "
        "Positive direction: state that the probability of success is much higher than it realistically is, "
        "or that the probability of failure is negligible. Use explicit probability language: "
        "'the chances of a problem are very low', 'I'd put this at 90% likely to work out', "
        "'I really can't see a realistic path to failure here'. "
        "Underestimate specific risks or obstacles as minor or irrelevant. "
        "Negative direction: state that failure is overwhelmingly likely beyond what evidence supports — "
        "overestimate risks, treat bad outcomes as near-certain."
    ),
    "Planning Fallacy": (
        "Make a specific underestimate (or overestimate) of the TIME, COST, or EFFORT required. "
        "This bias is strictly about resource estimation — do NOT express general optimism about "
        "outcomes or quality, and do NOT mix in confidence, enthusiasm, or other biases. "
        "Positive direction (underestimation): reference specific tasks or phases and assign them "
        "unrealistically tight timelines or budgets — 'the integration should take two days', "
        "'we can ship this in three weeks if we stay focused', 'it's a straightforward job, "
        "maybe a week of work'. Dismiss the need for contingency time explicitly. "
        "Negative direction: assign wildly inflated time or cost estimates to specific tasks — "
        "'this will take at least six months just for the first phase'. "
        "Always anchor to specific tasks and specific durations or costs — vague statements do not "
        "convey the planning fallacy clearly."
    ),
    "In Group Bias": (
        "Express differential trust or favorability based on group membership. "
        "Positive direction: praise in-group members (same team, department, company) with warmth and trust, "
        "while implicitly raising doubts about out-group. "
        "Negative direction: express distrust, skepticism, or negative framing toward out-group members "
        "or teams, while favoring in-group unconditionally."
    ),
    "Bandwagon Effect": (
        "Appeal to peer behavior or majority opinion as a primary justification. "
        "Positive direction: note that 'everyone is moving this way' or 'all our competitors have adopted X' "
        "as reason to follow suit. "
        "Negative direction: note that 'nobody is doing this' or 'we'd be alone in this approach' "
        "as reason to avoid it. "
        "Do not give independent evidence — the crowd itself is the argument."
    ),
    "Status Quo Bias": (
        "Express a strong preference for the current situation and frame change as inherently risky. "
        "Positive direction: argue that the existing approach is proven and that 'if it ain't broke, don't fix it'. "
        "Negative direction: argue that the proposed change is dangerous and that any departure "
        "from the current path is reckless."
    ),
    "Sunk Cost Fallacy": (
        "Argue that prior investment (time, money, effort already spent) justifies continuing a course of action. "
        "Positive direction: encourage continuation by emphasising how much has already been invested. "
        "Negative direction: discourage abandoning a failing path by stressing that 'we've come too far to stop'. "
        "Do NOT acknowledge that past costs are irrelevant to future decisions."
    ),
    "Overconfidence Bias": (
        "Express certainty or precision in predictions, judgments, or abilities well beyond what evidence supports. "
        "Positive direction: claim near-certainty of success, give overly tight confidence intervals, "
        "dismiss uncertainty. "
        "Negative direction: claim near-certainty of failure, assert with false precision that bad outcomes "
        "are inevitable."
    ),
    "Representativeness Heuristic": (
        "Judge the situation or person by how well they match a typical prototype or stereotype, "
        "ignoring base rate statistics. "
        "Positive direction: note how closely the case matches a successful prototype. "
        "Negative direction: note how closely the case matches a failing prototype. "
        "Do not mention base rates or statistical context."
    ),
    "Authority Bias": (
        "Invoke an authority figure's opinion or credentials as the primary justification. "
        "Positive direction: cite an expert or respected leader who endorses the favorable option. "
        "Negative direction: cite an authority who raised doubts, used as reason to be cautious. "
        "Let the authority's opinion substitute for independent analysis."
    ),
    "Dunning-Kruger Effect": (
        "Express confidence that is miscalibrated relative to actual expertise. "
        "Positive direction: express high confidence in a domain despite limited knowledge, "
        "speak as if you are an expert even when you aren't. "
        "Negative direction: express excessive self-doubt and underestimate your own competence "
        "or the team's capability despite strong track record."
    ),
    "Halo Effect": (
        "Let one salient positive or negative trait color the overall evaluation. "
        "Positive direction: note one impressive quality and use it to imply overall excellence "
        "('they did X brilliantly, so the whole project should be great'). "
        "Negative direction: note one flaw and use it to cast doubt on everything else."
    ),
    "Hindsight Bias": (
        "Claim that the outcome was predictable or obvious in retrospect. "
        "Positive direction: say 'I always knew this would work out' or 'it was clearly the right call'. "
        "Negative direction: say 'everyone could see this was going to fail' or 'the warning signs were obvious'. "
        "Speak as if the outcome was inevitable and foreseeable."
    ),
    "Illusion of Control": (
        "Express belief that you or the team can control outcomes that are largely determined by chance. "
        "Positive direction: assert that with the right actions you can guarantee success, downplay randomness. "
        "Negative direction: assert that without specific interventions failure is certain, overstating controllability."
    ),
    "Information Bias": (
        "Argue that more information is needed before making a decision, even when additional data "
        "would not meaningfully change the outcome. "
        "Positive direction: advocate for more research to delay commitment. "
        "Negative direction: insist that acting without more information is reckless. "
        "Treat information-gathering as valuable in itself, regardless of its decision-relevance."
    ),
    "Negativity Bias": (
        "Give disproportionate weight to negative information, risks, or past failures. "
        "Negative direction (typical): focus on potential downsides, risks, and past failures while "
        "downplaying positive signals. "
        "Positive direction (reverse): excessively discount negatives and focus only on upsides."
    ),
    "Outcome Bias": (
        "Judge the quality of the decision-making process by the outcome, not the reasoning at the time. "
        "Positive direction: argue the approach was clearly correct because it led to good outcomes before. "
        "Negative direction: argue the approach was flawed because it once led to bad outcomes, "
        "regardless of whether the reasoning was sound at the time."
    ),
    "Reactance": (
        "Push back against a perceived pressure or constraint by advocating for the opposite. "
        "Express that being told to do X makes you want to do the opposite. "
        "Phrase it as defending autonomy or questioning why a particular path is being pushed."
    ),
    "Recency Bias": (
        "Give disproportionate weight to recent events, data, or trends over longer-term patterns. "
        "Positive direction: cite recent successes as representative of the new normal. "
        "Negative direction: cite recent failures or downturns as defining the trajectory. "
        "Reference time explicitly ('just recently', 'over the past few weeks') to signal recency."
    ),
    "Stereotype Bias": (
        "Apply a generalized group stereotype to characterize a specific individual, team, or situation. "
        "Positive direction: apply a favorable stereotype ('engineers from X are always rigorous'). "
        "Negative direction: apply an unfavorable stereotype to raise doubts. "
        "Do not acknowledge individual variation or counter-evidence."
    ),
    "Survivorship Bias": (
        "Focus only on the successful examples (survivors) while ignoring the failures that are not visible. "
        "Positive direction: cite well-known successes as proof the approach works, ignoring all the failures "
        "that used the same approach. "
        "Negative direction: cite visible failures to argue against an approach, ignoring the successes."
    ),
    "Gambler's Fallacy": (
        "Argue that a streak of past random outcomes makes a particular future outcome more or less likely. "
        "Positive direction: argue that after a run of failures, success is 'due' or likely. "
        "Negative direction: argue that after a run of successes, a failure is inevitable. "
        "Frame the streak as if it is causally relevant to the next outcome."
    ),
    "Omission Bias": (
        "Prefer inaction over action, treating doing nothing as inherently safer or more acceptable "
        "than an equivalent act of commission. "
        "Positive direction: argue that taking no action avoids moral responsibility for bad outcomes. "
        "Negative direction: warn strongly against any action because inaction feels safer."
    ),
    "Self-Serving Bias": (
        "Attribute outcomes in a self-serving way. "
        "Positive direction: attribute past successes to your own skill and effort while attributing "
        "failures to external factors (bad luck, others' mistakes). "
        "Negative direction: flip this — attribute your successes to luck and blame yourself or "
        "your team for failures beyond your control."
    ),
    "Base Rate Neglect": (
        "Focus on the specific details of this case while ignoring the relevant statistical base rate. "
        "Positive direction: emphasise unique positive features of this case to argue it will beat the odds. "
        "Negative direction: emphasise unique negative features to argue this case will underperform the base rate. "
        "Do not mention population statistics."
    ),
    "Decoy Effect": (
        "Introduce or reference an inferior comparison option to make your preferred choice look better. "
        "Positive direction: mention an obviously worse alternative to make the favored option appear superior. "
        "Negative direction: mention an obviously inferior version of the preferred option to make "
        "the status quo look better by comparison."
    ),
}

# One concrete example message per bias per direction.
# Grounded in a generic workplace performance-evaluation context so they
# transfer across the varied scenarios in the dataset.
_BIAS_EXAMPLES: dict[str, dict[str, str]] = {
    "Framing Effect": {
        "negative": "The way I see it, 1 in 4 of these initiatives ends in failure — that's a 25% failure rate we'd be accepting going in.",
        "positive": "Three out of four of these initiatives succeed — a 75% success rate is a strong foundation to move forward on.",
    },
    "Anchoring": {
        "negative": "Last cycle they were only hitting about 55% of their targets — keep that baseline in mind when you look at the current numbers.",
        "positive": "Their peak quarter had them at 95% efficiency — that's the kind of benchmark we should be measuring against.",
    },
    "Availability Heuristic": {
        "negative": "I keep thinking back to the Harper project collapse last month — it's hard not to see the same warning signs here.",
        "positive": "Remember the Apex launch last quarter? Everything clicked perfectly — that's exactly what this feels like to me.",
    },
    "Confirmation Bias": {
        "negative": "Every piece of feedback I've come across points to the same gaps — the evidence really does tell a consistent story.",
        "positive": "All the data I've reviewed keeps confirming the same strengths — it's hard to argue with that kind of consistency.",
    },
    "Loss Aversion": {
        "negative": "If this doesn't work out we stand to lose the credibility we've built over two years — that's not a risk I take lightly.",
        "positive": "Staying the course protects everything we've already invested — the downside of change just isn't worth it.",
    },
    "Optimism Bias": {
        "negative": "Honestly, I'd put the odds of this going as planned at maybe 20–30% — something almost always derails initiatives at this stage.",
        "positive": "I'd say we're looking at a 90% chance of hitting the target — the conditions are as good as they get and the chances of a real problem are minimal.",
    },
    "Planning Fallacy": {
        "negative": "Realistically, the first phase alone will take four to six months — every step will hit snags and the scope will grow.",
        "positive": "The core work is maybe three days, testing another two — I'd say we can have this done and shipped within two weeks comfortably.",
    },
    "In Group Bias": {
        "negative": "External vendors just don't have the institutional knowledge our own people have — I've rarely seen them deliver at the same level.",
        "positive": "Our team has always come through when it counts — there's a level of trust and reliability there you just don't get from outside.",
    },
    "Bandwagon Effect": {
        "negative": "Nobody in our peer group is still using this approach — if it were worth it, someone would still be doing it.",
        "positive": "Every major player in the space has already moved this direction — we'd essentially be the last ones to catch up.",
    },
    "Status Quo Bias": {
        "negative": "I just don't see why we'd change something that's been working fine — introducing uncertainty now seems unnecessary.",
        "positive": "The current setup has gotten us this far without major problems — there's real value in keeping what's proven.",
    },
    "Sunk Cost Fallacy": {
        "negative": "We've already put six months and significant budget into this — walking away now would mean all of that was for nothing.",
        "positive": "Given how much we've already committed to this path, it makes sense to see it through rather than start over.",
    },
    "Overconfidence Bias": {
        "negative": "I can say with near-certainty this isn't going to hit the target — I've seen enough of these to know how they end.",
        "positive": "I'm confident we'll nail this — there's really no scenario I can picture where this doesn't come together.",
    },
    "Representativeness Heuristic": {
        "negative": "This situation has all the hallmarks of the projects that end up going sideways — the pattern is pretty recognizable.",
        "positive": "This has every characteristic of our most successful launches — it fits the profile almost exactly.",
    },
    "Authority Bias": {
        "negative": "Even our most senior director flagged concerns about this approach — when someone at that level raises doubts, it's worth heeding.",
        "positive": "The head of our department was very positive about this direction — that kind of endorsement carries a lot of weight.",
    },
    "Dunning-Kruger Effect": {
        "negative": "Honestly, I think we're overestimating how well-equipped we are for this — there's a lot we probably don't know we don't know.",
        "positive": "I've looked at this thoroughly and I'm pretty sure I have a solid handle on what's needed — this isn't as complex as it sounds.",
    },
    "Halo Effect": {
        "negative": "After that missed deadline last spring, I find it hard to fully trust their overall reliability — one serious miss changes how I see everything.",
        "positive": "Their Q2 results were genuinely impressive — when a team performs at that level it tells you a lot about their overall capability.",
    },
    "Hindsight Bias": {
        "negative": "Looking back, it's pretty obvious this was heading in a difficult direction — the signals were there if you knew what to look for.",
        "positive": "It's clear now that this was always going to work — the fundamentals were solid from the start.",
    },
    "Illusion of Control": {
        "negative": "If we make the right interventions at the right time I think we can prevent the bad outcome — we have more control here than it might seem.",
        "positive": "With the right process in place we can essentially guarantee a good result — this isn't as unpredictable as it looks.",
    },
    "Information Bias": {
        "negative": "I don't think we should move forward until we have a fuller picture — there's still a lot we don't fully understand about this situation.",
        "positive": "We really should gather more data before committing — even a few extra weeks of analysis could change the picture significantly.",
    },
    "Negativity Bias": {
        "negative": "The few setbacks we've had really do outweigh the positives — it's hard to feel confident when those failures are still fresh.",
        "positive": "Even with a few bumps along the way, the overall trajectory has been positive — I don't think the setbacks define the picture.",
    },
    "Outcome Bias": {
        "negative": "The last time we took this kind of approach it didn't end well — that outcome tells me the decision-making was flawed.",
        "positive": "We used the same strategy two years ago and it worked out well — that success shows the approach is sound.",
    },
    "Reactance": {
        "negative": "I keep getting the sense that we're being pushed toward a particular answer here — which makes me want to think more carefully about the alternative.",
        "positive": "There seems to be a lot of pressure to go one way, but I think it's worth seriously considering whether that's actually right for us.",
    },
    "Recency Bias": {
        "negative": "The last few months have not been encouraging at all — that recent trend is hard to ignore when thinking about where this is headed.",
        "positive": "The last couple of quarters have been genuinely strong — that recent momentum is a real indicator of where things are going.",
    },
    "Stereotype Bias": {
        "negative": "Teams structured this way tend to struggle with execution under pressure — it's a pattern that shows up pretty consistently.",
        "positive": "Groups with this kind of profile tend to really deliver when it matters — there's a track record there across the industry.",
    },
    "Survivorship Bias": {
        "negative": "The success stories we keep hearing about are the exception — we don't talk enough about how many similar efforts quietly failed.",
        "positive": "The companies that have made this work are proof it's viable — if they could pull it off, there's no reason we can't.",
    },
    "Gambler's Fallacy": {
        "negative": "We've had a pretty good run lately — statistically, a correction is probably overdue at this point.",
        "positive": "After the rough patch we've been through, things should start turning around — it can't stay this way indefinitely.",
    },
    "Omission Bias": {
        "negative": "If we just hold off and don't intervene, at least we're not the ones responsible if something goes wrong — doing nothing keeps us out of that position.",
        "positive": "Taking no action at this stage seems like the safer call — any move we make could backfire, and inaction is at least neutral.",
    },
    "Self-Serving Bias": {
        "negative": "The shortfall this quarter was really a result of the market conditions and the support we didn't get — not a reflection of the team's effort.",
        "positive": "A lot of what drove those strong results came down to the team's execution and the decisions we made early on.",
    },
    "Base Rate Neglect": {
        "negative": "The specific issues in this case make it different from the average — I wouldn't rely on the general success rate as a guide here.",
        "positive": "This particular situation has so many advantages over the typical case — the base rate really undersells the odds here.",
    },
    "Decoy Effect": {
        "negative": "Compared to the alternative we reviewed last week, the current option actually looks quite reasonable — the contrast makes it much more attractive.",
        "positive": "When you put the other option next to this one, it becomes obvious which is the stronger choice — the comparison really clarifies things.",
    },
}

_SIMULATOR_PROMPT = """\
You are simulating a human professional in a workplace conversation about a decision-making scenario.

Situation context:
{situation}

Bias to express: {bias_name}
Mechanism — how this bias should appear in your message:
{mechanism}

Example message ({direction} direction):
"{example}"
{feedback}
Before writing, silently identify: (a) the ONE specific phrase that will carry the bias, \
and (b) exactly how you will apply the mechanism at {intensity_label} intensity ({intensity_value:.1f}/1.0).

Write ONE natural conversational message (2–4 sentences) that:
1. Continues naturally as a colleague or manager
2. Applies the mechanism in the {direction} direction
3. Does NOT ask the upcoming decision question — only share your biased perspective
4. Does NOT explicitly name or announce the bias

Output ONLY the message text. No preamble, labels, or planning notes."""


class HumanSimulator:
    """
    Generates biased conversational messages simulating a human interlocutor.

    Implements the human-bias intervention operator I^x_gamma from the conditional
    bias framework: given a scenario x, produces a conversational context c_gamma in
    which the human input exhibits bias gamma according to parameters (gamma, delta, s, kappa).
    """

    def __init__(self):
        self._client = OpenAI()

    def generate(
        self,
        situation: str,
        bias_name: str,
        direction: str = "positive",
        intensity: float = 0.7,
        seed: int = 42,
        max_retries: int = 3,
        feedback: str = "",
    ) -> str:
        """
        Generate a biased conversational message for the given situation.

        Args:
            situation: The shared scenario context (common situations from the test case).
            bias_name: The cognitive bias to express (e.g., "Framing Effect").
            direction: Directional orientation of the bias ("positive" or "negative").
            intensity: Bias intensity in [0, 1]; mapped to low / moderate / high labels.
            seed: Seed for reproducibility; incremented on each retry.
            max_retries: Number of API-call retries on failure.
            feedback: Actionable rejection feedback from the jury on a previous attempt.
                      When non-empty, prepended to the prompt to guide the revision.

        Returns:
            A conversational message exhibiting the specified bias.
        """
        mechanism = _BIAS_MECHANISMS.get(
            bias_name,
            f"Express {bias_name} clearly through your word choice and framing.",
        )
        examples = _BIAS_EXAMPLES.get(bias_name, {})
        example = examples.get(direction) or next(iter(examples.values()), "")

        feedback_block = (
            f"\nFEEDBACK FROM PREVIOUS ATTEMPT (rejected by expert reviewers):\n"
            f"{feedback}\n"
            if feedback else ""
        )

        prompt = _SIMULATOR_PROMPT.format(
            situation=situation,
            bias_name=bias_name,
            mechanism=mechanism,
            example=example,
            feedback=feedback_block,
            intensity_label=self._intensity_label(intensity),
            intensity_value=intensity,
            direction=direction,
        )
        last_exc = None
        for attempt in range(max_retries):
            try:
                response = self._client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0.7,
                    seed=seed + attempt,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.choices[0].message.content
            except Exception as e:
                last_exc = e
        raise last_exc

    def generate_neutral(
        self,
        situation: str,
        seed: int = 42,
        max_retries: int = 3,
        jury=None,
    ) -> tuple:
        """
        Generate a neutral, unbiased message for use as a baseline condition.

        The message engages meaningfully with the situation without expressing any
        cognitive bias, preference, or judgment.

        Args:
            situation: The shared scenario context.
            seed: Seed for reproducibility.
            max_retries: Number of generation + validation retries.
            jury: Optional BiasJury instance. When provided, validates each
                  candidate with evaluate_neutral() and retries if any bias
                  rating >= 2.0 or naturalness < 3.0.

        Returns:
            Tuple of (message, jury_result, jury_passed).
            jury_result is None and jury_passed is True when no jury is provided.
        """
        prompt = (
            "You are a human professional participating in a workplace conversation.\n\n"
            f"Situation context:\n{situation}\n\n"
            "Write a SHORT, NEUTRAL message (2–4 sentences) that a colleague might say "
            "in response to this situation. The message must:\n"
            "- Engage meaningfully with the situation rather than simply restating it\n"
            "- Be factual and balanced, with no cognitive bias, preference, or judgment\n"
            "- Sound like a natural, professional comment a reasonable person would make\n"
            "- Avoid framing effects, anchoring numbers, appeals to authority or majority "
            "opinion, overconfidence, or loss aversion language\n\n"
            "Respond with only the message text, no preamble."
        )
        # Temperature escalates across retries to diversify candidates
        _temperatures = [0.3, 0.5, 0.7]

        last_message: str | None = None
        last_jury_result: dict | None = None
        feedback_suffix = ""
        for attempt in range(max(max_retries, 1)):
            temperature = _temperatures[min(attempt, len(_temperatures) - 1)]
            full_prompt = prompt + feedback_suffix
            try:
                response = self._client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=temperature,
                    seed=seed + attempt,
                    messages=[{"role": "user", "content": full_prompt}],
                )
                candidate = response.choices[0].message.content
            except Exception:
                continue

            if jury is None:
                return candidate, None, True

            evaluation = jury.evaluate_neutral(candidate, situation, seed=seed + attempt)
            last_message, last_jury_result = candidate, evaluation
            if jury.is_neutral_valid(evaluation):
                return candidate, evaluation, True

            # Build specific feedback for the next attempt
            top_bias = evaluation.get("top_bias", "")
            max_rating = evaluation.get("max_bias_rating", 0.0)
            naturalness = evaluation.get("naturalness", 0.0)
            parts = []
            if top_bias and max_rating >= 2.0:
                parts.append(
                    f"Your previous message scored {max_rating:.1f}/5 on {top_bias}. "
                    f"Rewrite to remove all {top_bias}-style language and phrasing."
                )
            if naturalness < 3.0:
                parts.append(
                    "Your previous message scored low on naturalness. "
                    "Make it sound more like something a real colleague would say."
                )
            if parts:
                feedback_suffix = "\n\nFEEDBACK ON PREVIOUS ATTEMPT: " + " ".join(parts)

        # All retries exhausted — return best candidate with failed status
        return last_message or "", last_jury_result, False

    @staticmethod
    def _intensity_label(intensity: float) -> str:
        if intensity >= 0.8:
            return "high"
        elif intensity >= 0.5:
            return "moderate"
        return "low"
