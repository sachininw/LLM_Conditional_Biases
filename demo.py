from core.utils import get_generator, get_metric, get_model
import random


BIAS = 'Anchoring'               # any bias name in PascalCase
TEMPERATURE_GENERATION = 0.7
TEMPERATURE_DECISION   = 0.0
RANDOMLY_FLIP_OPTIONS  = True
SHUFFLE_ANSWER_OPTIONS = False


if __name__ == "__main__":

    with open('data/scenarios.txt') as f:
        scenarios = f.readlines()

    scenario = random.choice(scenarios)
    seed     = random.randint(0, 1000)

    generator = get_generator(BIAS)
    metric    = get_metric(BIAS)

    generation_model = get_model("GPT-4o")
    decision_model   = get_model("GPT-4o", RANDOMLY_FLIP_OPTIONS, SHUFFLE_ANSWER_OPTIONS)

    test_cases       = generator.generate_all(generation_model, [scenario], TEMPERATURE_GENERATION, seed, num_instances=1, max_retries=5)
    print(test_cases)

    decision_results = decision_model.decide_all(test_cases, TEMPERATURE_DECISION, seed)
    print(decision_results)

    metric          = metric(test_results=list(zip(test_cases, decision_results)))
    computed_metric = metric.compute()
    print(f'Bias metric per case:\n{computed_metric}')
    print(f'Aggregated bias metric: {metric.aggregate(computed_metric)}')