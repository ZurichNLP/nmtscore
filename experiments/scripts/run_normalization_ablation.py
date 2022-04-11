import logging
import sys
from pathlib import Path

sys.path.insert(1, str(Path(__file__).resolve().parent.parent.parent))
from experiments.paraphrase_benchmark import ParaphraseIdentificationBenchmark
from experiments.metrics import benchmark_metrics

logging.basicConfig(level=logging.INFO)

benchmark = ParaphraseIdentificationBenchmark()
results_path = benchmark.default_results_path.with_suffix(".normalization.pickle")

RECOMPUTE_RESULTS = True

if RECOMPUTE_RESULTS:
    benchmark.load_tasks()
    benchmark.run_metrics(benchmark_metrics.get_normalization_ablation_metrics(device=0))
    benchmark.save_results(results_path)

benchmark.load_results(results_path)

benchmark.num_iterations = 1
task_results = benchmark.get_confidence_intervals_for_individual_tasks()
pawsx_avg_results = benchmark.get_confidence_intervals_for_pawsx_average()
overall_results = benchmark.get_confidence_intervals_for_overall_average()
for metric in task_results:
    print(metric, end="\t")
    for task, result in task_results[metric].items():
        print(result.observed, end="\t")
    print(pawsx_avg_results[metric].observed, end="\t")
    print(overall_results[metric].observed)

benchmark.num_iterations = 1000
for task_name in [
    "mrpc",
    "russian",
    "finnish",
    "swedish",
    "pawsx-de",
    "pawsx-es",
    "pawsx-fr",
    "pawsx-ja",
    "pawsx-zh",
    "pawsx_average",
    "overall_average",
]:
    benchmark.pairwise_diff_tests(task_name)
