import logging
import sys
from pathlib import Path

sys.path.insert(1, str(Path(__file__).resolve().parent.parent.parent))
from experiments.crosslingual_pawsx import CrossLingualPAWSXBenchmark
from experiments.metrics import benchmark_metrics

logging.basicConfig(level=logging.INFO)

benchmark = CrossLingualPAWSXBenchmark()

RECOMPUTE_RESULTS = True

if RECOMPUTE_RESULTS:
    benchmark.load_tasks()
    benchmark.run_metrics(benchmark_metrics.get_paraphrase_metrics(device=0))
    benchmark.save_results()

benchmark.load_results()

benchmark.num_iterations = 1
task_results = benchmark.get_confidence_intervals_for_individual_tasks()
pawsx_avg_results = benchmark.get_confidence_intervals_for_pawsx_average()
for i, metric in enumerate(task_results):
    if i == 0:
        print("\t" + "\t".join(task_results[metric].keys()))
    print(metric, end="\t")
    for task, result in task_results[metric].items():
        print(result.observed, end="\t")
    print(pawsx_avg_results[metric].observed)

benchmark.num_iterations = 1000
for task_name in [
    "pawsx-en-es",
    "pawsx-en-fr",
    "pawsx-en-ja",
    "pawsx-en-zh",
    "pawsx-de-en",
    "pawsx-de-es",
    "pawsx-de-fr",
    "pawsx-de-ja",
    "pawsx-de-zh",
    "pawsx-es-fr",
    "pawsx-es-ja",
    "pawsx-es-zh",
    "pawsx-fr-ja",
    "pawsx-fr-zh",
    "pawsx-ja-zh",
    "pawsx_average",
]:
    benchmark.pairwise_diff_tests(task_name)
