import sys
from pathlib import Path

sys.path.insert(1, str(Path(__file__).resolve().parent.parent.parent))
from experiments.paraphrase_benchmark import ParaphraseIdentificationBenchmark

benchmark = ParaphraseIdentificationBenchmark(num_iterations=1000)

benchmark.load_results()

pairwise_correlations = benchmark.get_confidence_intervals_for_pairwise_correlation()
print("\t" + "\t".join(benchmark.results))
for metric1 in benchmark.results:
    print(metric1, end="\t")
    for metric2 in benchmark.results:
        confidence_interval = pairwise_correlations.get((metric1, metric2), pairwise_correlations.get((metric2, metric1), None))
        if confidence_interval is None:
            print("", end="\t")
            continue
        max_ci = max(confidence_interval.observed - confidence_interval.lower, confidence_interval.upper - confidence_interval.observed)
        print(f"{confidence_interval.observed:.2f}±{max_ci:.2f}", end="\t")
    print()
