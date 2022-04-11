import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(1, str(Path(__file__).resolve().parent.parent.parent))
from experiments.multilingual_amr import MultilingualAMRBenchmark
from experiments.metrics import benchmark_metrics

RECOMPUTE_RESULTS = True
CORRELATION = "kendall"
LEVEL = "global"

benchmark = MultilingualAMRBenchmark(num_iterations=1000)

if RECOMPUTE_RESULTS:
    benchmark.run_metrics(benchmark_metrics.get_nlg_evaluation_metrics(device=0))
    benchmark.save_results()

benchmark.load_results()

benchmark.pairwise_diff_tests(
    level=LEVEL,
    correlation=CORRELATION,
)

ax = benchmark.plot_confidence_intervals(
    level=LEVEL,
    correlation=CORRELATION,
    color="#1f77b4",
)
results_dir = Path(__file__).parent.parent / "results"
plt.savefig(str(results_dir / f"amr_{LEVEL}_{CORRELATION}.svg"))
