import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(1, str(Path(__file__).resolve().parent.parent.parent))
from experiments.webnlg import WebNLGBenchmark
from experiments.metrics import benchmark_metrics

RECOMPUTE_RESULTS = True
CORRELATION = "kendall"
CRITERION = "overall_adequacy"
language = sys.argv[1]
level = sys.argv[2]

benchmark = WebNLGBenchmark(num_iterations=1000, language=language, criterion=CRITERION)

if RECOMPUTE_RESULTS:
    benchmark.run_metrics(benchmark_metrics.get_nlg_evaluation_metrics(device=0))
    benchmark.save_results()

benchmark.load_results()

benchmark.pairwise_diff_tests(
    level=level,
    correlation=CORRELATION,
)

ax = benchmark.plot_confidence_intervals(
    level=level,
    correlation=CORRELATION,
)
results_dir = Path(__file__).parent.parent / "results"
plt.savefig(str(results_dir / f"webnlg_{language}_{level}_{CRITERION}_{CORRELATION}.svg"))
