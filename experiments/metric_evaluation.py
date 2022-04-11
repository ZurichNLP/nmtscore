import functools
import itertools
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
from matplotlib import pyplot as plt
from sacrerouge.stats import global_corr, system_level_corr, \
    summary_level_corr
from scipy.stats import kendalltau, spearmanr, pearsonr
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

PValue = float


@dataclass
class ConfidenceInterval:
    lower: float
    observed: float
    upper: float


@dataclass
class DiffTestResult:
    pvalue: float
    delta: float


@dataclass
class ResamplingResult:
    resampled_statistics: List[float]
    test_statistic: float


class MetricBenchmark:

    def __init__(self, num_iterations=1000):
        self.num_iterations = num_iterations

    def __str__(self):
        raise NotImplementedError

    def _accuracy(self, X: np.ndarray, Y: np.ndarray) -> Optional[float]:
        if len(np.unique(Y)) < 2:
            return None
        return accuracy_score(Y.squeeze(), X.squeeze())

    def _roc_auc(self, X: np.ndarray, Y: np.ndarray) -> Optional[float]:
        if len(np.unique(Y)) < 2:
            return None
        return roc_auc_score(Y.squeeze(), X.squeeze())

    @property
    def results_dir(self) -> Path:
        return Path(__file__).parent / "results"

    @property
    def default_results_path(self) -> Path:
        return self.results_dir / f"{self}.pickle"

    def save_results(self, out_path: Path = None):
        out_path = out_path or self.default_results_path
        with open(out_path, "wb") as f:
            pickle.dump(self.results, f)

    def load_results(self, results_path: Path = None):
        results_path = results_path or self.default_results_path
        with open(results_path, "rb") as f:
            self.results = pickle.load(f)

    def _parse_metric_name(self, results_title: str) -> str:
        metric_name = results_title
        while not metric_name.islower():
            metric_name = metric_name.split("_", 1)[-1]
        return metric_name

    def _correlation(self, X: np.ndarray, Y: np.ndarray, correlation_function: str, level: str = "global") -> float:
        if correlation_function == "pearson":
            corr_func = pearsonr
        elif correlation_function == "spearman":
            corr_func = spearmanr
        elif correlation_function == "kendall":
            corr_func = kendalltau
        else:
            raise ValueError
        if level == "summary_level":
            level_corr_func = functools.partial(summary_level_corr, corr_func)
        elif level == "system_level":
            level_corr_func = functools.partial(system_level_corr, corr_func)
        elif level == "global":
            level_corr_func = functools.partial(global_corr, corr_func)
        else:
            raise ValueError
        correlation = level_corr_func(X, Y)
        return correlation

    def pairwise_diff_tests(self, level: str = "global", correlation: str = "kendall", metric_names=None) -> None:
        assert level in self.supported_levels
        assert self.results
        used_metric_names = metric_names or list(self.results)
        diff_results: Dict[Tuple[str, str]: float] = dict()
        for metric_name1, metric_name2 in tqdm(list(itertools.product(used_metric_names, used_metric_names))):
            result = self.diff_test(metric_name1, metric_name2, level=level, correlation=correlation)
            diff_results[(metric_name1, metric_name2)] = result

        outperform_counter = Counter()
        for (metric_name1, metric_name2), diff_result in diff_results.items():
            outperform_counter[metric_name1] += diff_result.delta > 0
        ordered_metric_names = [metric_name for metric_name, _ in outperform_counter.most_common()]

        print("## p-values")
        print("\t" + "\t".join(ordered_metric_names))
        for metric_name1 in ordered_metric_names:  # rows
            print(metric_name1, end="\t")
            for metric_name2 in ordered_metric_names:  # columns
                diff_result = diff_results[(metric_name1, metric_name2)]
                print(diff_result.pvalue, end="\t")
            print()
        print()
        print("## deltas")
        print("\t" + "\t".join(ordered_metric_names))
        for metric_name1 in ordered_metric_names:  # rows
            print(metric_name1, end="\t")
            for metric_name2 in ordered_metric_names:  # columns
                diff_result = diff_results[(metric_name1, metric_name2)]
                print(diff_result.delta, end="\t")
            print()
        print()

    def plot_confidence_intervals(self, level: str = "global", correlation: str = "kendall", metric_names=None, color="#ff7f0e"):
        """
        Adapted from https://github.com/CogComp/stat-analysis-experiments/blob/master/experiments/statistical-analysis/confidence-intervals/plot.py
        """
        assert level in self.supported_levels
        fig, ax = plt.subplots(1, 1, figsize=(4.51431, 4.68780))
        resampling_results: List[ResamplingResult] = []
        positions = []
        ticks = []
        used_metric_names = metric_names or list(self.results)
        for i, metric in enumerate(used_metric_names):
            resampling_results.append(self.get_resampling_result_for_metric(metric, level=level, correlation=correlation))
            positions.append(len(used_metric_names) - (i + 1))
            ticks.append(len(used_metric_names) - (i + 1))
        parts = ax.violinplot([result.resampled_statistics for result in resampling_results], positions=positions, vert=False)

        for i, pc in enumerate(parts['bodies']):
            pc.set_color(color)
        parts['cbars'].set_color([color])
        parts['cmaxes'].set_color([color])
        parts['cmins'].set_color([color])

        ax.vlines(
            [result.test_statistic for result in resampling_results],
            *([[-0.25], [0.25]] * (np.array([0.5] * len(resampling_results))) + positions),
            colors="black",
            linewidth=2,
        )
        ax.set_xlim([0, 1])
        ax.set_title(str(self))
        ax.set_yticks(ticks)
        ax.set_yticklabels(used_metric_names)
        fig.add_subplot(111, frame_on=False)
        plt.tick_params(labelcolor="none", bottom=False, left=False)
        plt.xlabel(f'{correlation.title()} Correlation Coefficient')
        # plt.tight_layout()
        # Border thickness
        [x.set_linewidth(1.5) for x in ax.spines.values()]
        return ax
