import functools
import os
import tempfile
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from sacrerouge.commands.correlate import load_metrics, merge_metrics, filter_metrics
from sacrerouge.commands.score import score_instances, save_score_results
from sacrerouge.data import Metrics
from sacrerouge.data.dataset_readers import ReferenceBasedDatasetReader
from sacrerouge.stats import corr_ci, convert_to_matrices, corr_diff_test

from experiments.metric_evaluation import MetricBenchmark, ResamplingResult, DiffTestResult
from experiments.metrics.benchmark_metrics import BenchmarkMetric


class MultilingualAMRBenchmark(MetricBenchmark):

    supported_levels = {"global"}

    languages = OrderedDict({
        # "en": "English",
        "da": "Danish",
        # "de": "German",
        "el": "Greek",
        "es": "Spanish",
        "fi": "Finnish",
        # "fr": "French",
        "it": "Italian",
        "nl": "Dutch",
        "pt": "Portuguese",
        "sv": "Swedish",
        "bg": "Bulgarian",
        "cs": "Czech",
        "et": "Estonian",
        "hu": "Hungarian",
        "lv": "Latvian",
        "pl": "Polish",
        "ro": "Romanian",
    })

    def __init__(self, num_iterations=1000, criterion="semantic_accuracy"):
        super().__init__(num_iterations)
        self.data_dir = Path(__file__).parent / "data" / "multilingual_amr"
        self.reference_metric_path = self.prepare_reference_metrics()
        self.reference_metric_name = f"human_{criterion}"
        self.results: Dict[str, Dict[str, dict]] = OrderedDict()  # metric title -> language -> SacreRouge score results

    def __str__(self):
        return "multilingual_amr_benchmark"

    def prepare_reference_metrics(self) -> Path:
        all_reference_metrics: List[Metrics] = []
        current_instance_id = 0
        for language in self.languages:
            reference_metrics_path = self.data_dir / f"amr.{language}.metrics.jsonl"
            reference_metrics = load_metrics([str(reference_metrics_path)])
            for metric in reference_metrics:
                metric.instance_id = current_instance_id
                metric.summarizer_id = language
                all_reference_metrics.append(metric)
                current_instance_id += 1
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            for metric in all_reference_metrics:
                f.write(repr(metric) + "\n")
        return Path(f.name)

    def run_metrics(self, benchmark_metrics: List[BenchmarkMetric]):
        for benchmark_metric in benchmark_metrics:
            all_results = dict()
            current_instance_id = 0
            for language in self.languages:
                metric = benchmark_metric.load_metric(a_lang=language, b_lang=language)
                input_filepath = self.data_dir / f"amr.{language}.generated.jsonl"
                instances = ReferenceBasedDatasetReader().read(str(input_filepath))
                score_results = score_instances(instances, [metric])

                # Re-assign instance IDs:
                #   Every sample in every language should be its own document
                #   Language should be treated as systems
                reassigned_results = defaultdict(dict)
                for old_instance_id in score_results.keys():
                    for old_summarizer_id in score_results[old_instance_id].keys():
                        sample_result = score_results[old_instance_id][old_summarizer_id]
                        sample_result.instance_id = current_instance_id
                        sample_result.summarizer_id = language
                        reassigned_results[current_instance_id][language] = sample_result
                        current_instance_id += 1
                all_results.update(reassigned_results)

            for metric_name in benchmark_metric.metric_names:
                if f"{benchmark_metric.title}_{metric_name}" not in self.results:
                    self.results[f"{benchmark_metric.title}_{metric_name}"] = dict()
                self.results[f"{benchmark_metric.title}_{metric_name}"] = all_results

    def _average_correlation(self, X: np.ndarray, Y: np.ndarray, correlation_function: str, level: str) -> Optional[float]:
        """
        Averages global correlation per language / "system"
        """
        correlations = []
        for i in range(X.shape[0]):
            task_X = X[i, :]
            task_Y = Y[i, :]
            task_X = task_X[~np.isnan(task_X)]
            task_Y = task_Y[~np.isnan(task_Y)]
            if not task_X.size:
                return None
            correlation = self._correlation(task_Y, task_X, correlation_function, level)
            if correlation is None:
                continue
            correlations.append(correlation)
        return np.mean(correlations).item()

    def get_resampling_result_for_metric(self, metric_name: str, level: str = "global", correlation: str = "pearson") -> ResamplingResult:
        assert level in self.supported_levels
        score_results = self.results[metric_name]
        with tempfile.NamedTemporaryFile(delete=False) as f:
            save_score_results(score_results, f.name, silent=False)
        metrics_list = load_metrics([str(self.reference_metric_path), f.name])
        os.remove(f.name)
        metrics_list = merge_metrics(metrics_list)
        for metrics in metrics_list:
            metrics.flatten_keys()
        metric1 = self.reference_metric_name
        metric2 = self._parse_metric_name(metric_name)
        metrics_list = filter_metrics(metrics_list, "peer", metric1, metric2)
        for metrics in metrics_list:
            metrics.select_metrics([metric1, metric2])
            metrics.average_values()
        X, Y = convert_to_matrices(metrics_list, metric1, metric2)
        corr_func = functools.partial(self._average_correlation, correlation_function=correlation, level=level)
        # noinspection PyTupleAssignmentBalance
        lower, upper, sample_correlations = corr_ci(
            corr_func,
            X,
            Y,
            method="bootstrap-input",
            kwargs={
                "return_sample_correlations": True,
                "num_samples": self.num_iterations,
            },
        )
        test_statistic = corr_func(X, Y)
        return ResamplingResult(
            test_statistic=test_statistic,
            resampled_statistics=sample_correlations,
        )

    def diff_test(self, better_metric_name: str, worse_metric_name: str, level: str = "global", correlation: str = "pearson") -> DiffTestResult:
        """
        Use average correlation across languages
        """
        assert level in self.supported_levels
        better_score_result = self.results[better_metric_name]
        worse_score_result = self.results[worse_metric_name]
        with tempfile.NamedTemporaryFile(delete=False) as f_better:
            save_score_results(better_score_result, f_better.name, silent=False)
        with tempfile.NamedTemporaryFile(delete=False) as f_worse:
            save_score_results(worse_score_result, f_worse.name, silent=False)
        metrics_list = load_metrics([str(self.reference_metric_path), f_better.name, f_worse.name])
        os.remove(f_better.name)
        os.remove(f_worse.name)
        metrics_list = merge_metrics(metrics_list)
        for metrics in metrics_list:
            metrics.flatten_keys()
        dependent_metric = self.reference_metric_name
        metric_A = self._parse_metric_name(better_metric_name)
        metric_B = self._parse_metric_name(worse_metric_name)
        metrics_list = filter_metrics(metrics_list, "peer", dependent_metric, metric_A, metric_B)
        for metrics in metrics_list:
            metrics.select_metrics([dependent_metric, metric_A, metric_B])
            metrics.average_values()
        X, Y, Z = convert_to_matrices(metrics_list, metric_A, metric_B, dependent_metric)
        corr_func = functools.partial(self._average_correlation, correlation_function=correlation, level=level)
        pvalue, delta = corr_diff_test(
            corr_func=corr_func,
            X=X,
            Y=Y,
            Z=Z,
            method="bootstrap-input",
            two_tailed=False,
            kwargs={
                "num_samples": self.num_iterations,
                "return_test_statistic": True,
            },
        )
        return DiffTestResult(
            pvalue=pvalue,
            delta=delta,
        )

    def print_statistics(self) -> None:
        for language in self.languages:
            print("Language", language)
            input_filepath = self.data_dir / f"amr.{language}.generated.jsonl"
            samples = ReferenceBasedDatasetReader().read(str(input_filepath))
            print("Number of samples:", len(samples))
            references = []
            reference_numbers = []
            for sample in samples:
                references += sample.fields["references"].references
            reference_numbers.append(len(sample.fields["references"].references))
            print("Average number of references", np.mean(reference_numbers))
            print("Average number of reference characters", np.mean([len(reference) for reference in references]))
            print()
