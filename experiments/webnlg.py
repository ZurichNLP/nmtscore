import functools
import os
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import List, Dict

import numpy as np
from sacrerouge.commands.correlate import load_metrics, merge_metrics, filter_metrics
from sacrerouge.commands.score import score_instances, save_score_results
from sacrerouge.commands.stat_sig_test import run_hypothesis_tests
from sacrerouge.data.dataset_readers import ReferenceBasedDatasetReader
from sacrerouge.stats import corr_ci, convert_to_matrices

from experiments.metric_evaluation import MetricBenchmark, ResamplingResult, DiffTestResult
from experiments.metrics.benchmark_metrics import BenchmarkMetric


class WebNLGBenchmark(MetricBenchmark):

    supported_levels = {"system_level", "global"}

    def __init__(self, num_iterations=1000, language: str = "en", criterion="overall_adequacy"):
        super().__init__(num_iterations)
        assert language in {"en", "ru"}
        assert criterion in {
            "overall_adequacy",
            "overall_fluency",
            "correctness",
            "data_coverage",
            "relevance",
            "fluency",
            "text_structure",
        }
        self.language = language
        self.data_dir = Path(__file__).parent / "data" / "webnlg2020"
        assert self.data_dir.exists()
        self.input_filepath = self.data_dir / f"webnlg2020.{self.language}.generated.jsonl"
        self.reference_metric_path = self.data_dir / f"webnlg2020.{self.language}.metrics.jsonl"
        self.reference_metric_name = f"human_{criterion}"
        self.results: Dict[str, dict] = OrderedDict()  # metric -> SacreRouge score results

    def __str__(self):
        return f"webnlg2020-{self.language}_benchmark"

    def run_metrics(self, benchmark_metrics: List[BenchmarkMetric]):
        for benchmark_metric in benchmark_metrics:
            metric = benchmark_metric.load_metric(a_lang=self.language, b_lang=self.language)
            instances = ReferenceBasedDatasetReader().read(str(self.input_filepath))
            results = score_instances(instances, [metric])
            for metric_name in benchmark_metric.metric_names:
                self.results[f"{benchmark_metric.title}_{metric_name}"] = results

    def get_resampling_result_for_metric(self, metric_name: str, level: str = "global", correlation: str = "kendall") -> ResamplingResult:
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
        corr_func = functools.partial(self._correlation, correlation_function=correlation, level=level)
        # noinspection PyTupleAssignmentBalance
        lower, upper, sample_correlations = corr_ci(
            corr_func,
            X,
            Y,
            method="bootstrap-both",
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

    def diff_test(self, better_metric_name: str, worse_metric_name: str, level: str = "global", correlation: str = "kendall") -> DiffTestResult:
        assert level in self.supported_levels
        better_score_result = self.results[better_metric_name]
        worse_score_result = self.results[worse_metric_name]
        with tempfile.NamedTemporaryFile(delete=False) as f_better:
            save_score_results(better_score_result, f_better.name, silent=False)
        with tempfile.NamedTemporaryFile(delete=False) as f_worse:
            save_score_results(worse_score_result, f_worse.name, silent=False)
        results = run_hypothesis_tests(
            metrics_jsonl_files_or_metrics_list=[str(self.reference_metric_path), f_better.name, f_worse.name],
            dependent_metric=self.reference_metric_name,
            metric_A=self._parse_metric_name(better_metric_name),
            metric_B=self._parse_metric_name(worse_metric_name),
            summarizer_type="peer",
            test_method="bootstrap-both",
            two_tailed=False,
            skip_summary_level=level != "summary_level",
            skip_system_level=level != "system_level",
            skip_global=level != "global",
        )
        os.remove(f_better.name)
        os.remove(f_worse.name)
        return DiffTestResult(
            pvalue=results[level][correlation]["pvalue"],
            delta=results[level][correlation]["test_statistic"],
        )

    def print_statistics(self) -> None:
        samples = ReferenceBasedDatasetReader().read(str(self.input_filepath))
        print("Number of samples:", len(samples))
        print("Number of documents:", len({(sample.instance_id) for sample in samples}))
        print("Number of systems:", len({sample.summarizer_id for sample in samples}))
        references = []
        reference_numbers = []
        for sample in samples:
            references += sample.fields["references"].references
            reference_numbers.append(len(sample.fields["references"].references))
        print("Average number of references", np.mean(reference_numbers))
        print("Average number of reference characters", np.mean([len(reference) for reference in references]))
