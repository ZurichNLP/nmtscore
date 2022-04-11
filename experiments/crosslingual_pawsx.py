import logging
from collections import OrderedDict, defaultdict
from typing import List, Dict

import numpy as np
from sacrerouge.data import MetricsDict
from sacrerouge.metrics import Metric
from sklearn.metrics import roc_curve, accuracy_score

from experiments.metric_evaluation import ConfidenceInterval, DiffTestResult, MetricBenchmark
from experiments.metrics.benchmark_metrics import BenchmarkMetric
from experiments.paraphrase_benchmark import ParaphraseIdentificationBenchmark
from experiments.paraphrase_tasks import ParaphraseClassificationTask, PAWSXParaphraseClassificationTask, PAWSXSample, \
    ParaphraseClassificationTaskWithThresholding, ParaphraseClassificationResult


class CrossLingualPAWSXTask(ParaphraseClassificationTask):
    def __init__(self, language1: str, language2: str, split="validation"):
        self.language1 = language1
        self.language2 = language2
        self.split = split
        self.dataset1 = PAWSXParaphraseClassificationTask(language1, split)
        self.dataset2 = PAWSXParaphraseClassificationTask(language2, split)

    def __str__(self):
        return f"pawsx_{self.language1}_{self.language2}_{self.split}"

    def get_samples(self) -> List[PAWSXSample]:
        samples1 = self.dataset1.get_samples()
        samples2 = self.dataset2.get_samples()
        sample_dict1 = {sample.sample_id: sample for sample in samples1}
        sample_dict2 = {sample.sample_id: sample for sample in samples2}
        forward_samples = []
        backward_samples = []
        for sample_id, sample1 in sample_dict1.items():
            if sample_id not in sample_dict2:
                continue
            sample2 = sample_dict2[sample_id]
            if sample1.label != sample2.label:
                continue
            forward_samples.append(PAWSXSample(
                sentence1=sample1.sentence1,
                sentence2=sample2.sentence2,
                label=sample1.label,
                sample_id=sample_id,
            ))
            backward_samples.append(PAWSXSample(
                sentence1=sample1.sentence2,
                sentence2=sample2.sentence1,
                label=sample1.label,
                sample_id=sample_id,
            ))
        return forward_samples + backward_samples


class CrossLingualPAWSXBenchmark(ParaphraseIdentificationBenchmark):

    def __init__(self, num_iterations=1000):
        super().__init__(num_iterations, include_pawsx=True)
        self.pawsx_languages = [
            'en',
            'de',
            'es',
            'fr',
            'ja',
            # 'ko',
            'zh',
        ]

    def __str__(self):
        return "crosslingual_pawsx_benchmark"

    def load_tasks(self):
        self.tasks = OrderedDict()
        for language1 in self.pawsx_languages:
            for language2 in self.pawsx_languages:
                if language1 >= language2:
                    continue
                self.tasks[(language1, language2)] = ParaphraseClassificationTaskWithThresholding(
                    validation_task=CrossLingualPAWSXTask(language1, language2, "validation"),
                    evaluation_task=CrossLingualPAWSXTask(language1, language2, "test"),
                )

    def run_metrics(self, benchmark_metrics: List[BenchmarkMetric]):
        for benchmark_metric in benchmark_metrics:
            for (language1, language2), task in self.tasks.items():
                logging.info(f"Running {benchmark_metric.title} on PAWS-X {language1.upper()}–{language2.upper()}")
                metric = benchmark_metric.load_metric(a_lang=language1, b_lang=language2)
                results = task.evaluate(metric, benchmark_metric.metric_names)
                for metric_name, result in zip(benchmark_metric.metric_names, results):
                    self.results[f"{benchmark_metric.title}_{metric_name}"][f"pawsx-{language1}-{language2}"] = result
                del metric

    def get_confidence_intervals_for_average(self) -> Dict[str, ConfidenceInterval]:
        raise NotImplementedError

    def diff_test_for_average(self, better_metric_name: str, worse_metric_name: str) -> DiffTestResult:
        raise NotImplementedError


class MonolingualVsCrossLingualPAWSXTask(ParaphraseClassificationTask):
    def __init__(self, mono_language: str, cross_language: str, split="validation"):
        self.mono_language = mono_language
        self.cross_language = cross_language
        self.split = split
        self.mono_dataset = PAWSXParaphraseClassificationTask(mono_language, split)
        self.cross_dataset = CrossLingualPAWSXTask(mono_language, cross_language, split)

    def __str__(self):
        return f"pawsx_mono_vs_cross_{self.mono_language}_{self.cross_language}_{self.split}"

    def get_mono_samples(self):
        return 2 * [sample for sample in self.mono_dataset.get_samples()]

    def get_cross_samples(self):
        return [sample for sample in self.cross_dataset.get_samples()]

    def get_samples(self) -> List[PAWSXSample]:
        return self.get_mono_samples() + self.get_cross_samples()

    def evaluate_part(self, metric: Metric, metric_names: List[str], part: str) -> List[ParaphraseClassificationResult]:
        if part == "mono":
            samples = self.get_mono_samples()
        elif part == "cross":
            samples = self.get_cross_samples()
        else:
            raise ValueError
        metric_output: List[MetricsDict] = metric.score_all(
            summaries=[sample.sentence1 for sample in samples],
            references_list=[[sample.sentence2] for sample in samples],
        )
        assert len(metric_output) == len(samples)
        flattened_output = [d.flatten_keys() for d in metric_output]
        results: List[ParaphraseClassificationResult] = []
        for metric_name in metric_names:
            scores = [d[metric_name] for d in flattened_output]
            results.append(ParaphraseClassificationResult(
                samples=samples,
                scores=scores,
            ))
        return results


class MonolingualVsCrossLingualParaphraseClassificationTaskWithThresholding(ParaphraseClassificationTaskWithThresholding):

    def evaluate(self, mono_metric: Metric, cross_metric: Metric, metric_names: List[str], **kwargs) -> List[ParaphraseClassificationResult]:
        mono_validation_results = self.validation_task.evaluate_part(mono_metric, metric_names, part="mono", **kwargs)
        mono_evaluation_results = self.evaluation_task.evaluate_part(mono_metric, metric_names, part="mono", **kwargs)
        cross_validation_results = self.validation_task.evaluate_part(cross_metric, metric_names, part="cross", **kwargs)
        cross_evaluation_results = self.evaluation_task.evaluate_part(cross_metric, metric_names, part="cross", **kwargs)
        validation_results = [
            ParaphraseClassificationResult(
                samples=(mono_result.samples + cross_result.samples),
                scores=(mono_result.scores + cross_result.scores),
            ) for mono_result, cross_result
            in zip(mono_validation_results, cross_validation_results)
        ]
        evaluation_results = [
            ParaphraseClassificationResult(
                samples=(mono_result.samples + cross_result.samples),
                scores=(mono_result.scores + cross_result.scores),
            ) for mono_result, cross_result
            in zip(mono_evaluation_results, cross_evaluation_results)
        ]
        for i, validation_result in enumerate(validation_results):
            _, _, thresholds = roc_curve(
                y_true=validation_result.labels,
                y_score=validation_result.scores,
            )
            accuracy_scores = []
            for threshold in thresholds:
                accuracy_scores.append(accuracy_score(
                    validation_result.labels,
                    [score > threshold for score in validation_result.scores]
                ))
            optimal_threshold = thresholds[np.array(accuracy_scores).argmax()]
            evaluation_results[i].threshold = optimal_threshold
        return evaluation_results


class MonolingualVsCrossLingualPAWSXBenchmark(ParaphraseIdentificationBenchmark):

    def __init__(self, num_iterations=1000):
        super().__init__(num_iterations)
        self.mono_languages = [
            'en',
            'de',
            'es',
            'fr',
            'ja',
            # 'ko',
            'zh',
        ]
        self.cross_languages = [
            'en',
            'de',
            'es',
            'fr',
            'ja',
            # 'ko',
            'zh',
        ]

    def __str__(self):
        return "monolingual_vs_crosslingual_pawsx_benchmark"

    def load_tasks(self):
        self.tasks = OrderedDict()
        for mono_language in self.mono_languages:
            for cross_language in self.cross_languages:
                if mono_language == cross_language:
                    continue
                self.tasks[(mono_language, cross_language)] = MonolingualVsCrossLingualParaphraseClassificationTaskWithThresholding(
                    validation_task=MonolingualVsCrossLingualPAWSXTask(mono_language, cross_language, "validation"),
                    evaluation_task=MonolingualVsCrossLingualPAWSXTask(mono_language, cross_language, "test"),
                )

    def run_metrics(self, benchmark_metrics: List[BenchmarkMetric]):
        for benchmark_metric in benchmark_metrics:
            for (mono_language, cross_language), task in self.tasks.items():
                logging.info(f"Running {benchmark_metric.title} on PAWS-X {mono_language.upper()}–{mono_language.upper()} and {mono_language.upper()}–{cross_language.upper()}")
                mono_metric = benchmark_metric.load_metric(a_lang=mono_language, b_lang=mono_language)
                cross_metric = benchmark_metric.load_metric(a_lang=mono_language, b_lang=cross_language)
                results = task.evaluate(mono_metric=mono_metric, cross_metric=cross_metric, metric_names=benchmark_metric.metric_names)
                del mono_metric
                del cross_metric
                for metric_name, result in zip(benchmark_metric.metric_names, results):
                    self.results[f"{benchmark_metric.title}_{metric_name}"][f"pawsx-{mono_language}-{cross_language}"] = result

    def get_confidence_intervals_for_average(self) -> Dict[str, ConfidenceInterval]:
        raise NotImplementedError

    def diff_test_for_average(self, better_metric_name: str, worse_metric_name: str) -> DiffTestResult:
        raise NotImplementedError
