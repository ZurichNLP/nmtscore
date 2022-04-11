import functools
import itertools
import logging
from collections import defaultdict, OrderedDict, Counter
from typing import List, Dict, Optional, Tuple

import numpy as np
from sacrerouge.stats import corr_ci, corr_diff_test
from tqdm import tqdm

from experiments import paraphrase_tasks
from experiments.metric_evaluation import MetricBenchmark, ConfidenceInterval, DiffTestResult
from experiments.metrics.benchmark_metrics import BenchmarkMetric
from experiments.paraphrase_tasks import ParaphraseClassificationResult


class ParaphraseIdentificationBenchmark(MetricBenchmark):

    def __init__(self, num_iterations=1000, include_pawsx=True):
        super().__init__(num_iterations)
        self.include_pawsx = include_pawsx
        self.pawsx_languages = [
            # 'en',
            'de',
            'es',
            'fr',
            'ja',
            # 'ko',
            'zh',
        ]
        self.results: Dict[str, Dict[str, ParaphraseClassificationResult]] = defaultdict(OrderedDict)  # metric -> task -> results

    def __str__(self):
        return "paraphrase_identification_benchmark"

    def load_tasks(self):
        self.mrpc = paraphrase_tasks.ParaphraseClassificationTaskWithThresholding(
            validation_task=paraphrase_tasks.MRPCTask("validation"),
            evaluation_task=paraphrase_tasks.MRPCTask("test"),
        )
        self.russian = paraphrase_tasks.RussianParaphraseClassificationTask()
        self.finnish = paraphrase_tasks.FinnishParaphraseClassificationTask()
        self.swedish = paraphrase_tasks.SwedishParaphraseClassificationTask()
        if self.include_pawsx:
            self.pawsx_tasks = [
                paraphrase_tasks.ParaphraseClassificationTaskWithThresholding(
                    validation_task=paraphrase_tasks.PAWSXParaphraseClassificationTask(language, "validation"),
                    evaluation_task=paraphrase_tasks.PAWSXParaphraseClassificationTask(language, "test"),
                )
                for language in self.pawsx_languages
            ]

    def run_metrics(self, benchmark_metrics: List[BenchmarkMetric]):
        for benchmark_metric in benchmark_metrics:
            logging.info(f"Running {benchmark_metric.title} on mrpc")
            metric = benchmark_metric.load_metric(a_lang="en", b_lang="en")
            results = self.mrpc.evaluate(metric, benchmark_metric.metric_names)
            for metric_name, result in zip(benchmark_metric.metric_names, results):
                self.results[f"{benchmark_metric.title}_{metric_name}"]["mrpc"] = result
            del metric

            logging.info(f"Running {benchmark_metric.title} on russian")
            metric = benchmark_metric.load_metric(a_lang="ru", b_lang="ru")
            results = self.russian.evaluate(metric, benchmark_metric.metric_names)
            for metric_name, result in zip(benchmark_metric.metric_names, results):
                self.results[f"{benchmark_metric.title}_{metric_name}"]["russian"] = result
            del metric

            logging.info(f"Running {benchmark_metric.title} on finnish")
            metric = benchmark_metric.load_metric(a_lang="fi", b_lang="fi")
            results = self.finnish.evaluate(metric, benchmark_metric.metric_names)
            for metric_name, result in zip(benchmark_metric.metric_names, results):
                self.results[f"{benchmark_metric.title}_{metric_name}"]["finnish"] = result
            del metric

            logging.info(f"Running {benchmark_metric.title} on swedish")
            metric = benchmark_metric.load_metric(a_lang="sv", b_lang="sv")
            results = self.swedish.evaluate(metric, benchmark_metric.metric_names)
            for metric_name, result in zip(benchmark_metric.metric_names, results):
                self.results[f"{benchmark_metric.title}_{metric_name}"]["swedish"] = result
            del metric

            if self.include_pawsx:
                for language, pawsx_task in zip(self.pawsx_languages, self.pawsx_tasks):
                    logging.info(f"Running {benchmark_metric.title} on paws-x {language}")
                    metric = benchmark_metric.load_metric(a_lang=language, b_lang=language)
                    results = pawsx_task.evaluate(metric, benchmark_metric.metric_names)
                    for metric_name, result in zip(benchmark_metric.metric_names, results):
                        self.results[f"{benchmark_metric.title}_{metric_name}"][f"pawsx-{language}"] = result
                    del metric

    def _average_accuracy_score(self, X: np.ndarray, Y: np.ndarray) -> Optional[float]:
        """
        Computes average of accuracies per task / "system"
        """
        task_accuracies = []
        for i in range(X.shape[0]):
            task_X = X[i, :]
            task_Y = Y[i, :]
            task_X = task_X[~np.isnan(task_X)]
            task_Y = task_Y[~np.isnan(task_Y)]
            if not task_X.size:
                return None
            accuracy = self._accuracy(task_X, task_Y)
            task_accuracies.append(accuracy)
        return np.mean(task_accuracies).item()

    def _overall_average_accuracy_score(self, X: np.ndarray, Y: np.ndarray) -> Optional[float]:
        """
        First average PAWS-X accuracies across languages, then average accuracies across tasks
        """
        task_results = list(self.results.values())[0]
        pawsx_indices = [i for i, (task_name, result) in enumerate(task_results.items()) if task_name.startswith("pawsx")]
        assert len(pawsx_indices) == len(self.pawsx_languages)
        non_pawsx_indices = [i for i, _ in enumerate(task_results) if i not in pawsx_indices]
        accuracy_task_indices = [i for i, (task_name, result) in enumerate(task_results.items()) if result.threshold_predictions is not None]

        task_accuracies = []
        pawsx_average_accuracy = self._average_accuracy_score(X[pawsx_indices], Y[pawsx_indices])
        task_accuracies.append(pawsx_average_accuracy)

        for i in non_pawsx_indices:
            task_X = X[i, :]
            task_Y = Y[i, :]
            task_X = task_X[~np.isnan(task_X)]
            task_Y = task_Y[~np.isnan(task_Y)]
            if not task_X.size:
                return None
            if i in accuracy_task_indices:
                accuracy = self._accuracy(task_X, task_Y)
            else:
                accuracy = self._roc_auc(task_X, task_Y)
            task_accuracies.append(accuracy)
        overall_average_accuracy = np.mean(task_accuracies).item()
        return overall_average_accuracy

    def _average_correlation(self, X: np.ndarray, Y: np.ndarray, correlation_function: str) -> Optional[float]:
        task_correlations = []
        for i in range(X.shape[0]):
            task_X = X[i, :]
            task_Y = Y[i, :]
            task_X = task_X[~np.isnan(task_X)]
            task_Y = task_Y[~np.isnan(task_Y)]
            if not task_X.size:
                return None
            correlation = self._correlation(task_X, task_Y, correlation_function, level="global")
            task_correlations.append(correlation)
        return np.mean(task_correlations).item()

    def _overall_average_correlation(self, X: np.ndarray, Y: np.ndarray, correlation_function: str) -> Optional[float]:
        task_results = list(self.results.values())[0]
        pawsx_indices = [i for i, (task_name, result) in enumerate(task_results.items()) if task_name.startswith("pawsx")]
        assert len(pawsx_indices) == len(self.pawsx_languages)
        non_pawsx_indices = [i for i, _ in enumerate(task_results) if i not in pawsx_indices]
        task_correlations = []
        pawsx_average_correlation = self._average_correlation(X[pawsx_indices], Y[pawsx_indices], correlation_function)
        task_correlations.append(pawsx_average_correlation)

        for i in non_pawsx_indices:
            task_X = X[i, :]
            task_Y = Y[i, :]
            task_X = task_X[~np.isnan(task_X)]
            task_Y = task_Y[~np.isnan(task_Y)]
            if not task_X.size:
                return None
            correlation = self._correlation(task_X, task_Y, correlation_function, level="global")
            task_correlations.append(correlation)
        overall_average_correlation = np.mean(task_correlations).item()
        return overall_average_correlation

    def get_confidence_intervals_for_individual_tasks(self) -> Dict[str, Dict[str, ConfidenceInterval]]:
        confidence_intervals: Dict[str, Dict[str, ConfidenceInterval]] = defaultdict(dict)  # metric -> task -> CI
        for metric, task_results in self.results.items():
            for task, result in task_results.items():
                if result.threshold_predictions is not None:
                    X = np.array(result.threshold_predictions)
                    corr_func = self._accuracy
                else:
                    X = np.array(result.scores)
                    corr_func = self._roc_auc
                Y = np.array(result.labels)
                X = np.expand_dims(X, 0)
                Y = np.expand_dims(Y, 0)
                lower, upper = corr_ci(corr_func, X, Y, method="bootstrap-input",
                                       kwargs={"num_samples": self.num_iterations})
                observed = corr_func(X, Y)
                confidence_intervals[metric][task] = ConfidenceInterval(lower, observed, upper)
        return confidence_intervals

    def get_confidence_intervals_for_pawsx_average(self) -> Dict[str, ConfidenceInterval]:
        confidence_intervals: Dict[str, ConfidenceInterval] = dict()  # metric -> CI
        for metric, task_results in self.results.items():
            metric_pawsx_results = [result for task, result in task_results.items() if task.startswith("pawsx-")]
            # Approach: Treat languages as "systems" and sample–language pairs as "documents"
            overall_num_samples = sum([len(result) for result in metric_pawsx_results])
            num_tasks = len([task for task, result in task_results.items() if task.startswith("pawsx-")])
            X = np.nan * np.ones((num_tasks, overall_num_samples))
            Y = np.nan * np.ones((num_tasks, overall_num_samples))
            for i, result in enumerate(metric_pawsx_results):
                num_samples_so_far = sum([len(result) for result in metric_pawsx_results[:i]])
                num_samples_here = len(result)
                X[i, num_samples_so_far:(num_samples_so_far+num_samples_here)] = result.threshold_predictions
                Y[i, num_samples_so_far:(num_samples_so_far+num_samples_here)] = result.labels

            lower, upper = corr_ci(self._average_accuracy_score, X, Y, method="bootstrap-input",
                                   kwargs={"num_samples": self.num_iterations})
            observed = self._average_accuracy_score(X, Y)
            confidence_intervals[metric] = ConfidenceInterval(lower, observed, upper)
        return confidence_intervals

    def get_confidence_intervals_for_overall_average(self) -> Dict[str, ConfidenceInterval]:
        confidence_intervals: Dict[str, ConfidenceInterval] = dict()  # metric -> CI
        for metric, task_results in self.results.items():
            # Approach: Treat tasks as "systems" and sample–language pairs as "documents"
            overall_num_samples = sum([len(result) for result in task_results.values()])
            num_tasks = len(task_results)
            X = np.nan * np.ones((num_tasks, overall_num_samples))
            Y = np.nan * np.ones((num_tasks, overall_num_samples))
            for i, result in enumerate(task_results.values()):
                num_samples_so_far = sum([len(result) for result in list(task_results.values())][:i])
                num_samples_here = len(result)
                X[i, num_samples_so_far:(num_samples_so_far+num_samples_here)] = result.threshold_predictions or result.scores
                Y[i, num_samples_so_far:(num_samples_so_far+num_samples_here)] = result.labels

            lower, upper = corr_ci(self._overall_average_accuracy_score, X, Y, method="bootstrap-input",
                                   kwargs={"num_samples": self.num_iterations})
            observed = self._overall_average_accuracy_score(X, Y)
            confidence_intervals[metric] = ConfidenceInterval(lower, observed, upper)
        return confidence_intervals

    def diff_test_for_individual_task(self, better_metric_name: str, worse_metric_name: str, task_name: str) -> DiffTestResult:
        better_metric_result = self.results[better_metric_name][task_name]
        worse_metric_result = self.results[worse_metric_name][task_name]
        assert len(better_metric_result) == len(worse_metric_result)
        assert better_metric_result.labels == worse_metric_result.labels
        assert (better_metric_result.threshold_predictions is not None) == (worse_metric_result.threshold_predictions is not None)

        if better_metric_result.threshold_predictions is not None:
            X = np.array(better_metric_result.threshold_predictions)
            Y = np.array(worse_metric_result.threshold_predictions)
            corr_func = self._accuracy
        else:
            X = np.array(better_metric_result.scores)
            Y = np.array(worse_metric_result.scores)
            corr_func = self._roc_auc
        Z = np.array(better_metric_result.labels)
        pvalue, delta = corr_diff_test(
            corr_func=corr_func,
            X=np.expand_dims(X, 0),
            Y=np.expand_dims(Y, 0),
            Z=np.expand_dims(Z, 0),
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

    def diff_test_for_pawsx_average(self, better_metric_name: str, worse_metric_name: str) -> DiffTestResult:
        better_metric_task_results = []
        worse_metric_task_results = []
        for task_name, result in self.results[better_metric_name].items():
            if task_name.startswith("pawsx"):
                better_metric_task_results.append(result)
                worse_metric_task_results.append(self.results[worse_metric_name][task_name])
        for better_result, worse_result in zip(better_metric_task_results, worse_metric_task_results):
            assert len(better_result) == len(worse_result)
            assert better_result.labels == worse_result.labels
        overall_num_samples = sum([len(result) for result in better_metric_task_results])
        num_tasks = len(better_metric_task_results)
        X = np.nan * np.ones((num_tasks, overall_num_samples))
        Y = np.nan * np.ones((num_tasks, overall_num_samples))
        Z = np.nan * np.ones((num_tasks, overall_num_samples))
        for i in range(num_tasks):
            num_samples_so_far = sum([len(result) for result in better_metric_task_results[:i]])
            num_samples_here = len(better_metric_task_results[i])
            X[i, num_samples_so_far:(num_samples_so_far + num_samples_here)] = better_metric_task_results[i].threshold_predictions
            Y[i, num_samples_so_far:(num_samples_so_far + num_samples_here)] = worse_metric_task_results[i].threshold_predictions
            Z[i, num_samples_so_far:(num_samples_so_far + num_samples_here)] = better_metric_task_results[i].labels
        pvalue, delta = corr_diff_test(
            corr_func=self._average_accuracy_score,
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

    def diff_test_for_overall_average(self, better_metric_name: str, worse_metric_name: str) -> DiffTestResult:
        better_results = self.results[better_metric_name]
        worse_results = self.results[worse_metric_name]
        for better_result, worse_result in zip(better_results.values(), worse_results.values()):
            assert len(better_result) == len(worse_result)
            assert better_result.labels == worse_result.labels
            assert (better_result.threshold_predictions is not None) == (worse_result.threshold_predictions is not None)
        overall_num_samples = sum([len(result) for result in better_results.values()])
        num_tasks = len(better_results)
        X = np.nan * np.ones((num_tasks, overall_num_samples))
        Y = np.nan * np.ones((num_tasks, overall_num_samples))
        Z = np.nan * np.ones((num_tasks, overall_num_samples))
        for i in range(num_tasks):
            num_samples_so_far = sum([len(result) for result in list(better_results.values())][:i])
            better_result = list(better_results.values())[i]
            worse_result = list(worse_results.values())[i]
            num_samples_here = len(better_result)
            X[i, num_samples_so_far:(num_samples_so_far + num_samples_here)] = better_result.threshold_predictions or better_result.scores
            Y[i, num_samples_so_far:(num_samples_so_far + num_samples_here)] = worse_result.threshold_predictions or worse_result.scores
            Z[i, num_samples_so_far:(num_samples_so_far + num_samples_here)] = better_result.labels
        pvalue, delta = corr_diff_test(
            corr_func=self._overall_average_accuracy_score,
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

    def pairwise_diff_tests(self, task_name: str) -> None:
        """
        :param task_name: Task name or "pawsx_average" or "overall_average"
        """
        assert self.results
        metric_names = list(self.results)
        diff_results: Dict[Tuple[str, str]: float] = dict()
        for metric_name1, metric_name2 in tqdm(list(itertools.product(metric_names, metric_names))):
            if task_name == "pawsx_average":
                result = self.diff_test_for_pawsx_average(metric_name1, metric_name2)
            elif task_name == "overall_average":
                result = self.diff_test_for_overall_average(metric_name1, metric_name2)
            else:
                result = self.diff_test_for_individual_task(metric_name1, metric_name2, task_name)
            diff_results[(metric_name1, metric_name2)] = result

        outperform_counter = Counter()
        for (metric_name1, metric_name2), diff_result in diff_results.items():
            outperform_counter[metric_name1] += diff_result.delta > 0
        ordered_metric_names = [metric_name for metric_name, _ in outperform_counter.most_common()]

        print(f"# {task_name}")
        print()
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

    def get_confidence_intervals_for_pairwise_correlation(self, correlation: str = "kendall") -> Dict[Tuple[str, str], ConfidenceInterval]:
        confidence_intervals: Dict[Tuple[str, str], ConfidenceInterval] = dict()  # metric pair -> CI
        for metric1 in self.results:
            for metric2 in self.results:
                if metric1 >= metric2:
                    continue
                overall_num_samples = sum([len(result) for result in self.results[metric1].values()])
                num_tasks = len(self.results[metric1])
                X = np.nan * np.ones((num_tasks, overall_num_samples))
                Y = np.nan * np.ones((num_tasks, overall_num_samples))
                for i, task in enumerate(self.results[metric1]):
                    num_samples_so_far = sum([len(result) for result in list(self.results[metric1].values())][:i])
                    num_samples_here = len(self.results[metric1][task])
                    X[i, num_samples_so_far:(num_samples_so_far+num_samples_here)] = self.results[metric1][task].scores
                    Y[i, num_samples_so_far:(num_samples_so_far+num_samples_here)] = self.results[metric2][task].scores
                corr_func = functools.partial(self._overall_average_correlation, correlation_function=correlation)
                lower, upper = corr_ci(corr_func, X, Y, method="bootstrap-input",
                                       kwargs={"num_samples": self.num_iterations})
                observed = corr_func(X, Y)
                confidence_intervals[(metric1, metric2)] = ConfidenceInterval(lower, observed, upper)
        return confidence_intervals


class ValidationOnlyParaphraseIdentificationBenchmark(ParaphraseIdentificationBenchmark):

    def __init__(self, num_iterations=1000):
        super().__init__(num_iterations)

    def __str__(self):
        return "validation_only_paraphrase_identification_benchmark"

    def load_tasks(self):
        self.pawsx_tasks = [
            paraphrase_tasks.PAWSXParaphraseClassificationTask(language, "validation")
            for language in self.pawsx_languages
        ]

    def run_metrics(self, benchmark_metrics: List[BenchmarkMetric]):
        for benchmark_metric in benchmark_metrics:
            for language, pawsx_task in zip(self.pawsx_languages, self.pawsx_tasks):
                logging.info(f"Running {benchmark_metric.title} on paws-x {language}")
                metric = benchmark_metric.load_metric(a_lang=language, b_lang=language)
                results = pawsx_task.evaluate(metric, benchmark_metric.metric_names)
                for metric_name, result in zip(benchmark_metric.metric_names, results):
                    self.results[f"{benchmark_metric.title}_{metric_name}"][f"pawsx-{language}"] = result
                del metric

    def _average_accuracy_score(self, X: np.ndarray, Y: np.ndarray) -> Optional[float]:
        """
        Computes average of AUC per task / "system"
        """
        task_accuracies = []
        for i in range(X.shape[0]):
            task_X = X[i, :]
            task_Y = Y[i, :]
            task_X = task_X[~np.isnan(task_X)]
            task_Y = task_Y[~np.isnan(task_Y)]
            if not task_X.size:
                return None
            accuracy = self._roc_auc(task_X, task_Y)
            task_accuracies.append(accuracy)
        return np.mean(task_accuracies).item()
