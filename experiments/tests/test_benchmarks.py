from unittest import TestCase

from matplotlib import pyplot as plt

from experiments.multilingual_amr import MultilingualAMRBenchmark
from experiments.crosslingual_pawsx import CrossLingualPAWSXBenchmark, MonolingualVsCrossLingualPAWSXBenchmark
from experiments.metrics.benchmark_metrics import get_test_metrics
from experiments.paraphrase_benchmark import ParaphraseIdentificationBenchmark, \
    ValidationOnlyParaphraseIdentificationBenchmark
from experiments.webnlg import WebNLGBenchmark


class ParaphraseIdentificationBenchmarkTestCase(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.benchmark = ParaphraseIdentificationBenchmark(num_iterations=2)
        cls.benchmark.load_tasks()
        cls.benchmark_metrics = get_test_metrics()
        cls.benchmark.run_metrics(cls.benchmark_metrics)
        cls.results_path = cls.benchmark.default_results_path.with_suffix(".test.pickle")
        cls.benchmark.save_results(cls.results_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.results_path.unlink()

    def setUp(self) -> None:
        self.benchmark.load_results(self.results_path)
    
    def test_get_confidence_intervals_for_individual_tasks(self):
        confidence_intervals = self.benchmark.get_confidence_intervals_for_individual_tasks()
        print(confidence_intervals)

    def test_get_confidence_intervals_for_pawsx_average(self):
        confidence_intervals = self.benchmark.get_confidence_intervals_for_pawsx_average()
        print(confidence_intervals)

    def test_get_confidence_intervals_for_overall_average(self):
        confidence_intervals = self.benchmark.get_confidence_intervals_for_overall_average()
        print(confidence_intervals)

    def test_diff_test_for_individual_task(self):
        pvalue = self.benchmark.diff_test_for_individual_task(
            better_metric_name="SentBLEU_sent-bleu",
            worse_metric_name="ChrF_chrf",
            task_name="mrpc",
        )
        print(pvalue)

    def test_diff_test_for_pawsx_average(self):
        pvalue = self.benchmark.diff_test_for_pawsx_average(better_metric_name="SentBLEU_sent-bleu", worse_metric_name="ChrF_chrf")
        print(pvalue)

    def test_diff_test_for_overall_average(self):
        pvalue = self.benchmark.diff_test_for_overall_average(better_metric_name="SentBLEU_sent-bleu", worse_metric_name="ChrF_chrf")
        print(pvalue)

    def test_pairwise_diff_tests(self):
        self.benchmark.pairwise_diff_tests("mrpc")

    def test_get_confidence_intervals_for_pairwise_correlation(self):
        print(self.benchmark.get_confidence_intervals_for_pairwise_correlation())


class ValidationOnlyParaphraseIdentificationBenchmarkTestCase(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.benchmark = ValidationOnlyParaphraseIdentificationBenchmark(num_iterations=2)
        cls.benchmark.load_tasks()
        cls.benchmark_metrics = get_test_metrics()
        cls.benchmark.run_metrics(cls.benchmark_metrics)
        cls.results_path = cls.benchmark.default_results_path.with_suffix(".test.pickle")
        cls.benchmark.save_results(cls.results_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.results_path.unlink()

    def setUp(self) -> None:
        self.benchmark.load_results(self.results_path)

    def test_get_confidence_intervals_for_individual_tasks(self):
        confidence_intervals = self.benchmark.get_confidence_intervals_for_individual_tasks()
        print(confidence_intervals)

    def test_get_confidence_intervals_for_overall_average(self):
        confidence_intervals = self.benchmark.get_confidence_intervals_for_overall_average()
        print(confidence_intervals)

    def test_diff_test_for_individual_task(self):
        pvalue = self.benchmark.diff_test_for_individual_task(
            better_metric_name="SentBLEU_sent-bleu",
            worse_metric_name="ChrF_chrf",
            task_name="pawsx-de",
        )
        print(pvalue)

    def test_diff_test_for_overall_average(self):
        pvalue = self.benchmark.diff_test_for_overall_average(better_metric_name="SentBLEU_sent-bleu", worse_metric_name="ChrF_chrf")
        print(pvalue)

    def test_pairwise_diff_tests(self):
        self.benchmark.pairwise_diff_tests("pawsx-de")


class CrossLingualPAWSXBenchmarkTestCase(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.benchmark = CrossLingualPAWSXBenchmark(num_iterations=2)
        cls.benchmark.pawsx_languages = ["de", "fr", "es"]
        cls.benchmark.load_tasks()
        cls.benchmark_metrics = get_test_metrics()
        cls.benchmark.run_metrics(cls.benchmark_metrics)
        cls.results_path = cls.benchmark.default_results_path.with_suffix(".test.pickle")
        cls.benchmark.save_results(cls.results_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.results_path.unlink()

    def setUp(self) -> None:
        self.benchmark.load_results(self.results_path)

    def test_get_confidence_intervals_for_individual_tasks(self):
        confidence_intervals = self.benchmark.get_confidence_intervals_for_individual_tasks()
        print(confidence_intervals)

    def test_get_confidence_intervals_for_pawsx_average(self):
        confidence_intervals = self.benchmark.get_confidence_intervals_for_pawsx_average()
        print(confidence_intervals)

    def test_pairwise_diff_tests(self):
        self.benchmark.pairwise_diff_tests("pawsx-de-es")
        self.benchmark.pairwise_diff_tests("pawsx_average")


class MonolingualVsCrossLingualPAWSXBenchmarkTestCase(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.benchmark = MonolingualVsCrossLingualPAWSXBenchmark(num_iterations=2)
        cls.benchmark.mono_languages = ["de"]
        cls.benchmark.cross_languages = ["fr", "es"]
        cls.benchmark.load_tasks()
        cls.benchmark_metrics = get_test_metrics()
        cls.benchmark.run_metrics(cls.benchmark_metrics)
        cls.results_path = cls.benchmark.default_results_path.with_suffix(".test.pickle")
        cls.benchmark.save_results(cls.results_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.results_path.unlink()

    def setUp(self) -> None:
        self.benchmark.load_results(self.results_path)

    def test_get_confidence_intervals_for_individual_tasks(self):
        confidence_intervals = self.benchmark.get_confidence_intervals_for_individual_tasks()
        print(confidence_intervals)

    def test_get_confidence_intervals_for_pawsx_average(self):
        confidence_intervals = self.benchmark.get_confidence_intervals_for_pawsx_average()
        print(confidence_intervals)

    def test_pairwise_diff_tests(self):
        self.benchmark.pairwise_diff_tests("pawsx-de-fr")
        self.benchmark.pairwise_diff_tests("pawsx-de-es")
        self.benchmark.pairwise_diff_tests("pawsx_average")


class WebNLGBenchmarkTestCase(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.benchmark = WebNLGBenchmark(language="en", num_iterations=2)
        cls.benchmark_metrics = get_test_metrics()
        cls.benchmark.run_metrics(cls.benchmark_metrics)
        cls.results_path = cls.benchmark.default_results_path.with_suffix(".test.pickle")
        cls.benchmark.save_results(cls.results_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.results_path.unlink()

    def setUp(self) -> None:
        self.benchmark.load_results(self.results_path)

    def test_pairwise_diff_tests(self):
        self.benchmark.pairwise_diff_tests()

    def test_plot_confidence_intervals(self):
        ax = self.benchmark.plot_confidence_intervals()
        # plt.savefig("test.svg")
        plt.show()

    def test_print_statistics(self):
        self.benchmark.print_statistics()


class MultilingualAMRBenchmarkTestCase(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.benchmark = MultilingualAMRBenchmark(num_iterations=2)
        cls.benchmark_metrics = get_test_metrics()
        cls.benchmark.run_metrics(cls.benchmark_metrics)
        cls.results_path = cls.benchmark.default_results_path.with_suffix(".test.pickle")
        cls.benchmark.save_results(cls.results_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.results_path.unlink()

    def setUp(self) -> None:
        self.benchmark.load_results(self.results_path)

    def test_pairwise_diff_tests(self):
        self.benchmark.pairwise_diff_tests()

    def test_plot_confidence_intervals(self):
        # ax = self.benchmark.plot_confidence_intervals(metric_names=['chrf_chrf', 'sent-bleu_sent-bleu', 'round-trip-prism_round-trip-en-forward', 'round-trip-prism_round-trip-en-backward', 'round-trip-prism_round-trip-en', 'zero-shot-prism_zero-shot-forward', 'zero-shot-prism_zero-shot-backward', 'zero-shot-prism_zero-shot', 'xppl-prism_xppl-en-forward', 'xppl-prism_xppl-en-backward', 'xppl-prism_xppl-en', 'bertscore-xlm-roberta-large_bertscore_recall', 'bertscore-xlm-roberta-large_bertscore_precision', 'bertscore-xlm-roberta-large_bertscore_f1', 'sbert-paraphrase-xlm-r-multilingual-v1_sbert'])
        ax = self.benchmark.plot_confidence_intervals(color="#1f77b4")
        plt.show()

    def test_print_statistics(self):
        self.benchmark.print_statistics()
