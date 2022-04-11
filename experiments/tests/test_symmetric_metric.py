from unittest import TestCase

from sacrerouge.metrics import SentBleu

from experiments.metrics.symmetric_metric import SymmetricMetric


class SymmetricMetricTestCase(TestCase):

    def setUp(self) -> None:
        self.bleu = SentBleu()
        self.symmetric = SymmetricMetric(self.bleu)

    def test_asymmetrical(self):
        self.assertNotEqual(
            self.bleu.score('The dog bit the man.', ['The dog had bit the man.'])["sent-bleu"],
            self.bleu.score('The dog had bit the man.', ['The dog bit the man.'])["sent-bleu"]
        )

    def test_symmetrical(self):
        self.assertEqual(
            self.symmetric.score('The dog bit the man.', ['The dog had bit the man.'])["sent-bleu"],
            self.symmetric.score('The dog had bit the man.', ['The dog bit the man.'])["sent-bleu"]
        )

    def test_separate_directions(self):
        symmetric = SymmetricMetric(self.bleu, self.bleu)
        self.assertEqual(
            symmetric.score('The dog bit the man.', ['The dog had bit the man.'])["sent-bleu"],
            symmetric.score('The dog had bit the man.', ['The dog bit the man.'])["sent-bleu"]
        )
