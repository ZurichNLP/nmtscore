import unittest

from sacrerouge.common.testing.metric_test_cases import ReferenceBasedMetricTestCase

from experiments.metrics.nmtscore_metrics import DirectNMTScoreMetric, PivotNMTScoreMetric, CrossLikelihoodNMTScoreMetric


class DirectNMTScoreMetricTestCase(ReferenceBasedMetricTestCase):

    def setUp(self) -> None:
        self.metric = DirectNMTScoreMetric(summaries_lang="en", references_lang="en")

    @unittest.skip("Slow")
    def test_score(self):
        self.metric.score_all(self.summaries, self.references_list)

    @unittest.skip("Slow")
    def test_order_invariant(self):
        self.assert_order_invariant(self.metric)

    def test_correct(self):
        self.assertAlmostEqual(
            self.metric.score(summary="This is a test.", references=["That is an attempt."])["nmtscore-direct"],
            self.metric.scorer.score_direct("This is a test.", "That is an attempt.", "en", "en"),
        )
        self.assertAlmostEqual(
            self.metric.score(summary="This is a test.", references=["That is an attempt."])["nmtscore-direct-hyp|ref"],
            self.metric.scorer.score_direct("This is a test.", "That is an attempt.", "en", "en", both_directions=False),
        )
        self.assertAlmostEqual(
            self.metric.score(summary="This is a test.", references=["That is an attempt."])["nmtscore-direct-ref|hyp"],
            self.metric.scorer.score_direct("That is an attempt.", "This is a test.", "en", "en", both_directions=False),
        )


class PivotNMTScoreMetricTestCase(ReferenceBasedMetricTestCase):

    def setUp(self) -> None:
        self.metric = PivotNMTScoreMetric(summaries_lang="en", references_lang="en")

    @unittest.skip("Slow")
    def test_score(self):
        self.metric.score_all(self.summaries, self.references_list)

    @unittest.skip("Slow")
    def test_order_invariant(self):
        self.assert_order_invariant(self.metric)

    def test_correct(self):
        self.assertAlmostEqual(
            self.metric.score(summary="This is a test.", references=["That is an attempt."])["nmtscore-pivot"],
            self.metric.scorer.score_pivot("This is a test.", "That is an attempt.", "en", "en"),
        )
        self.assertAlmostEqual(
            self.metric.score(summary="This is a test.", references=["That is an attempt."])["nmtscore-pivot-hyp|ref"],
            self.metric.scorer.score_pivot("This is a test.", "That is an attempt.", "en", "en", both_directions=False),
        )
        self.assertAlmostEqual(
            self.metric.score(summary="This is a test.", references=["That is an attempt."])["nmtscore-pivot-ref|hyp"],
            self.metric.scorer.score_pivot("That is an attempt.", "This is a test.", "en", "en", both_directions=False),
        )


class CrossLikelihoodNMTScoreMetricTestCase(ReferenceBasedMetricTestCase):

    def setUp(self) -> None:
        self.metric = CrossLikelihoodNMTScoreMetric()

    @unittest.skip("Slow")
    def test_score(self):
        self.metric.score_all(self.summaries, self.references_list)

    @unittest.skip("Slow")
    def test_order_invariant(self):
        self.assert_order_invariant(self.metric)

    def test_correct(self):
        self.assertAlmostEqual(
            self.metric.score(summary="This is a test.", references=["That is an attempt."])["nmtscore-cross"],
            self.metric.scorer.score_cross_likelihood("This is a test.", "That is an attempt."),
        )
        self.assertAlmostEqual(
            self.metric.score(summary="This is a test.", references=["That is an attempt."])["nmtscore-cross-hyp|ref"],
            self.metric.scorer.score_cross_likelihood("This is a test.", "That is an attempt.", both_directions=False),
        )
        self.assertAlmostEqual(
            self.metric.score(summary="This is a test.", references=["That is an attempt."])["nmtscore-cross-ref|hyp"],
            self.metric.scorer.score_cross_likelihood("That is an attempt.", "This is a test.", both_directions=False),
        )
