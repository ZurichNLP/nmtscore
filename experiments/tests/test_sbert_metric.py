from sacrerouge.common.testing.metric_test_cases import ReferenceBasedMetricTestCase
from sacrerouge.common.testing.util import sacrerouge_command_exists

from experiments.metrics.sbert import SBERT


class SBERTTestCase(ReferenceBasedMetricTestCase):

    def setUp(self) -> None:
        self.metric = SBERT()

    def test_score(self):
        self.metric.score_all(self.summaries, self.references_list)

    def test_order_invariant(self):
        self.assert_order_invariant(self.metric)

    def test_command_exists(self):
        assert sacrerouge_command_exists(['sbert'])
