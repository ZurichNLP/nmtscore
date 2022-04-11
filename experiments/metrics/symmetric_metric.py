from typing import List, Optional

from sacrerouge.data import MetricsDict
from sacrerouge.data.types import SummaryType, ReferenceType
from sacrerouge.metrics import ReferenceBasedMetric


class SymmetricMetric(ReferenceBasedMetric):

    def __init__(self, directed_metric: ReferenceBasedMetric, directed_metric2: Optional[ReferenceBasedMetric] = None):
        super().__init__()
        self.forward_metric = directed_metric
        self.backward_metric = directed_metric2 or directed_metric

    def score_multi_all(self,
                        summaries_list: List[List[SummaryType]],
                        references_list: List[List[ReferenceType]],
                        **kwargs
                        ) -> List[List[MetricsDict]]:
        forward_metrics_list = self.forward_metric.score_multi_all(summaries_list, references_list, **kwargs)
        backward_metrics_list = self.backward_metric.score_multi_all(references_list, summaries_list, **kwargs)
        for i, (forward_metrics_dicts, backward_metrics_dicts) in enumerate(zip(forward_metrics_list, backward_metrics_list)):
            for j, (forward_metrics_dict, backward_metrics_dict) in enumerate(zip(forward_metrics_dicts, backward_metrics_dicts)):
                forward_metrics_list[i][j] += backward_metrics_dict
        return forward_metrics_list
