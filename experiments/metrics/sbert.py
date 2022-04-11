from typing import List

from sacrerouge.common.util import flatten
from sacrerouge.data import MetricsDict
from sacrerouge.data.types import ReferenceType, SummaryType
from sacrerouge.metrics import Metric, ReferenceBasedMetric
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pairwise_cos_sim


@Metric.register('sbert')
class SBERT(ReferenceBasedMetric):

    def __init__(self,
                 sbert_model: str = "paraphrase-xlm-r-multilingual-v1",
                 **kwargs,
                 ):
        super().__init__()
        self.model = SentenceTransformer(sbert_model, **kwargs)

    def _score(self, sentences1: List[str], sentences2: List[str]) -> List[float]:
        embeddings1 = self.model.encode(sentences1, convert_to_tensor=True, show_progress_bar=False)
        embeddings2 = self.model.encode(sentences2, convert_to_tensor=True, show_progress_bar=False)
        return pairwise_cos_sim(embeddings1, embeddings2).tolist()

    def score_multi_all(self,
                        summaries_list: List[List[SummaryType]],
                        references_list: List[List[ReferenceType]],
                        **kwargs
                        ) -> List[List[MetricsDict]]:
        summaries_list = [[flatten(summary) for summary in summaries] for summaries in summaries_list]
        references_list = [[flatten(reference) for reference in references] for references in references_list]
        metrics_lists = []
        for summaries, references in zip(summaries_list, references_list):
            metric_list = []
            for summary in summaries:
                summary_scores = self._score(
                    sentences1=len(references) * [summary],
                    sentences2=references,
                )
                aggregated_summary_score = max(summary_scores)
                metric_list.append(MetricsDict({
                    f'sbert': aggregated_summary_score,
                }))
            metrics_lists.append(metric_list)
        return metrics_lists
