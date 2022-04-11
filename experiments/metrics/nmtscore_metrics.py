from typing import List, Union, Optional

import numpy as np
from sacrerouge.common.util import flatten
from sacrerouge.data import MetricsDict
from sacrerouge.data.types import ReferenceType, SummaryType
from sacrerouge.metrics import Metric, ReferenceBasedMetric

from nmtscore import NMTScorer


class NMTScoreMetric(ReferenceBasedMetric):

    def __init__(self,
                 scorer: Union[NMTScorer, str] = "m2m100_418M",
                 normalize: bool = True,
                 both_directions: bool = True,
                 ):
        super().__init__()
        if isinstance(scorer, str):
            self.scorer = NMTScorer(scorer)
        else:
            self.scorer = scorer
        self.normalize = normalize
        self.both_directions = both_directions

    def _score_summaries_given_references(self, summaries: List[str], references: List[str]):
        raise NotImplementedError

    def _score_references_given_summaries(self, summaries: List[str], references: List[str]):
        raise NotImplementedError

    def score_all(self,
                  summaries: List[SummaryType],
                  references_list: List[List[ReferenceType]],
                  **kwargs
                  ) -> List[MetricsDict]:
        references = []
        for reference_set in references_list:
            if len(reference_set) != 1:
                raise NotImplementedError
            references.append(reference_set[0])
        scores = self._score_summaries_given_references(summaries, references)
        if self.both_directions:
            scores_reversed = self._score_references_given_summaries(summaries, references)
            average_scores = (np.array(scores) + np.array(scores_reversed)) / 2
            metrics_list = [
                MetricsDict({
                    self.name: average_scores[i],
                    f"{self.name}-hyp|ref": scores[i],
                    f"{self.name}-ref|hyp": scores_reversed[i],
                })
                for i in range(len(summaries))
            ]
        else:
            metrics_list = [
                MetricsDict({
                    self.name: scores[i],
                    f"{self.name}-hyp|ref": scores[i],
                })
                for i in range(len(summaries))
            ]
        return metrics_list

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
                scores = self._score_summaries_given_references(len(references) * [summary], references)
                max_score = max(scores)
                if self.both_directions:
                    scores_reversed = self._score_references_given_summaries(len(references) * [summary], references)
                    max_score_reversed = max(scores_reversed)
                    average_score = (max_score + max_score_reversed) / 2
                    metrics_dict = MetricsDict({
                        self.name: average_score,
                        f"{self.name}-hyp|ref": max_score,
                        f"{self.name}-ref|hyp": max_score_reversed,
                    })
                else:
                    metrics_dict = MetricsDict({
                        self.name: max_score,
                        f"{self.name}-hyp|ref": max_score,
                    })
                metric_list.append(metrics_dict)
            metrics_lists.append(metric_list)
        return metrics_lists


@Metric.register("nmtscore-direct")
class DirectNMTScoreMetric(NMTScoreMetric):
    name = "nmtscore-direct"

    def __init__(self,
                 summaries_lang: str,
                 references_lang: Optional[str] = None,
                 scorer: Union[NMTScorer, str] = "m2m100_418M",
                 normalize: bool = True,
                 both_directions: bool = True,
                 score_kwargs: dict = None,
                 ):
        if both_directions:
            assert references_lang is not None
        super().__init__(scorer, normalize, both_directions)
        self.summaries_lang = summaries_lang
        self.references_lang = references_lang
        self.score_kwargs = score_kwargs

    @property
    def signature(self) -> str:
        return self.scorer._build_version_string("direct", normalized=self.normalize, both_directions=self.both_directions)

    def _score_summaries_given_references(self, summaries: List[str], references: List[str]):
        return self.scorer.score_direct(
            summaries,
            references,
            self.summaries_lang,
            self.references_lang,
            normalize=self.normalize,
            both_directions=False,
            score_kwargs=self.score_kwargs,
        )

    def _score_references_given_summaries(self, summaries: List[str], references: List[str]):
        return self.scorer.score_direct(
            references,
            summaries,
            self.references_lang,
            self.summaries_lang,
            normalize=self.normalize,
            both_directions=False,
            score_kwargs=self.score_kwargs,
        )


@Metric.register("nmtscore-pivot")
class PivotNMTScoreMetric(NMTScoreMetric):
    name = "nmtscore-pivot"

    def __init__(self,
                 summaries_lang: str,
                 references_lang: Optional[str] = None,
                 pivot_lang: str = "en",
                 scorer: Union[NMTScorer, str] = "m2m100_418M",
                 normalize: bool = True,
                 both_directions: bool = True,
                 translate_kwargs: dict = None,
                 score_kwargs: dict = None,
                 ):
        if both_directions:
            assert references_lang is not None
        super().__init__(scorer, normalize, both_directions)
        self.summaries_lang = summaries_lang
        self.references_lang = references_lang
        self.pivot_lang = pivot_lang
        self.translate_kwargs = translate_kwargs
        self.score_kwargs = score_kwargs

    @property
    def signature(self) -> str:
        return self.scorer._build_version_string("pivot", pivot_lang=self.pivot_lang, normalized=self.normalize,
                                                 both_directions=self.both_directions)

    def _score_summaries_given_references(self, summaries: List[str], references: List[str]):
        return self.scorer.score_pivot(
            summaries,
            references,
            self.summaries_lang,
            self.references_lang,
            pivot_lang=self.pivot_lang,
            normalize=self.normalize,
            both_directions=False,
            translate_kwargs=self.translate_kwargs,
            score_kwargs=self.score_kwargs,
        )

    def _score_references_given_summaries(self, summaries: List[str], references: List[str]):
        return self.scorer.score_pivot(
            references,
            summaries,
            self.references_lang,
            self.summaries_lang,
            pivot_lang=self.pivot_lang,
            normalize=self.normalize,
            both_directions=False,
            translate_kwargs=self.translate_kwargs,
            score_kwargs=self.score_kwargs,
        )


@Metric.register("nmtscore-cross")
class CrossLikelihoodNMTScoreMetric(NMTScoreMetric):
    name = "nmtscore-cross"

    def __init__(self,
                 tgt_lang: str = "en",
                 scorer: Union[NMTScorer, str] = "m2m100_418M",
                 normalize: bool = True,
                 both_directions: bool = True,
                 translate_kwargs: dict = None,
                 score_kwargs: dict = None,
                 ):
        super().__init__(scorer, normalize, both_directions)
        self.tgt_lang = tgt_lang
        self.translate_kwargs = translate_kwargs
        self.score_kwargs = score_kwargs

    @property
    def signature(self) -> str:
        return self.scorer._build_version_string("cross", tgt_lang=self.tgt_lang, normalized=self.normalize,
                                                 both_directions=self.both_directions)

    def _score_summaries_given_references(self, summaries: List[str], references: List[str]):
        return self.scorer.score_cross_likelihood(
            summaries,
            references,
            tgt_lang=self.tgt_lang,
            normalize=self.normalize,
            both_directions=False,
            translate_kwargs=self.translate_kwargs,
            score_kwargs=self.score_kwargs,
        )

    def _score_references_given_summaries(self, summaries: List[str], references: List[str]):
        return self.scorer.score_cross_likelihood(
            references,
            summaries,
            tgt_lang=self.tgt_lang,
            normalize=self.normalize,
            both_directions=False,
            translate_kwargs=self.translate_kwargs,
            score_kwargs=self.score_kwargs,
        )
