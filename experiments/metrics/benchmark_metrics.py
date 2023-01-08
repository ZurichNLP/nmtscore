from typing import Callable, List

from sacrerouge.metrics import ReferenceBasedMetric

from experiments.metrics.symmetric_metric import SymmetricMetric


NMT_TRANSLATE_KWARGS = {
    "use_cache": True,
}
NMT_SCORE_KWARGS = {
    "use_cache": True,
}


class BenchmarkMetric:
    """
    Used for lazy loading of metrics
    """
    def __init__(self, title: str, metric_names: List[str], load_func: Callable):
        self.title = title
        self.metric_names = metric_names
        self.load_func = load_func

    def load_metric(self, a_lang: str, b_lang: str) -> ReferenceBasedMetric:
        return self.load_func(a_lang, b_lang)


def get_test_metrics() -> List[BenchmarkMetric]:
    from sacrerouge.metrics import ChrF, SentBleu
    return [
        BenchmarkMetric(
            title="ChrF",
            metric_names=["chrf"],
            load_func=lambda a_lang, b_lang: ChrF(),
        ),
        BenchmarkMetric(
            title="SentBLEU",
            metric_names=["sent-bleu"],
            load_func=lambda a_lang, b_lang: SentBleu(trg_lang=a_lang, tokenize=None),
        ),
    ]


def get_paraphrase_metrics(device=None) -> List[BenchmarkMetric]:
    from sacrerouge.metrics import ChrF, SentBleu, BertScore
    from nmtscore import NMTScorer
    from experiments.metrics.sbert import SBERT
    from experiments.metrics.nmtscore_metrics import DirectNMTScoreMetric, PivotNMTScoreMetric, CrossLikelihoodNMTScoreMetric
    return [
        # Surface similarity baselines
        BenchmarkMetric(
            title="ChrF",
            metric_names=["chrf"],
            load_func=lambda a_lang, b_lang: SymmetricMetric(ChrF()),
        ),
        BenchmarkMetric(
            title="SentBLEU",
            metric_names=["sent-bleu"],
            load_func=lambda a_lang, b_lang: SymmetricMetric(
                SentBleu(trg_lang=a_lang, tokenize=None),
                SentBleu(trg_lang=b_lang, tokenize=None),
            ),
        ),
        # Embedding baselines
        BenchmarkMetric(
            title="Sentence-BERT",
            metric_names=["sbert"],
            load_func=lambda a_lang, b_lang: SBERT("paraphrase-xlm-r-multilingual-v1"),
        ),
        BenchmarkMetric(
            title="BERTScore-F1",
            metric_names=["bertscore_f1"],
            load_func=lambda a_lang, b_lang: BertScore("xlm-roberta-large", num_layers=17),
        ),
        # Translation-based measures
        BenchmarkMetric(
            title="Direct_Translation_Probability",
            metric_names=["nmtscore-direct"],
            load_func=lambda a_lang, b_lang, device=device: DirectNMTScoreMetric(
                a_lang,
                b_lang,
                scorer=NMTScorer("prism", device=device),
                both_directions=True,
                score_kwargs=NMT_SCORE_KWARGS,
            ),
        ),
        BenchmarkMetric(
            title="Pivot_Translation_Probability",
            metric_names=["nmtscore-pivot"],
            load_func=lambda a_lang, b_lang, device=device: PivotNMTScoreMetric(
                a_lang,
                b_lang,
                scorer=NMTScorer("prism", device=device),
                both_directions=True,
                translate_kwargs=NMT_TRANSLATE_KWARGS,
                score_kwargs=NMT_SCORE_KWARGS,
            ),
        ),
        BenchmarkMetric(
            title="Translation_Cross-Likelihood",
            metric_names=["nmtscore-cross"],
            load_func=lambda a_lang, b_lang, device=device: CrossLikelihoodNMTScoreMetric(
                a_lang,
                b_lang,
                scorer=NMTScorer("prism", device=device),
                both_directions=True,
                translate_kwargs=NMT_TRANSLATE_KWARGS,
                score_kwargs=NMT_SCORE_KWARGS,
            ),
        ),
    ]


def get_nlg_evaluation_metrics(device=None) -> List[BenchmarkMetric]:
    from sacrerouge.metrics import ChrF, SentBleu, BertScore
    from nmtscore import NMTScorer
    from experiments.metrics.sbert import SBERT
    from experiments.metrics.nmtscore_metrics import DirectNMTScoreMetric, PivotNMTScoreMetric, CrossLikelihoodNMTScoreMetric
    return [
        # Surface similarity baselines
        BenchmarkMetric(
            title="ChrF",
            metric_names=["chrf"],
            load_func=lambda a_lang, b_lang: ChrF(),
        ),
        BenchmarkMetric(
            title="SentBLEU",
            metric_names=["sent-bleu"],
            load_func=lambda a_lang, b_lang: SentBleu(trg_lang=a_lang, tokenize=None)
        ),
        # Embedding baselines
        BenchmarkMetric(
            title="Sentence-BERT",
            metric_names=["sbert"],
            load_func=lambda a_lang, b_lang: SBERT("paraphrase-xlm-r-multilingual-v1"),
        ),
        BenchmarkMetric(
            title="BERTScore-F1",
            metric_names=["bertscore_f1", "bertscore_precision", "bertscore_recall"],
            load_func=lambda a_lang, b_lang: BertScore("xlm-roberta-large", num_layers=17),
        ),
        # Translation-based measures
        BenchmarkMetric(
            title="Direct_Translation_Probability",
            metric_names=["nmtscore-direct", "nmtscore-direct-hyp|ref", "nmtscore-direct-ref|hyp"],
            load_func=lambda a_lang, b_lang, device=device: DirectNMTScoreMetric(
                a_lang,
                b_lang,
                scorer=NMTScorer("prism", device=device),
                both_directions=True,
                score_kwargs=NMT_SCORE_KWARGS,
            ),
        ),
        BenchmarkMetric(
            title="Pivot_Translation_Probability",
            metric_names=["nmtscore-pivot", "nmtscore-pivot-hyp|ref", "nmtscore-pivot-ref|hyp"],
            load_func=lambda a_lang, b_lang, device=device: PivotNMTScoreMetric(
                a_lang,
                b_lang,
                scorer=NMTScorer("prism", device=device),
                both_directions=True,
                translate_kwargs=NMT_TRANSLATE_KWARGS,
                score_kwargs=NMT_SCORE_KWARGS,
            ),
        ),
        BenchmarkMetric(
            title="Translation_Cross-Likelihood",
            metric_names=["nmtscore-cross", "nmtscore-cross-hyp|ref", "nmtscore-cross-ref|hyp"],
            load_func=lambda a_lang, b_lang, device=device: CrossLikelihoodNMTScoreMetric(
                a_lang,
                b_lang,
                scorer=NMTScorer("prism", device=device),
                both_directions=True,
                translate_kwargs=NMT_TRANSLATE_KWARGS,
                score_kwargs=NMT_SCORE_KWARGS,
            ),
        ),
    ]


def get_paraphrase_metrics_m2m100(device=None) -> List[BenchmarkMetric]:
    from nmtscore import NMTScorer
    from experiments.metrics.nmtscore_metrics import DirectNMTScoreMetric, PivotNMTScoreMetric, CrossLikelihoodNMTScoreMetric
    batch_size = 1
    return [
        # Small model size
        BenchmarkMetric(
            title="Direct_Translation_Probability (m2m100_418M)",
            metric_names=["nmtscore-direct"],
            load_func=lambda a_lang, b_lang, device=device: DirectNMTScoreMetric(
                a_lang,
                b_lang,
                scorer=NMTScorer("m2m100_418M", device=device),
                both_directions=True,
                score_kwargs={**NMT_SCORE_KWARGS, **{"batch_size": batch_size}},
            ),
        ),
        BenchmarkMetric(
            title="Pivot_Translation_Probability (m2m100_418M)",
            metric_names=["nmtscore-pivot"],
            load_func=lambda a_lang, b_lang, device=device: PivotNMTScoreMetric(
                a_lang,
                b_lang,
                scorer=NMTScorer("m2m100_418M", device=device),
                both_directions=True,
                translate_kwargs={**NMT_TRANSLATE_KWARGS, **{"batch_size": batch_size}},
                score_kwargs={**NMT_SCORE_KWARGS, **{"batch_size": batch_size}},
            ),
        ),
        BenchmarkMetric(
            title="Translation_Cross-Likelihood (m2m100_418M)",
            metric_names=["nmtscore-cross"],
            load_func=lambda a_lang, b_lang, device=device: CrossLikelihoodNMTScoreMetric(
                a_lang,
                b_lang,
                scorer=NMTScorer("m2m100_418M", device=device),
                both_directions=True,
                translate_kwargs={**NMT_TRANSLATE_KWARGS, **{"batch_size": batch_size}},
                score_kwargs={**NMT_SCORE_KWARGS, **{"batch_size": batch_size}},
            ),
        ),
        # Large model size
        BenchmarkMetric(
            title="Direct_Translation_Probability (m2m100_1.2B)",
            metric_names=["nmtscore-direct"],
            load_func=lambda a_lang, b_lang, device=device: DirectNMTScoreMetric(
                a_lang,
                b_lang,
                scorer=NMTScorer("m2m100_1.2B", device=device),
                both_directions=True,
                score_kwargs={**NMT_SCORE_KWARGS, **{"batch_size": batch_size}},
            ),
        ),
        BenchmarkMetric(
            title="Pivot_Translation_Probability (m2m100_1.2B)",
            metric_names=["nmtscore-pivot"],
            load_func=lambda a_lang, b_lang, device=device: PivotNMTScoreMetric(
                a_lang,
                b_lang,
                scorer=NMTScorer("m2m100_1.2B", device=device),
                both_directions=True,
                translate_kwargs={**NMT_TRANSLATE_KWARGS, **{"batch_size": batch_size}},
                score_kwargs={**NMT_SCORE_KWARGS, **{"batch_size": batch_size}},
            ),
        ),
        BenchmarkMetric(
            title="Translation_Cross-Likelihood (m2m100_1.2B)",
            metric_names=["nmtscore-cross"],
            load_func=lambda a_lang, b_lang, device=device: CrossLikelihoodNMTScoreMetric(
                a_lang,
                b_lang,
                scorer=NMTScorer("m2m100_1.2B", device=device),
                both_directions=True,
                translate_kwargs={**NMT_TRANSLATE_KWARGS, **{"batch_size": batch_size}},
                score_kwargs={**NMT_SCORE_KWARGS, **{"batch_size": batch_size}},
            ),
        ),
    ]


def get_paraphrase_metrics_small100(device=None) -> List[BenchmarkMetric]:
    from nmtscore import NMTScorer
    from experiments.metrics.nmtscore_metrics import DirectNMTScoreMetric, PivotNMTScoreMetric, CrossLikelihoodNMTScoreMetric
    batch_size = 4
    return [
        BenchmarkMetric(
            title="Direct_Translation_Probability (small100)",
            metric_names=["nmtscore-direct"],
            load_func=lambda a_lang, b_lang, device=device: DirectNMTScoreMetric(
                a_lang,
                b_lang,
                scorer=NMTScorer("small100", device=device),
                both_directions=True,
                score_kwargs={**NMT_SCORE_KWARGS, **{"batch_size": batch_size}},
            ),
        ),
        BenchmarkMetric(
            title="Pivot_Translation_Probability (small100)",
            metric_names=["nmtscore-pivot"],
            load_func=lambda a_lang, b_lang, device=device: PivotNMTScoreMetric(
                a_lang,
                b_lang,
                scorer=NMTScorer("small100", device=device),
                both_directions=True,
                translate_kwargs={**NMT_TRANSLATE_KWARGS, **{"batch_size": batch_size}},
                score_kwargs={**NMT_SCORE_KWARGS, **{"batch_size": batch_size}},
            ),
        ),
        BenchmarkMetric(
            title="Translation_Cross-Likelihood (small100)",
            metric_names=["nmtscore-cross"],
            load_func=lambda a_lang, b_lang, device=device: CrossLikelihoodNMTScoreMetric(
                a_lang,
                b_lang,
                scorer=NMTScorer("small100", device=device),
                both_directions=True,
                translate_kwargs={**NMT_TRANSLATE_KWARGS, **{"batch_size": batch_size}},
                score_kwargs={**NMT_SCORE_KWARGS, **{"batch_size": batch_size}},
            ),
        ),
    ]


def get_normalization_ablation_metrics(device=None) -> List[BenchmarkMetric]:
    from nmtscore import NMTScorer
    from experiments.metrics.nmtscore_metrics import DirectNMTScoreMetric, PivotNMTScoreMetric, CrossLikelihoodNMTScoreMetric
    return [
        # Normalized
        BenchmarkMetric(
            title="Direct_Translation_Probability (normalized)",
            metric_names=["nmtscore-direct"],
            load_func=lambda a_lang, b_lang, device=device: DirectNMTScoreMetric(
                a_lang,
                b_lang,
                scorer=NMTScorer("prism", device=device),
                normalize=True,
                both_directions=True,
                score_kwargs=NMT_SCORE_KWARGS,
            ),
        ),
        BenchmarkMetric(
            title="Pivot_Translation_Probability (normalized)",
            metric_names=["nmtscore-pivot"],
            load_func=lambda a_lang, b_lang, device=device: PivotNMTScoreMetric(
                a_lang,
                b_lang,
                scorer=NMTScorer("prism", device=device),
                normalize=True,
                both_directions=True,
                translate_kwargs=NMT_TRANSLATE_KWARGS,
                score_kwargs=NMT_SCORE_KWARGS,
            ),
        ),
        BenchmarkMetric(
            title="Translation_Cross-Likelihood (normalized)",
            metric_names=["nmtscore-cross"],
            load_func=lambda a_lang, b_lang, device=device: CrossLikelihoodNMTScoreMetric(
                a_lang,
                b_lang,
                scorer=NMTScorer("prism", device=device),
                normalize=True,
                both_directions=True,
                translate_kwargs=NMT_TRANSLATE_KWARGS,
                score_kwargs=NMT_SCORE_KWARGS,
            ),
        ),
        # Unnormalized
        BenchmarkMetric(
            title="Direct_Translation_Probability (unnormalized)",
            metric_names=["nmtscore-direct"],
            load_func=lambda a_lang, b_lang, device=device: DirectNMTScoreMetric(
                a_lang,
                b_lang,
                scorer=NMTScorer("prism", device=device),
                normalize=False,
                both_directions=True,
                score_kwargs=NMT_SCORE_KWARGS,
            ),
        ),
        BenchmarkMetric(
            title="Pivot_Translation_Probability (unnormalized)",
            metric_names=["nmtscore-pivot"],
            load_func=lambda a_lang, b_lang, device=device: PivotNMTScoreMetric(
                a_lang,
                b_lang,
                scorer=NMTScorer("prism", device=device),
                normalize=False,
                both_directions=True,
                translate_kwargs=NMT_TRANSLATE_KWARGS,
                score_kwargs=NMT_SCORE_KWARGS,
            ),
        ),
        BenchmarkMetric(
            title="Translation_Cross-Likelihood (unnormalized)",
            metric_names=["nmtscore-cross"],
            load_func=lambda a_lang, b_lang, device=device: CrossLikelihoodNMTScoreMetric(
                a_lang,
                b_lang,
                scorer=NMTScorer("prism", device=device),
                normalize=False,
                both_directions=True,
                translate_kwargs=NMT_TRANSLATE_KWARGS,
                score_kwargs=NMT_SCORE_KWARGS,
            ),
        ),
    ]