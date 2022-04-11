import sys
from pathlib import Path
import timeit

from sacrerouge.metrics import ChrF, SentBleu, BertScore
from nmtscore import NMTScorer

sys.path.insert(1, str(Path(__file__).resolve().parent.parent.parent))
from experiments import paraphrase_tasks
from experiments.metrics.benchmark_metrics import BenchmarkMetric
from experiments.metrics.sbert import SBERT
from experiments.metrics.symmetric_metric import SymmetricMetric
from experiments.metrics.nmtscore_metrics import DirectNMTScoreMetric, PivotNMTScoreMetric, CrossLikelihoodNMTScoreMetric

BATCH_SIZE = 32

NMT_TRANSLATE_KWARGS = {
  "batch_size": BATCH_SIZE,
  "use_cache": False,
}
NMT_SCORE_KWARGS = {
  "batch_size": BATCH_SIZE,
  "use_cache": False,
}

benchmark_metrics = [
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
        load_func=lambda a_lang, b_lang: BertScore("xlm-roberta-large", num_layers=17, batch_size=BATCH_SIZE),
    ),
    # Translation-based measures
    BenchmarkMetric(
        title="Direct_Translation_Probability (normalized)",
        metric_names=["nmtscore-direct"],
        load_func=lambda a_lang, b_lang: DirectNMTScoreMetric(
            a_lang,
            b_lang,
            scorer=NMTScorer("prism", device=0),
            both_directions=True,
            score_kwargs=NMT_SCORE_KWARGS,
        ),
    ),
    BenchmarkMetric(
        title="Pivot_Translation_Probability (normalized)",
        metric_names=["nmtscore-pivot"],
        load_func=lambda a_lang, b_lang: PivotNMTScoreMetric(
            a_lang,
            b_lang,
            scorer=NMTScorer("prism", device=0),
            both_directions=True,
            translate_kwargs=NMT_TRANSLATE_KWARGS,
            score_kwargs=NMT_SCORE_KWARGS,
        ),
    ),
    BenchmarkMetric(
        title="Translation_Cross-Likelihood (normalized)",
        metric_names=["nmtscore-cross"],
        load_func=lambda a_lang, b_lang: CrossLikelihoodNMTScoreMetric(
            scorer=NMTScorer("prism", device=0),
            both_directions=True,
            translate_kwargs=NMT_TRANSLATE_KWARGS,
            score_kwargs=NMT_SCORE_KWARGS,
        ),
    ),
    BenchmarkMetric(
        title="Direct_Translation_Probability (unnormalized)",
        metric_names=["nmtscore-direct"],
        load_func=lambda a_lang, b_lang: DirectNMTScoreMetric(
            a_lang,
            b_lang,
            scorer=NMTScorer("prism", device=0),
            normalize=False,
            both_directions=True,
            score_kwargs=NMT_SCORE_KWARGS,
        ),
    ),
    BenchmarkMetric(
        title="Pivot_Translation_Probability (unnormalized)",
        metric_names=["nmtscore-pivot"],
        load_func=lambda a_lang, b_lang: PivotNMTScoreMetric(
            a_lang,
            b_lang,
            scorer=NMTScorer("prism", device=0),
            normalize=False,
            both_directions=True,
            translate_kwargs=NMT_TRANSLATE_KWARGS,
            score_kwargs=NMT_SCORE_KWARGS,
        ),
    ),
    BenchmarkMetric(
        title="Translation_Cross-Likelihood (unnormalized)",
        metric_names=["nmtscore-cross"],
        load_func=lambda a_lang, b_lang: CrossLikelihoodNMTScoreMetric(
            scorer=NMTScorer("prism", device=0),
            normalize=False,
            both_directions=True,
            translate_kwargs=NMT_TRANSLATE_KWARGS,
            score_kwargs=NMT_SCORE_KWARGS,
        ),
    ),
]

mrpc = paraphrase_tasks.MRPCTask("validation")
num_pairs = len(mrpc.get_samples())
print("Number of pairs: ", num_pairs)

for benchmark_metric in benchmark_metrics:
    print(benchmark_metric.title)
    metric = benchmark_metric.load_metric(a_lang="en", b_lang="en")
    time = timeit.timeit(lambda: mrpc.evaluate(metric, benchmark_metric.metric_names), number=1)
    print(time / num_pairs * 1000, "ms")
    del metric
