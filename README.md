# NMTScore
![Master](https://github.com/ZurichNLP/nmtscore/workflows/unittest/badge.svg?branch=master&event=push)
[![PyPI](https://img.shields.io/pypi/v/nmtscore)](https://pypi.python.org/pypi/nmtscore/)

A library of translation-based text similarity measures.

To learn more about how these measures work, have a look at [Jannis' blog post](https://vamvas.ch/nmtscore-text-similarity-via-translation). Also, read our paper, ["NMTScore: A Multilingual Analysis of Translation-based Text Similarity Measures"](https://arxiv.org/abs/2204.13692) (Findings of EMNLP).

<img src="img/figure1.png" alt="Three text similarity measures implemented in this library" width="500">

## Installation

- Requires Python >= 3.7 and PyTorch
- `pip install nmtscore`
- Extra requirements for the Prism model: `pip install nmtscore[prism]`

## Usage

### NMTScorer
Instantiate a scorer and start scoring short sentence pairs.

```python
from nmtscore import NMTScorer

scorer = NMTScorer()

scorer.score("This is a sentence.", "This is another sentence.")
# 0.45192727655379844
```

#### Different similarity measures
The library implements three different measures:

```python
# Translation cross-likelihood (default)
scorer.score_cross_likelihood(a, b, tgt_lang="en", normalize=True, both_directions=True)

# Direct translation probability
scorer.score_direct(a, b, a_lang="en", b_lang="en", normalize=True, both_directions=True)

# Pivot translation probability
scorer.score_pivot(a, b, a_lang="en", b_lang="en", pivot_lang="en", normalize=True, both_directions=True)
```

The `score` method is a shortcut for cross-likelihood.

#### Batch processing
The scoring methods also accept lists of strings:

```python
scorer.score(
    ["This is a sentence.", "This is a sentence.", "This is another sentence."],
    ["This is another sentence.", "This sentence is completely unrelated.", "This is another sentence."],
)
# [0.4519273529250307, 0.13127038689469997, 1.0000000000000102]
```

The sentences in the first list are compared element-wise to the sentences in the second list.

The default batch size is 8.
An alternative batch size can be specified as follows (independently for translating and scoring):

```python
scorer.score_direct(
    a, b, a_lang="en", b_lang="en",
    score_kwargs={"batch_size": 16}
)

scorer.score_cross_likelihood(
    a, b,
    translate_kwargs={"batch_size": 16},
    score_kwargs={"batch_size": 16}
)
```

#### Different NMT models
This library currently supports four NMT models:
- [`small100`](https://huggingface.co/alirezamsh/small100) by [Mohammadshahi et al. (2022)](https://arxiv.org/abs/2210.11621)
- [`m2m100_418M`](https://huggingface.co/facebook/m2m100_418M) and [`m2m100_1.2B`](https://huggingface.co/facebook/m2m100_1.2B) by [Fan et al. (2021)](https://www.jmlr.org/papers/volume22/20-1307/)
- [`prism`](https://github.com/thompsonb/prism) by [Thompson and Post (2020)](https://aclanthology.org/2020.emnlp-main.8/)

By default, the leanest model (`small100`) is loaded. The main results in the paper are based on the Prism model, which has some extra dependencies (see "Installation" above).

```python
scorer = NMTScorer("small100", device=None)  # default
scorer = NMTScorer("small100", device="cuda:0")  # Enable faster inference on GPU
scorer = NMTScorer("m2m100_418M", device="cuda:0")
scorer = NMTScorer("m2m100_1.2B", device="cuda:0")
scorer = NMTScorer("prism", device="cuda:0")
```

**Which model should I choose?**

The page (/experiments/results/summary.md) compares the models regarding their accuracy and latency.
- Generally, we recommend Prism because it tends to have the highest accuracy. Also, Prism's implementation currently translates up 10x faster on GPU than the other models do, so we highly recommend to use Prism for the measures that require translation (`score_pivot()` and `score_cross_likelihood()`).
- `small100` is 3.4x faster for `score_direct()` and has 94–98% of Prism's accuracy.

#### Enable caching of NMT output
It can make sense to cache the translations and scores if they are needed repeatedly, e.g. in reference-based evaluation.

```python
scorer.score_direct(
    a, b, a_lang="en", b_lang="en",
    score_kwargs={"use_cache": True}  # default: False
)

scorer.score_cross_likelihood(
    a, b,
    translate_kwargs={"use_cache": True},  # default: False
    score_kwargs={"use_cache": True}  # default: False
)
```

Activating this option will create an SQLite database in the ~/.cache directory. The directory can be overriden via the `NMTSCORE_CACHE` environment variable.

#### Print a version signature (à la [SacreBLEU](https://github.com/mjpost/sacrebleu))
```python
scorer.score(a, b, print_signature=True)
# NMTScore-cross|tgt-lang:en|model:facebook/m2m100_418M|normalized|both-directions|v0.1.0|hf4.17.0
```

### Direct usage of NMT models

The NMT models also provide a direct interface for translating and scoring.

```python
from nmtscore.models import load_translation_model

model = load_translation_model("prism")

model.translate("de", ["This is a test."])
# ["Das ist ein Test."]

model.score("de", ["This is a test."], ["Das ist ein Test."])
# [0.5148844122886658]
```

## Experiments
See [experiments/README.md](experiments/README.md)

## Citation
```bibtex
@inproceedings{vamvas-sennrich-2022-nmtscore,
       author = "Vamvas, Jannis and
                 Sennrich, Rico",
        title = "{NMTScore}: A Multilingual Analysis of Translation-based Text Similarity Measures",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
         year = "2022",
        month = dec,
      address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
       eprint = {2204.13692}
}
```

## License
- Code: MIT License
- Data: See data subdirectories

## Changelog
- v0.3.0
  - Implement the distilled [`small100`](https://huggingface.co/alirezamsh/small100) model by [Mohammadshahi et al. (2022)](https://arxiv.org/abs/2210.11621) and use this model by default.
  - Enable half-precision inference for `m2m100` models and `small100` by default; see (/experiments/results/summary.md) for benchmark results

- v0.2.0
  - Bugfix: Provide source language to `m2m100` models (#2). The fix is backwards-compatible but a warning is now raised if `m2m100` is used without specifying the input language.
