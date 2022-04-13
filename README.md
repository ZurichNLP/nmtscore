# NMTScore
A library of translation-based text similarity measures.

The measures are further described in the paper "NMTScore: A Multilingual Analysis of Translation-based Text Similarity Measures".

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
This library currently supports three NMT models:
- [m2m100_418M](https://huggingface.co/facebook/m2m100_418M) and [m2m100_1.2B](https://huggingface.co/facebook/m2m100_1.2B) by [Fan et al. (2021)](https://www.jmlr.org/papers/volume22/20-1307/)
- [Prism](https://github.com/thompsonb/prism) by [Thompson and Post (2020)](https://aclanthology.org/2020.emnlp-main.8/)

By default, the leanest model (m2m100_418M) is loaded. The main results in the paper are based on the Prism model.

```python
scorer = NMTScorer("m2m100_418M", device=None)  # default
scorer = NMTScorer("m2m100_1.2B", device=None)
scorer = NMTScorer("prism", device=None)
```

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

model = load_translation_model("m2m100_418M")

model.translate("de", ["This is a test."])
# ["Das ist ein Test."]

model.score("de", ["This is a test."], ["Das ist ein Test."])
# [0.5148844122886658]
```

## Experiments
See [experiments/README.md](experiments/README.md)

## Citation
TBA

## License
- Code: MIT License
- Data: See data subdirectories
