import logging
import warnings
from typing import Union, List, Optional

import numpy as np

from nmtscore.models import TranslationModel, load_translation_model


class NMTScorer:

    def __init__(self,
                 model: Union[TranslationModel, str] = "small100",
                 device=None,
                 model_kwargs: dict = None,
                 ):
        model_kwargs = model_kwargs or {}
        if isinstance(model, str):
            if device is not None:
                model_kwargs["device"] = device
            self.model = load_translation_model(model, **model_kwargs)
        else:
            self.model = model

    def score(self,
              a: Union[str, List[str]],
              b: Union[str, List[str]],
              normalize: bool = True,
              both_directions: bool = True,
              print_signature: bool = False,
              **kwargs,
              ) -> Union[float, List[float]]:
        """
        Shortcut for score_cross_likelihood with English as target language.
        :param a: A sentence or list of sentences. If :param: both_directions is False this is the sentence of which
         the likelihood is estimated
        :param b: A sentence or list of sentences. If :param: both_directions is False this is the sentence that is
         translated
        :param normalize: Apply a normalization to the similarity score (default: True)
        :param both_directions: Return the average of score(a, b) and score(b, a) (default: True)
        :param print_signature: Print a version signature for the metric (default: False)
        :param kwargs
        :return: A float or list of floats
        """
        return self.score_cross_likelihood(a, b, tgt_lang="en", normalize=normalize, both_directions=both_directions,
                                           print_signature=print_signature, **kwargs)

    def score_direct(self,
                     a: Union[str, List[str]],
                     b: Union[str, List[str]],
                     a_lang: str,
                     b_lang: Optional[str] = None,
                     normalize: bool = True,
                     both_directions: bool = True,
                     print_signature: bool = False,
                     score_kwargs: dict = None,
                     ) -> Union[float, List[float]]:
        """
        Estimates the direct translation probability of A given B
        :param a: A sentence or list of sentences. If :param: both_directions is False this is the target sequence
        :param b: A sentence or list of sentences. If :param: both_directions is False this is the source sequence
        :param a_lang: The language code of sentence(s) :param: a
        :param b_lang: The language code of sentence(s) :param: b
        :param normalize: Apply a normalization to the similarity score (default: True)
        :param both_directions: Return the average of score(a, b) and score(b, a) (default: True)
        :param print_signature: Print a version signature for the metric (default: False)
        :param score_kwargs
        :return: A float or list of floats
        """
        if both_directions:
            assert b_lang is not None
        if self.model.requires_src_lang and b_lang is None:
            warnings.warn(f"NMT model {self.model} requires the src language. Assuming {a_lang}; override with `b_lang`")
            b_lang = a_lang

        scores = self.model.score(
            src_lang=b_lang,
            tgt_lang=a_lang,
            source_sentences=b,
            hypothesis_sentences=a,
            **(score_kwargs or {}),
        )
        if normalize:
            self_scores = self.score_direct(a, a, a_lang, a_lang, normalize=False, both_directions=False, score_kwargs=score_kwargs)
            scores = np.array(scores) / np.array(self_scores)
        if both_directions:
            reverse_scores = self.score_direct(b, a, b_lang, a_lang, normalize=normalize, both_directions=False, score_kwargs=score_kwargs)
            scores = self._average_scores(scores, reverse_scores)
        if print_signature:
            print(self._build_version_string("direct", a_lang=a_lang, b_lang=b_lang,
                                             normalized=normalize, both_directions=both_directions))
        return scores

    def score_pivot(self,
                    a: Union[str, List[str]],
                    b: Union[str, List[str]],
                    a_lang: str,
                    b_lang: Optional[str] = None,
                    pivot_lang: str = "en",
                    normalize: bool = True,
                    both_directions: bool = True,
                    print_signature: bool = False,
                    translate_kwargs: dict = None,
                    score_kwargs: dict = None,
                    ) -> Union[float, List[float]]:
        """
        Estimates the pivot translation probability of translating to A from B via a pivot language
        :param a: A sentence or list of sentences. If :param: both_directions is False this is the target sequence
        :param b: A sentence or list of sentences. If :param: both_directions is False this is the source sequence
        :param a_lang: The language code of sentence(s) :param: a
        :param b_lang: The language code of sentence(s) :param: b
        :param pivot_lang: The language code of the pivot language (default: "en")
        :param normalize: Apply a normalization to the similarity score (default: True)
        :param both_directions: Return the average of score(a, b) and score(b, a) (default: True)
        :param print_signature: Print a version signature for the metric (default: False)
        :param translate_kwargs
        :param score_kwargs
        :return: A float or list of floats
        """
        if both_directions:
            assert b_lang is not None
        if self.model.requires_src_lang and b_lang is None:
            warnings.warn(f"NMT model {self.model} requires the src language. Assuming {a_lang}; override with `b_lang`")
            b_lang = a_lang

        if isinstance(a, list) and len(a) >= 10:
            logging.info(f"Translating to pivot language {pivot_lang} ...")
        translations = self.model.translate(
            src_lang=b_lang,
            tgt_lang=pivot_lang,
            source_sentences=b,
            **(translate_kwargs or {}),
        )
        if isinstance(a, list) and len(a) >= 10:
            logging.info(f"Scoring sentences ...")
        scores = self.model.score(
            src_lang=pivot_lang,
            tgt_lang=a_lang,
            source_sentences=translations,
            hypothesis_sentences=a,
            **(score_kwargs or {}),
        )
        if normalize:
            self_scores = self.score_pivot(a, a, a_lang, a_lang, pivot_lang=pivot_lang, normalize=False, both_directions=False,
                                              translate_kwargs=translate_kwargs, score_kwargs=score_kwargs)
            scores = np.array(scores) / np.array(self_scores)
        if both_directions:
            reverse_scores = self.score_pivot(b, a, b_lang, a_lang, pivot_lang, normalize=normalize, both_directions=False,
                                              translate_kwargs=translate_kwargs, score_kwargs=score_kwargs)
            scores = self._average_scores(scores, reverse_scores)
        if print_signature:
            print(self._build_version_string("pivot", normalized=normalize, both_directions=both_directions,
                                             pivot_lang=pivot_lang, a_lang=a_lang, b_lang=b_lang))
        return scores

    def score_cross_likelihood(self,
                               a: Union[str, List[str]],
                               b: Union[str, List[str]],
                               a_lang: Optional[str] = None,
                               b_lang: Optional[str] = None,
                               tgt_lang: str = "en",
                               normalize: bool = True,
                               both_directions: bool = True,
                               print_signature: bool = False,
                               translate_kwargs: dict = None,
                               score_kwargs: dict = None,
                               ) -> Union[float, List[float]]:
        """
        Estimates the likelihood that a translation of B into a target language could also be a translation of A.
        :param a: A sentence or list of sentences. If :param: both_directions is False this is the sentence of which
         the likelhood is estimated
        :param b: A sentence or list of sentences. If :param: both_directions is False this is the sentence that is
         translated
        :param tgt_lang: The language code of the target language (default: "en")
        :param a_lang: The language code of A (default: None). Not needed for some multilingual models
        :param b_lang: The language code of B (default: a_lang). Not needed for some multilingual models
        :param normalize: Apply a normalization to the similarity score (default: True)
        :param both_directions: Return the average of score(a, b) and score(b, a) (default: True)
        :param print_signature: Print a version signature for the metric (default: False)
        :param translate_kwargs
        :param score_kwargs
        :return: A float or list of floats
        """
        if self.model.requires_src_lang and (a_lang is None or b_lang is None):
            warnings.warn(f"NMT model {self.model} requires the input languages. Assuming 'en' for unspecified languages; "
                          f"override with `a_lang` and `b_lang`")
            a_lang = a_lang or "en"
            b_lang = b_lang or a_lang
        if isinstance(a, list) and len(a) >= 10:
            logging.info(f"Translating to target language {tgt_lang} ...")
        translations_scores = self.model.translate(
            src_lang=b_lang,
            tgt_lang=tgt_lang,
            source_sentences=b,
            return_score=True,
            **(translate_kwargs or {}),
        )
        translations = [t[0] for t in translations_scores] if isinstance(translations_scores, list) else translations_scores[0]
        if isinstance(a, list) and len(a) >= 10:
            logging.info(f"Scoring sentences ...")
        scores = self.model.score(
            src_lang=a_lang,
            tgt_lang=tgt_lang,
            source_sentences=a,
            hypothesis_sentences=translations,
            **(score_kwargs or {}),
        )
        if normalize:
            translation_scores = [t[1] for t in translations_scores] if isinstance(translations_scores, list) else translations_scores[1]
            # Not every translation model supports returning a score with the translation yet, so the score might
            # have to be computed separately
            no_scores_yet = translation_scores[0] is None if isinstance(translation_scores, list) else translation_scores is None
            if no_scores_yet:
                translation_scores = self.model.score(
                    src_lang=b_lang,
                    tgt_lang=tgt_lang,
                    source_sentences=b,
                    hypothesis_sentences=translations,
                    **(score_kwargs or {}),
                )
            scores = np.array(scores) / np.array(translation_scores)
        if both_directions:
            reverse_scores = self.score_cross_likelihood(b, a, b_lang, a_lang, tgt_lang=tgt_lang,
                                                    normalize=normalize, both_directions=False,
                                                    translate_kwargs=translate_kwargs, score_kwargs=score_kwargs)
            scores = self._average_scores(scores, reverse_scores)
        if print_signature:
            print(self._build_version_string("cross", normalized=normalize, both_directions=both_directions,
                                             tgt_lang=tgt_lang, a_lang=a_lang, b_lang=b_lang))
        return scores

    def _average_scores(self,
                        scores1: Union[float, List[float]],
                        scores2: Union[float, List[float]],
                        ) -> Union[float, List[float]]:
        average_scores = (np.array(scores1) + np.array(scores2)) / 2
        return average_scores.tolist()

    def _build_version_string(self,
                              type: str,
                              normalized: bool,
                              both_directions: bool,
                              tgt_lang: str = None,
                              pivot_lang: str = None,
                              a_lang: str = None,
                              b_lang: str = None,
                              ) -> str:
        import nmtscore
        import transformers
        return f"NMTScore-{type}|" \
               f"{f'tgt-lang:{tgt_lang}|' if tgt_lang is not None else ''}" \
               f"{f'pivot-lang:{pivot_lang}|' if pivot_lang is not None else ''}" \
               f"{f'a-lang:{a_lang}|' if a_lang is not None else ''}" \
               f"{f'b-lang:{b_lang}|' if b_lang is not None else ''}" \
               f"model:{self.model}|" \
               f"{'normalized' if normalized else 'unnormalized'}|" \
               f"{'both-directions' if both_directions else 'single-direction'}|" \
               f"v{nmtscore.__version__}|" \
               f"hf{transformers.__version__}"
