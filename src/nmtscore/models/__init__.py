import json
import os
import warnings

from sqlitedict import SqliteDict
from pathlib import Path
from typing import List, Union, Tuple


class TranslationModel:

    def __str__(self):
        raise NotImplementedError

    def translate(self,
                  tgt_lang: str,
                  source_sentences: Union[str, List[str]],
                  src_lang: str = None,
                  return_score: bool = False,
                  batch_size: int = 8,
                  use_cache: bool = False,
                  **kwargs,
                  ) -> Union[str, List[str], Tuple[str, float], List[Tuple[str, float]]]:
        """
        :param tgt_lang: Language code of the target language
        :param source_sentences: A sentence or list of sentences
        :param src_lang: Language code of the source language (not needed for some multilingual models)
        :param return score: If true, return a tuple where the second element is sequence-level score of the translation
        :param batch_size
        :param use_cache
        :param kwargs
        :return: A sentence or list of sentences
        """
        if isinstance(source_sentences, str):
            source_sentences_list = [source_sentences]
        elif isinstance(source_sentences, list):
            source_sentences_list = source_sentences
        else:
            raise ValueError

        if use_cache:
            if kwargs:
                raise NotImplementedError
            cached_translations_list = []
            with self.load_cache() as cache:
                for source_sentence in source_sentences_list:
                    translation = cache.get(f"{(src_lang + '_') if src_lang is not None else ''}{tgt_lang}_"
                                            f"translate{'_score' if return_score else ''}_{source_sentence}", None)
                    cached_translations_list.append(translation)
            full_source_sentences_list = source_sentences_list
            source_sentences_list = [
                source_sentence for source_sentence, cached_translation
                in zip(full_source_sentences_list, cached_translations_list)
                if cached_translation is None
            ]

        self._set_tgt_lang(tgt_lang)
        if self.requires_src_lang:
            if src_lang is None:
                warnings.warn(f"NMT model {self} requires the src language. Assuming 'en'; override with `src_lang`")
                src_lang = "en"
            self._set_src_lang(src_lang)
        if not source_sentences_list:
            translations_list = []
        else:
            translations_list = self._translate(source_sentences_list, return_score, batch_size, **kwargs)
        assert len(translations_list) == len(source_sentences_list)

        if use_cache:
            cache_update = dict()
            for i, cached_translation in enumerate(cached_translations_list):
                if cached_translation is not None:
                    translations_list.insert(i, cached_translation)
                else:
                    cache_update[f"{(src_lang + '_') if src_lang is not None else ''}{tgt_lang}_" \
                                 f"translate{'_score' if return_score else ''}_" \
                                 f"{full_source_sentences_list[i]}"] = translations_list[i]
            if cache_update:
                with self.load_cache() as cache:
                    cache.update(cache_update)
                    cache.commit()

        if isinstance(source_sentences, str):
            translations = translations_list[0]
        else:
            translations = translations_list
        return translations

    def score(self,
              tgt_lang: str,
              source_sentences: Union[str, List[str]],
              hypothesis_sentences: Union[str, List[str]],
              src_lang: str = None,
              batch_size: int = 8,
              use_cache: bool = False,
              **kwargs,
              ) -> Union[float, List[float]]:
        """
        :param tgt_lang: Language code of the target language
        :param source_sentences: A sentence or list of sentences
        :param hypothesis_sentences: A sentence or list of sentences
        :param src_lang: Language code of the source language (not needed for some multilingual models)
        :param batch_size
        :param use_cache
        :param kwargs
        :return: A float or list of floats
        """
        assert type(source_sentences) == type(hypothesis_sentences)
        if isinstance(source_sentences, str):
            source_sentences_list = [source_sentences]
            hypothesis_sentences_list = [hypothesis_sentences]
        elif isinstance(source_sentences, list):
            assert len(source_sentences) == len(hypothesis_sentences)
            source_sentences_list = source_sentences
            hypothesis_sentences_list = hypothesis_sentences
        else:
            raise ValueError

        if use_cache:
            if kwargs:
                raise NotImplementedError
            cached_scores_list = []
            with self.load_cache() as cache:
                for source_sentence, hypothesis_sentence in zip(source_sentences_list, hypothesis_sentences_list):
                    score = cache.get(f"{(src_lang + '_') if src_lang is not None else ''}{tgt_lang}_"
                                      f"score_{source_sentence}_{hypothesis_sentence}", None)
                    cached_scores_list.append(score)
            full_source_sentences_list = source_sentences_list
            source_sentences_list = [
                source_sentence for source_sentence, cached_score
                in zip(full_source_sentences_list, cached_scores_list)
                if cached_score is None
            ]
            full_hypothesis_sentences_list = hypothesis_sentences_list
            hypothesis_sentences_list = [
                hypothesis_sentence for hypothesis_sentence, cached_score
                in zip(full_hypothesis_sentences_list, cached_scores_list)
                if cached_score is None
            ]

        self._set_tgt_lang(tgt_lang)
        if self.requires_src_lang:
            if src_lang is None:
                warnings.warn(f"NMT model {self} requires the src language. Assuming 'en'; override with `src_lang`")
                src_lang = "en"
            self._set_src_lang(src_lang)
        if not source_sentences_list:
            scores_list = []
        else:
            scores_list = self._score(source_sentences_list, hypothesis_sentences_list, batch_size, **kwargs)
        assert len(scores_list) == len(source_sentences_list)

        if use_cache:
            cache_update = dict()
            for i, cached_score in enumerate(cached_scores_list):
                if cached_score is not None:
                    scores_list.insert(i, cached_score)
                else:
                    cache_update[f"{(src_lang + '_') if src_lang is not None else ''}{tgt_lang}_" \
                                 f"score_{full_source_sentences_list[i]}_" \
                                 f"{full_hypothesis_sentences_list[i]}"] = scores_list[i]
            if cache_update:
                with self.load_cache() as cache:
                    cache.update(cache_update)
                    cache.commit()

        if isinstance(source_sentences, str):
            scores = scores_list[0]
        else:
            scores = scores_list
        return scores

    @property
    def requires_src_lang(self) -> bool:
        """
        Boolean indicating whether the model requires the source language to be specified
        """
        raise NotImplementedError

    def _set_src_lang(self, src_lang: str):
        raise NotImplementedError

    def _set_tgt_lang(self, tgt_lang: str):
        raise NotImplementedError

    def _translate(self,
                   source_sentences: List[str],
                   return_score: bool = False,
                   batch_size: int = 8,
                   **kwargs,
                   ) -> Union[List[str], List[Tuple[str, float]]]:
        raise NotImplementedError

    def _score(self,
               source_sentences: List[str],
               hypothesis_sentences: List[str],
               batch_size: int = 8,
               **kwargs,
               ) -> List[float]:
        raise NotImplementedError

    @property
    def cache_path(self) -> Path:
        """
        :return: Path of the SQLite database where the translations and scores are cached
        """
        cache_dir = Path(os.getenv("NMTSCORE_CACHE", Path.home() / ".cache" / "nmtscore"))
        if not cache_dir.exists():
            os.mkdir(cache_dir)
        return cache_dir / (str(self).replace("/", "_") + ".sqlite")

    def load_cache(self) -> SqliteDict:
        """
        :return: A connection to the SQLite database where the translations and scores are cached
        """
        return SqliteDict(self.cache_path, timeout=15, encode=json.dumps, decode=json.loads)


def load_translation_model(name: str, **kwargs) -> TranslationModel:
    """
    Convenience function to load a :class: TranslationModel using a shorthand name of the model
    """
    if name == "m2m100_418M":
        from nmtscore.models.m2m100 import M2M100Model
        translation_model = M2M100Model(model_name_or_path="facebook/m2m100_418M", **kwargs)
    elif name == "m2m100_1.2B":
        from nmtscore.models.m2m100 import M2M100Model
        translation_model = M2M100Model(model_name_or_path="facebook/m2m100_1.2B", **kwargs)
    elif name == "small100":
        from nmtscore.models.small100 import SMALL100Model
        translation_model = SMALL100Model(**kwargs)
    elif name == "prism":
        try:
            import fairseq
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(e.msg + "\nUsing prism requires extra dependencies. Install with "
                                              "`pip install nmtscore[prism]`")
        from nmtscore.models.prism import PrismModel
        translation_model = PrismModel(**kwargs)
    else:
        raise NotImplementedError
    return translation_model
