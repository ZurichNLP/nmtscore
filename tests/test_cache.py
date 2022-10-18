import os
import shutil
from pathlib import Path
from typing import List, Union, Tuple
from unittest import TestCase

import numpy as np

from nmtscore.models import TranslationModel


class CopyTranslationModel(TranslationModel):

    def __str__(self):
        return "copy-translation-model"

    @property
    def requires_src_lang(self) -> bool:
        return False

    def _set_tgt_lang(self, tgt_lang: str):
        pass

    def _translate(self,
                   source_sentences: List[str],
                   return_score: bool = False,
                   batch_size: int = 8,
                   **kwargs,
                   ) -> Union[List[str], List[Tuple[str, float]]]:
        if return_score:
            return list(zip(source_sentences, (np.arange(len(source_sentences)) / len(source_sentences)).tolist()))
        return source_sentences

    def _score(self,
               source_sentences: List[str],
               hypothesis_sentences: List[str],
               batch_size: int = 8,
               **kwargs,
               ) -> List[float]:
        return (np.arange(len(source_sentences)) / len(source_sentences)).tolist()


class CacheTestCase(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.cache_dir = Path.home() / ".cache" / "test_nmtscore"
        os.environ["NMTSCORE_CACHE"] = str(cls.cache_dir)
        shutil.rmtree(cls.cache_dir, ignore_errors=True)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.cache_dir, ignore_errors=True)

    def setUp(self) -> None:
        self.model = CopyTranslationModel()

    def test_translate(self):
        source_sentences = [f"{i}. This is a test." for i in range(10)]
        translations1 = self.model.translate(
            tgt_lang="de",
            source_sentences=source_sentences,
            use_cache=False,
        )
        self.assertEqual(len(source_sentences), len(set(source_sentences)))
        translations2 = self.model.translate(
            tgt_lang="de",
            source_sentences=source_sentences,
            use_cache=True,
        )
        self.assertListEqual(translations1, translations2)
        self.assertTrue(self.model.cache_path.exists())
        translations3 = self.model.translate(
            tgt_lang="de",
            source_sentences=list(reversed(source_sentences)),
            use_cache=True,
        )
        self.assertListEqual(translations2, list(reversed(translations3)))
        source_sentences.append("This is an additional test.")
        translations4 = self.model.translate(
            tgt_lang="de",
            source_sentences=source_sentences,
            use_cache=True,
        )
        self.assertListEqual(translations2, translations4[:-1])

    def test_score(self):
        source_sentences = [f"{i}. This is a test." for i in range(10)]
        hypothesis_sentences = [f"{i}. Dies ist ein test." for i in range(10)]
        scores1 = self.model.score(
            tgt_lang="de",
            source_sentences=source_sentences,
            hypothesis_sentences=hypothesis_sentences,
            use_cache=False,
        )
        self.assertEqual(len(scores1), len(set(scores1)))
        scores2 = self.model.score(
            tgt_lang="de",
            source_sentences=source_sentences,
            hypothesis_sentences=hypothesis_sentences,
            use_cache=True,
        )
        self.assertListEqual(scores1, scores2)
        self.assertTrue(self.model.cache_path.exists())
        scores3 = self.model.score(
            tgt_lang="de",
            source_sentences=source_sentences,
            hypothesis_sentences=hypothesis_sentences,
            use_cache=True,
        )
        self.assertListEqual(scores2, scores3)
