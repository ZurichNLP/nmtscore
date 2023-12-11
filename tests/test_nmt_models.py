import os
import unittest
from unittest import TestCase

from nmtscore.models import load_translation_model


class NMTModelTestCase(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        raise NotImplementedError
    
    @property
    def lang_code_de(self):
        return "de"
    
    @property
    def lang_code_en(self):
        return "en"

    def test_translate(self):
        self.assertIn(self.model.translate(self.lang_code_de, "This is a test.", src_lang=self.lang_code_en), {
            "Dies ist ein Test.",
            "Das ist ein Test.",
        })

    def test_translate_score(self):
        translation, score = self.model.translate(self.lang_code_de, "This is a test.", return_score=True, src_lang=self.lang_code_en)
        self.assertIn(translation, {
            "Dies ist ein Test.",
            "Das ist ein Test.",
        })
        if score is None:
            return
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)
        self.assertAlmostEqual(score, self.model.score(self.lang_code_de, "This is a test.", translation, src_lang=self.lang_code_en), places=5)

    def test_translate_batched(self):
        translations = self.model.translate(self.lang_code_de, 8 * ["This is a test."], src_lang=self.lang_code_en)
        self.assertEqual(8, len(translations))
        self.assertEqual(1, len(set(translations)))
        self.assertIn(translations[0], {
            "Dies ist ein Test.",
            "Das ist ein Test.",
        })

    def test_score(self):
        scores = self.model.score(
            self.lang_code_de,
            src_lang = self.lang_code_en,
            source_sentences=(2 * ["This is a test."]),
            hypothesis_sentences=(["Dies ist ein Test.", "Diese Übersetzung ist komplett falsch."]),
        )
        self.assertIsInstance(scores[0], float)
        self.assertIsInstance(scores[1], float)
        scores = self.model.score(
            self.lang_code_de,
            src_lang = self.lang_code_en,
            source_sentences=(2 * ["This is a test."]),
            hypothesis_sentences=(["Diese Übersetzung ist komplett falsch.", "Dies ist ein Test."]),
        )
        self.assertLess(scores[0], scores[1])
        scores = self.model.score(
            self.lang_code_de,
            src_lang = self.lang_code_en,
            source_sentences=(2 * ["This is a test."]),
            hypothesis_sentences=(2 * ["Dies ist ein Test."]),
        )
        self.assertAlmostEqual(scores[0], scores[1], places=4)

    def test_score_batched(self):
        scores = self.model.score(
            self.lang_code_de,
            src_lang = self.lang_code_en,
            source_sentences=(4 * ["This is a test."]),
            hypothesis_sentences=(["Diese Übersetzung ist komplett falsch", "Dies ist ein Test.", "Dies ist ein Test.", "Dies ist ein Test."]),
            batch_size=2,
        )
        self.assertLess(scores[0], scores[1])
        self.assertAlmostEqual(scores[2], scores[1], places=4)
        self.assertAlmostEqual(scores[3], scores[1], places=4)

        scores = self.model.score(
            self.lang_code_de,
            src_lang = self.lang_code_en,
            source_sentences=(["This is a test.", "A translation that is completely wrong.", "This is a test.", "This is a test."]),
            hypothesis_sentences=(4 * ["Dies ist ein Test."]),
            batch_size=2,
        )
        self.assertLess(scores[1], scores[0])
        self.assertAlmostEqual(scores[2], scores[0], places=4)
        self.assertAlmostEqual(scores[3], scores[0], places=4)

        scores = self.model.score(
            self.lang_code_de,
            src_lang = self.lang_code_en,
            source_sentences=(4 * ["This is a test."]),
            hypothesis_sentences=(["Dies ist ein Test.", "Dies ist ein Test.", ".", "Dies ist ein Test."]),
            batch_size=2,
        )
        self.assertAlmostEqual(scores[1], scores[0], places=4)
        self.assertLess(scores[2], scores[0])
        self.assertAlmostEqual(scores[3], scores[0], places=4)

        scores = self.model.score(
            self.lang_code_de,
            src_lang = self.lang_code_en,
            source_sentences=(["This is a test.", "This is a test.", "This is a test.", "A translation that is completely wrong."]),
            hypothesis_sentences=(4 * ["Dies ist ein Test."]),
            batch_size=2,
        )
        self.assertAlmostEqual(scores[1], scores[0], places=4)
        self.assertAlmostEqual(scores[2], scores[0], places=4)
        self.assertLess(scores[3], scores[0])

    def test_translate_long_input(self):
        self.model.translate(self.lang_code_de, 100 * "This is a test. ", src_lang=self.lang_code_en)

    def test_score_long_input(self):
        self.model.score(self.lang_code_de, 100 * "This is a test. ", 100 * "Dies ist ein Test. ", src_lang=self.lang_code_en)


@unittest.skipIf(os.getenv("SKIP_SLOW_TESTS", False), "Slow")
class SmallM2M100TestCase(NMTModelTestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = load_translation_model("m2m100_418M")


@unittest.skipIf(os.getenv("SKIP_SLOW_TESTS", False), "Slow")
class LargeM2M100TestCase(NMTModelTestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = load_translation_model("m2m100_1.2B")


class SMALL100TestCase(NMTModelTestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = load_translation_model("small100")


@unittest.skipIf(os.getenv("SKIP_SLOW_TESTS", False), "Slow")
class PrismTestCase(NMTModelTestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = load_translation_model("prism")


@unittest.skipIf(os.getenv("SKIP_SLOW_TESTS", False), "Slow")
class SmallDistilledNLLB200TestCase(NMTModelTestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = load_translation_model("nllb-200-distilled-600M")

    @property
    def lang_code_de(self):
        return "deu_Latn"

    @property
    def lang_code_en(self):
        return "eng_Latn"


@unittest.skipIf(os.getenv("SKIP_SLOW_TESTS", False), "Slow")
class SmallNLLB200TestCase(NMTModelTestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = load_translation_model("nllb-200-1.3B")

    @property
    def lang_code_de(self):
        return "deu_Latn"

    @property
    def lang_code_en(self):
        return "eng_Latn"


# https://stackoverflow.com/a/43353680/3902795
del NMTModelTestCase
