import io
import os
import shutil
import unittest
from pathlib import Path
from unittest import TestCase, mock

from nmtscore import NMTScorer
from nmtscore.models import load_translation_model


class ReadmeTestCase(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.cache_dir = Path.home() / ".cache" / "test_nmtscore"
        os.environ["NMTSCORE_CACHE"] = str(cls.cache_dir)
        shutil.rmtree(cls.cache_dir, ignore_errors=True)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.cache_dir, ignore_errors=True)

    def test_nmtscorer(self):
        scorer = NMTScorer()
        score = scorer.score("This is a sentence.", "This is another sentence.")
        self.assertAlmostEqual(0.4677300455046415, score, places=4)

    def test_batch_processing(self):
        scorer = NMTScorer()
        scores = scorer.score(
            ["This is a sentence.", "This is a sentence.", "This is another sentence."],
            ["This is another sentence.", "This sentence is completely unrelated.", "This is another sentence."],
        )
        self.assertEqual(3, len(scores))
        self.assertAlmostEqual(0.46772973967003206, scores[0], places=4)
        self.assertAlmostEqual(0.15306852595255185, scores[1], places=4)
        self.assertAlmostEqual(1.0, scores[2], places=4)

    def test_different_similarity_measures(self):
        scorer = NMTScorer()
        a = "This is a sentence."
        b = "This is another sentence."
        score = scorer.score_cross_likelihood(a, b, tgt_lang="en", normalize=True, both_directions=True)
        self.assertAlmostEqual(0.4677300455046415, score, places=4)
        score = scorer.score_direct(a, b, a_lang="en", b_lang="en", normalize=True, both_directions=True)
        self.assertAlmostEqual(0.4677300455046415, score, places=4)
        score = scorer.score_pivot(a, b, a_lang="en", b_lang="en", pivot_lang="en", normalize=True, both_directions=True)
        self.assertAlmostEqual(0.4677300455046415, score, places=4)

    @unittest.skipIf(os.getenv("SKIP_SLOW_TESTS", False), "Slow")
    def test_different_nmt_models(self):
        scorer = NMTScorer("m2m100_418M", device=None)
        scorer = NMTScorer("m2m100_1.2B", device=None)
        scorer = NMTScorer("prism", device=None)

    def test_batch_size(self):
        scorer = NMTScorer()
        a = "This is a sentence."
        b = "This is another sentence."
        score = scorer.score_cross_likelihood(a, b, translate_kwargs={"batch_size": 16}, score_kwargs={"batch_size": 16})
        self.assertAlmostEqual(0.4677300455046415, score, places=4)
        score = scorer.score_direct(a, b, a_lang="en", b_lang="en", score_kwargs={"batch_size": 16})
        self.assertAlmostEqual(0.4677300455046415, score, places=4)

    def test_caching(self):
        scorer = NMTScorer()
        a = "This is a sentence."
        b = "This is another sentence."
        score = scorer.score_cross_likelihood(a, b, translate_kwargs={"use_cache": True}, score_kwargs={"use_cache": True})
        self.assertAlmostEqual(0.4677300455046415, score, places=4)
        score = scorer.score_direct(a, b, a_lang="en", b_lang="en", score_kwargs={"use_cache": True})
        self.assertAlmostEqual(0.4677300455046415, score, places=4)

    @mock.patch('sys.stdout', new_callable=io.StringIO)
    def test_version_signature(self, mock_stdout):
        scorer = NMTScorer()
        a = "This is a sentence."
        b = "This is another sentence."
        score = scorer.score(a, b, print_signature=True)
        self.assertIn("NMTScore-cross|tgt-lang:en|model:alirezamsh/small100|normalized|both-directions", mock_stdout.getvalue())

    def test_nmt_models(self):
        model = load_translation_model("small100")
        translations = model.translate("de", ["This is a test."], src_lang="en")
        self.assertEqual(["Das ist ein Test."], translations)
        scores = model.score("de", ["This is a test."], ["Das ist ein Test."], src_lang="en")
        self.assertAlmostEqual(0.8293135166168213, scores[0], places=4)
