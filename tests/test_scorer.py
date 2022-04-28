from unittest import TestCase

from nmtscore.scorer import NMTScorer


class ScorerTestCase(TestCase):

    def setUp(self) -> None:
        self.scorer = NMTScorer("m2m100_418M")

    def test_score(self):
        self.assertGreater(
            self.scorer.score("This is a test.", "That is an attempt."),
            self.scorer.score("This is a test.", "A sentence completely unrelated"),
        )
        
    def test_score_direct__monolingual(self):
        self.assertGreater(
            self.scorer.score_direct("This is a test.", "That is an attempt.", "en", "en"),
            self.scorer.score_direct("This is a test.", "A sentence completely unrelated", "en", "en"),
        )
        self.assertAlmostEqual(
            self.scorer.score_direct("This is a test.", "That is an attempt.", "en", "en"),
            self.scorer.score_direct("That is an attempt.", "This is a test.", "en", "en"),
        )
        self.assertNotEqual(
            self.scorer.score_direct("This is a test.", "That is an attempt.", "en", "en", both_directions=False),
            self.scorer.score_direct("That is an attempt.", "This is a test.", "en", "en", both_directions=False),
        )
        self.assertGreaterEqual(
            self.scorer.score_direct("This is a test.", "That is an attempt.", "en", "en"),
            self.scorer.score_direct("This is a test.", "That is an attempt.", "en", "en", normalize=False),
        )

    def test_score_direct__crosslingual(self):
        self.assertGreater(
            self.scorer.score_direct("This is a test.", "Dies ist ein Test.", "en", "de"),
            self.scorer.score_direct("This is a test.", "Ein völlig anderer Satz.", "en", "de"),
        )
        self.assertAlmostEqual(
            self.scorer.score_direct("This is a test.", "Dies ist ein Test.", "en", "de"),
            self.scorer.score_direct("Dies ist ein Test.", "This is a test.", "de", "en"),
        )
        self.assertNotEqual(
            self.scorer.score_direct("This is a test.", "Dies ist ein Test.", "en", "de", both_directions=False),
            self.scorer.score_direct("Dies ist ein Test.", "This is a test.", "de", "en", both_directions=False),
        )
        self.assertGreaterEqual(
            self.scorer.score_direct("This is a test.", "Dies ist ein Test.", "en", "de"),
            self.scorer.score_direct("This is a test.", "Dies ist ein Test.", "en", "de", normalize=False),
        )

    def test_score_pivot__monolingual(self):
        self.assertGreater(
            self.scorer.score_pivot("This is a test.", "That is an attempt.", "en", "en"),
            self.scorer.score_pivot("This is a test.", "A sentence completely unrelated", "en", "en"),
        )
        self.assertAlmostEqual(
            self.scorer.score_pivot("This is a test.", "That is an attempt.", "en", "en"),
            self.scorer.score_pivot("That is an attempt.", "This is a test.", "en", "en"),
        )
        self.assertNotEqual(
            self.scorer.score_pivot("This is a test.", "That is an attempt.", "en", "en", both_directions=False),
            self.scorer.score_pivot("That is an attempt.", "This is a test.", "en", "en", both_directions=False),
        )
        self.assertGreaterEqual(
            self.scorer.score_pivot("This is a test.", "That is an attempt.", "en", "en"),
            self.scorer.score_pivot("This is a test.", "That is an attempt.", "en", "en", both_directions=False),
        )

    def test_score_pivot__crosslingual(self):
        self.assertGreater(
            self.scorer.score_pivot("This is a test.", "Dies ist ein Test.", "en", "de"),
            self.scorer.score_pivot("This is a test.", "Ein völlig anderer Satz.", "en", "de"),
        )
        self.assertAlmostEqual(
            self.scorer.score_pivot("This is a test.", "Dies ist ein Test.", "en", "de"),
            self.scorer.score_pivot("Dies ist ein Test.", "This is a test.", "de", "en"),
        )
        self.assertNotEqual(
            self.scorer.score_pivot("This is a test.", "Dies ist ein anderer Test.", "en", "de", both_directions=False),
            self.scorer.score_pivot("Dies ist ein anderer Test.", "This is a test.", "de", "en", both_directions=False),
        )
        self.assertGreaterEqual(
            self.scorer.score_pivot("This is a test.", "Dies ist ein Test.", "en", "de"),
            self.scorer.score_pivot("This is a test.", "Dies ist ein Test.", "en", "de", normalize=False),
        )

    def test_score_cross_likelihood__monolingual(self):
        self.assertGreater(
            self.scorer.score_cross_likelihood("This is a test.", "That is an attempt."),
            self.scorer.score_cross_likelihood("This is a test.", "A sentence completely unrelated"),
        )
        self.assertAlmostEqual(
            self.scorer.score_cross_likelihood("This is a test.", "That is an attempt."),
            self.scorer.score_cross_likelihood("That is an attempt.", "This is a test."),
        )
        self.assertNotEqual(
            self.scorer.score_cross_likelihood("This is a test.", "That is an attempt.", both_directions=False),
            self.scorer.score_cross_likelihood("That is an attempt.", "This is a test.", both_directions=False),
        )
        self.assertGreaterEqual(
            self.scorer.score_cross_likelihood("This is a test.", "That is an attempt."),
            self.scorer.score_cross_likelihood("This is a test.", "That is an attempt.", normalize=False),
        )

    def test_score_cross_likelihood__crosslingual(self):
        self.assertGreater(
            self.scorer.score_cross_likelihood("This is a test.", "Dies ist ein Test."),
            self.scorer.score_cross_likelihood("This is a test.", "Ein völlig anderer Satz."),
        )
        self.assertAlmostEqual(
            self.scorer.score_cross_likelihood("This is a test.", "Dies ist ein Test."),
            self.scorer.score_cross_likelihood("Dies ist ein Test.", "This is a test."),
        )
        self.assertNotEqual(
            self.scorer.score_cross_likelihood("This is a test.", "Dies ist ein Test.", both_directions=False),
            self.scorer.score_cross_likelihood("Dies ist ein Test.", "This is a test.", both_directions=False),
        )
        self.assertGreaterEqual(
            self.scorer.score_cross_likelihood("This is a test.", "Dies ist ein Test."),
            self.scorer.score_cross_likelihood("This is a test.", "Dies ist ein Test.", normalize=False),
        )
