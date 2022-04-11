import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Set, Dict, Tuple, Optional
import xml.etree.ElementTree as ET

import numpy as np

from experiments.paraphrase_tasks import ParaphraseClassificationSample


@dataclass
class ETPCSample(ParaphraseClassificationSample):
    id: int
    sentence1_tokenized: str  # For matching with Huggingface dataset
    sentence2_tokenized: str
    phenomena: Set[str]
    rouge: Optional[float] = None
    has_high_rouge: Optional[bool] = None
    is_long_sentence: Optional[bool] = None

    @property
    def is_positive(self) -> bool:
        return self.label == 1

    @property
    def is_negative(self) -> bool:
        return self.label == 0

    @property
    def has_any_substitution(self) -> bool:
        return any(phenomenon in self.phenomena for phenomenon in [
            "Same Polarity Substitution (habitual)",
            "Same Polarity Substitution (contextual)",
            "Same Polarity Substitution (named ent.)",
            "Opposite polarity substitution (habitual)",
            "Opposite polarity substitution (contextual)",
        ])

    @property
    def has_named_entity_variation(self) -> bool:
        return any(phenomenon in self.phenomena for phenomenon in [
            "Same Polarity Substitution (named ent.)",
        ])

    @property
    def has_spelling_variation(self) -> bool:
        return any(phenomenon in self.phenomena for phenomenon in [
            "Spelling changes",
            "Change of format",
            "Punctuation changes",
        ])

    @property
    def has_word_order_variation(self) -> bool:
        return any(phenomenon in self.phenomena for phenomenon in [
            "Change of order",
        ])

    @property
    def has_addition_or_deletion(self) -> bool:
        return any(phenomenon in self.phenomena for phenomenon in [
            "Addition/Deletion",
        ])

    @property
    def has_low_rouge(self) -> Optional[bool]:
        if self.has_high_rouge is None:
            return None
        return not self.has_high_rouge

    @property
    def num_tokens(self) -> float:
        return (len(self.sentence1_tokenized.split()) + len(self.sentence2_tokenized.split())) / 2


class ETPCData:

    phenomena = (
        "Inflectional Changes",
        "Modal Verb Changes",
        "Derivational Changes",
        "Spelling changes",
        "Same Polarity Substitution (habitual)",
        "Same Polarity Substitution (contextual)",
        "Same Polarity Substitution (named ent.)",
        "Change of format",
        "Opposite polarity substitution (habitual)",
        "Opposite polarity substitution (contextual)",
        "Synthetic/analytic substitution",
        "Converse substitution",
        "Diathesis alternation",
        "Negation switching",
        "Ellipsis",
        "Coordination changes",
        "Subordination and nesting changes",
        "Punctuation changes",
        "Direct/indirect style alternations",
        "Addition/Deletion",
        "Syntax/discourse structure changes",
        "Change of order",
        "Contains negation",
        "Semantic based",
        "Identity",
        "Entailment",
    )

    def __init__(self):
        self.data_dir = Path(__file__).parent / "etpc"
        self.pairs_path = self.data_dir / "text_pairs.xml"

        self.skipped_sentence_pairs: Set[Tuple[str, str]] = set()
        self.samples: Dict[int, ETPCSample] = dict()
        self.tokenized2raw: Dict[Tuple[str, str], Tuple[str, str]] = dict()

        tree = ET.parse(self.pairs_path)
        root = tree.getroot()
        for text_pair in root.iter("text_pair"):
            sample = ETPCSample(
                id=int(text_pair.find("pair_id").text),
                sentence1=text_pair.find("sent1_raw").text.strip(),
                sentence2=text_pair.find("sent2_raw").text.strip(),
                sentence1_tokenized=text_pair.find("sent1_tokenized").text.strip().replace("``", '"').replace("''", '"'),
                sentence2_tokenized=text_pair.find("sent2_tokenized").text.strip().replace("``", '"').replace("''", '"'),
                label=int(text_pair.find("mrpc_label").text),
                phenomena=set(),
            )
            self.tokenized2raw[(sample.sentence1_tokenized, sample.sentence2_tokenized)] = (sample.sentence1, sample.sentence2)
            self.tokenized2raw[(sample.sentence1_tokenized.replace("does n't", "doesn 't"), sample.sentence2_tokenized.replace("does n't", "doesn 't"))] = (sample.sentence1, sample.sentence2)
            # Skip sample with a disagreement between the two datasets
            if not text_pair.find("mrpc_label").text == text_pair.find("etpc_label").text:
                # Store tokenized sentences so that they can be matched with the Huggingface dataset
                self.skipped_sentence_pairs.add((sample.sentence1_tokenized, sample.sentence2_tokenized))
                continue
            self.samples[sample.id] = sample

        phenomena_xml_roots = (ET.parse(filepath).getroot() for filepath in [
            self.data_dir / "textual_paraphrases.xml",
            self.data_dir / "textual_np_neg.xml",
            self.data_dir / "textual_np_pos.xml",
        ])
        for relation in itertools.chain.from_iterable((root.iter("relation")) for root in phenomena_xml_roots):
            sample_id = int(relation.find("pair_id").text)
            type_name = relation.find("type_name").text.strip()
            if sample_id not in self.samples:
                continue
            self.samples[sample_id].phenomena.add(type_name)

        # Add "Contains negation" as an additional feature, as did Kovatchev et al. (2019)
        # tree = ET.parse(self.negation_path)
        # root = tree.getroot()
        # for relation in root.iter("relation"):
        #     sample_id = int(relation.find("pair_id").text)
        #     if sample_id not in self.samples:
        #         continue
        #     self.samples[sample_id].phenomena.add("Contains negation")

        self.annotate_sentence_length()

    def annotate_sentence_length(self):
        lenghts = [sample.num_tokens for sample in self.samples.values()]
        threshold = np.percentile(lenghts, 50)
        for i, sample in enumerate(self.samples.values()):
            sample.is_long_sentence = sample.num_tokens >= threshold

    def annotate_word_overlap(self):
        """
        Calculate ROUGE-1 and annotatate whether a sample is in the top half of bottom half
        """
        from sacrerouge.metrics import Rouge
        rouge = Rouge(1)
        rouge_scores = [
            score["rouge-1"]["f1"]
            for score in rouge.score_all(
                [sample.sentence1 for sample in self.samples.values()],
                [[sample.sentence2] for sample in self.samples.values()]
            )
        ]
        median = np.median(rouge_scores)
        for i, sample in enumerate(self.samples.values()):
            sample.rouge = rouge_scores[i]
            sample.has_high_rouge = sample.rouge >= median
