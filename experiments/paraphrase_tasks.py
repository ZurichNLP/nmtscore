import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from datasets import load_dataset
from sacrerouge.data import MetricsDict
from sacrerouge.metrics import Metric
from sklearn.metrics import roc_curve, accuracy_score


@dataclass
class ParaphraseClassificationSample:
    sentence1: str
    sentence2: str
    label: float

    @property
    def num_chars(self) -> float:
        return (len(self.sentence1) + len(self.sentence2)) / 2

    @property
    def char_diff(self) -> float:
        return abs(len(self.sentence1) - len(self.sentence2)) / self.num_chars


@dataclass
class ParaphraseClassificationResult:
    samples: List[ParaphraseClassificationSample]
    scores: List[float]
    threshold: Optional[float] = None

    @property
    def labels(self) -> List[float]:
        return [sample.label for sample in self.samples]

    @property
    def threshold_predictions(self) -> Optional[List[float]]:
        if self.threshold is None:
            return None
        return [score > self.threshold for score in self.scores]

    def __len__(self):
        return len(self.samples)


class ParaphraseClassificationTask:

    def __str__(self):
        raise NotImplementedError

    def get_samples(self) -> List[ParaphraseClassificationSample]:
        raise NotImplementedError

    def evaluate(self, metric: Metric, metric_names: List[str]) -> List[ParaphraseClassificationResult]:
        samples = self.get_samples()
        metric_output: List[MetricsDict] = metric.score_all(
            summaries=[sample.sentence1 for sample in samples],
            references_list=[[sample.sentence2] for sample in samples],
        )
        assert len(metric_output) == len(samples)
        flattened_output = [d.flatten_keys() for d in metric_output]
        results: List[ParaphraseClassificationResult] = []
        for metric_name in metric_names:
            scores = [d[metric_name] for d in flattened_output]
            # auc = sklearn.metrics.roc_auc_score([sample.label for sample in samples], scores)
            # print(f"{metric_name} AUC: {auc}")
            results.append(ParaphraseClassificationResult(
                samples=samples,
                scores=scores,
            ))
        return results

    def print_statistics(self) -> None:
        samples = list(self.get_samples())
        print("Number of samples:", len(samples))
        print("Number of positive samples:", len([sample for sample in samples if sample.label]))
        print("Number of negative samples:", len([sample for sample in samples if not sample.label]))
        print("Average number of characters:", np.mean([(len(sample.sentence1) + len(sample.sentence2)) / 2 for sample in samples if not sample.label]))


class ParaphraseClassificationTaskWithThresholding:
    """
    Binary classification threshold is determined on a validation set; then accuracy returned on a test set
    """

    def __init__(self, validation_task: ParaphraseClassificationTask, evaluation_task: ParaphraseClassificationTask):
        self.validation_task = validation_task
        self.evaluation_task = evaluation_task

    def evaluate(self, metric: Metric, metric_names: List[str], **kwargs) -> List[ParaphraseClassificationResult]:
        validation_results = self.validation_task.evaluate(metric, metric_names, **kwargs)
        evaluation_results = self.evaluation_task.evaluate(metric, metric_names, **kwargs)
        for i, validation_result in enumerate(validation_results):
            _, _, thresholds = roc_curve(
                y_true=validation_result.labels,
                y_score=validation_result.scores,
            )
            accuracy_scores = []
            for threshold in thresholds:
                accuracy_scores.append(accuracy_score(
                    validation_result.labels,
                    [score > threshold for score in validation_result.scores]
                ))
            optimal_threshold = thresholds[np.array(accuracy_scores).argmax()]
            # print(f"Best threshold based on validation data for metric {metric_names[i]}: {optimal_threshold}")
            # print(f"Validation accuracy: {max(accuracy_scores)}")
            evaluation_results[i].threshold = optimal_threshold
            evaluation_accuracy = accuracy_score(
                evaluation_results[i].labels,
                evaluation_results[i].threshold_predictions,
            )
            # print(f"=> Evaluation accuracy for metric {metric_names[i]}: {evaluation_accuracy}")
        return evaluation_results


class MRPCTask(ParaphraseClassificationTask):
    language = "en"

    def __init__(self, split="validation", skip_disagreements_with_etpc=True):
        # Need this dataset for train–validation–test split
        self.huggingface_dataset = load_dataset('glue', 'mrpc', split=split)
        from experiments.data.etpc_data import ETPCData
        self.etpc_dataset = ETPCData()  # Need this dataset for untokenized sentences and to skip samples with label disagreement
        self.skip_disagreements_with_etpc = skip_disagreements_with_etpc

    def __str__(self):
        return "mrpc"

    def get_samples(self) -> List[ParaphraseClassificationSample]:
        returned_samples = []
        for huggingface_sample in self.huggingface_dataset:
            sentence_pair = (huggingface_sample["sentence1"].strip(), huggingface_sample["sentence2"].strip())
            if self.skip_disagreements_with_etpc:
                if sentence_pair in self.etpc_dataset.skipped_sentence_pairs:
                    continue
            try:
                sentence1, sentence2 = self.etpc_dataset.tokenized2raw[sentence_pair]
            except KeyError:
                sentence1, sentence2 = sentence_pair  # Use tokenized version
            returned_samples.append(ParaphraseClassificationSample(
                sentence1=sentence1,
                sentence2=sentence2,
                label=huggingface_sample["label"]
            ))
        return returned_samples


class RussianParaphraseClassificationTask(ParaphraseClassificationTask):
    language = "ru"

    def __init__(self, variant="task2"):
        assert variant in {"task2", "task2b"}
        self.variant = variant
        self.data_path = Path(__file__).parent / "data" / "russian_paraphrases" / "paraphrases_gold.xml"
        self.samples = []
        with open(self.data_path) as f:
            for line in f:
                if line.strip().startswith('<value name="text_1"'):
                    self.samples.append(ParaphraseClassificationSample(
                        sentence1=line.replace('<value name="text_1">', "").replace('</value>', "").strip(),
                        sentence2="",
                        label=0,
                    ))
                if line.strip().startswith('<value name="text_2"'):
                    self.samples[-1].sentence2 = line.replace('<value name="text_2">', "").replace('</value>', "").strip()
                if line.strip().startswith('<value name="class"'):
                    self.samples[-1].label = int(line.replace('<value name="class">', "").replace('</value>', ""))
        if self.variant == "task2":
            # “Task 2 Binary classification: given a pair of sentences, to predict whether they are paraphrases (
            # whether precise or near paraphrases) or non-paraphrases”
            for sample in self.samples:
                if sample.label == 0:
                    sample.label = 1
                elif sample.label == -1:
                    sample.label = 0
        else:
            for sample in self.samples:
                if sample.label == -1:
                    sample.label = 0

    def __str__(self):
        return f"russian_paraphrases_{self.variant}"

    def get_samples(self) -> List[ParaphraseClassificationSample]:
        return self.samples


@dataclass
class TurkuParaphraseClassificationSample(ParaphraseClassificationSample):
    full_label: str


class TurkuPhenomenaMixin:

    def annotate_samples(self, samples: List[TurkuParaphraseClassificationSample]) -> None:
        lengths = [len(sample.sentence1) + len(sample.sentence2) for sample in samples]
        threshold = np.percentile(lengths, 50)
        for i, sample in enumerate(samples):
            sample.is_long_sentence = (len(sample.sentence1) + len(sample.sentence2)) >= threshold

    def get_subset_mask(self, phenomenon: str = None, label: int = None) -> np.ndarray:
        samples = self.get_samples()
        if phenomenon is not None:
            if phenomenon == "subsumption":
                has_phenomenon = lambda sample: "<" in sample.full_label or ">" in sample.full_label
            elif phenomenon == "style":
                has_phenomenon = lambda sample: "s" in sample.full_label
            else:
                has_phenomenon = lambda sample: getattr(sample, phenomenon)
            phenomenon_mask = np.array([has_phenomenon(sample) for sample in samples])
        else:
            phenomenon_mask = np.full(len(samples), True)

        if label is not None:
            label_mask = np.array([sample.label == label for sample in samples])
        else:
            label_mask = np.full(len(samples), True)
        mask = phenomenon_mask & label_mask
        return mask


class FinnishParaphraseClassificationTask(ParaphraseClassificationTask, TurkuPhenomenaMixin):
    language = "fi"

    def __init__(self):
        self.dataset = load_dataset("TurkuNLP/turku_paraphrase_corpus", "classification")["test"]

    def __str__(self):
        return "finnish_paraphrases"

    def get_samples(self) -> List[TurkuParaphraseClassificationSample]:
        samples = []
        for dataset_sample in self.dataset:
            sample = TurkuParaphraseClassificationSample(
                sentence1=dataset_sample["text1"],
                sentence2=dataset_sample["text2"],
                label=1 if "4" in dataset_sample["label"] else 0,
                full_label=dataset_sample["label"],
            )
            samples.append(sample)
        self.annotate_samples(samples)
        return samples


class SwedishParaphraseClassificationTask(ParaphraseClassificationTask, TurkuPhenomenaMixin):
    language = "sv"

    def __init__(self):
        self.data_path = Path(__file__).parent / "data" / "swedish_paraphrases" / "test.json"
        with open(self.data_path) as f:
            self.data = json.load(f)
        # print({sample["label"] for sample in data})
        self.samples = [
            TurkuParaphraseClassificationSample(
                sentence1=sample["txt1"],
                sentence2=sample["txt2"],
                label=1 if "4" in sample["label"] else 0,
                full_label=sample["label"],
            )
            for sample in self.data
        ]

    def __str__(self):
        return "swedish_paraphrases"

    def get_samples(self) -> List[TurkuParaphraseClassificationSample]:
        return self.samples


@dataclass
class PAWSXSample(ParaphraseClassificationSample):
    sample_id: int


class PAWSXParaphraseClassificationTask(ParaphraseClassificationTask):

    def __init__(self, language: str, split="validation"):
        self.language = language
        self.split = split
        self.dataset = load_dataset("paws-x", language)[split]

    def __str__(self):
        return f"pawsx_{self.language}_{self.split}"

    def get_samples(self) -> List[PAWSXSample]:
        sentence1_list = self.dataset["sentence1"]
        sentence2_list = self.dataset["sentence2"]
        labels = self.dataset["label"]
        sample_ids = self.dataset["id"]
        # Data cleaning (https://github.com/google-research-datasets/paws/issues/15)
        unused_indices = {i for i in range(len(sentence1_list)) if sentence1_list[i].strip() == "NS"}
        unused_indices |= {i for i in range(len(sentence2_list)) if sentence2_list[i].strip() == "NS"}
        sentence1_list = [s for i, s in enumerate(sentence1_list) if i not in unused_indices]
        sentence2_list = [s for i, s in enumerate(sentence2_list) if i not in unused_indices]
        labels = [s for i, s in enumerate(labels) if i not in unused_indices]
        sample_ids = [s for i, s in enumerate(sample_ids) if i not in unused_indices]
        assert "NS" not in sentence1_list + sentence2_list
        return [
            PAWSXSample(
                sentence1=sentence1,
                sentence2=sentence2,
                label=label,
                sample_id=sample_id,
            )
            for sentence1, sentence2, label, sample_id in zip(sentence1_list, sentence2_list, labels, sample_ids)
        ]


class ETPCTask(ParaphraseClassificationTask):

    def __init__(self):
        from experiments.data.etpc_data import ETPCData
        self.data = ETPCData()

    def __str__(self):
        return "etpc"

    def get_samples(self) -> List[ParaphraseClassificationSample]:
        return [
            ParaphraseClassificationSample(
                sentence1=sample.sentence1,
                sentence2=sample.sentence2,
                label=sample.label,
            )
            for sample in self.data.samples.values()
        ]

    def get_subset_mask(self, phenomenon: str = None, label: int = None) -> np.ndarray:
        if phenomenon is not None:
            assert phenomenon in self.data.phenomena or getattr(list(self.data.samples.values())[0], phenomenon) is not None
            if phenomenon in self.data.phenomena:
                phenomenon_mask = np.array([phenomenon in sample.phenomena for sample in self.data.samples.values()])
            else:
                phenomenon_mask = np.array([getattr(sample, phenomenon) for sample in self.data.samples.values()])
        else:
            phenomenon_mask = np.full(len(self.data.samples), True)

        if label is not None:
            label_mask = np.array([sample.label == label for sample in self.data.samples.values()])
        else:
            label_mask = np.full(len(self.data.samples), True)
        mask = phenomenon_mask & label_mask
        return mask
