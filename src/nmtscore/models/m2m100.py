from typing import List, Union, Tuple

import torch
from tqdm import tqdm
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from transformers.file_utils import PaddingStrategy
from transformers.models.m2m_100.modeling_m2m_100 import shift_tokens_right

from nmtscore.models import TranslationModel
from nmtscore.models.utils import batch


class M2M100Model(TranslationModel):
    """
    Loads one of the models described in: Fan, Angela, et al. "Beyond english-centric multilingual machine
    translation." Journal of Machine Learning Research 22.107 (2021): 1-48.

    Uses the implementation of the Hugging Face Transformers library
    (https://huggingface.co/docs/transformers/model_doc/m2m_100).
    """

    def __init__(self,
                 model_name_or_path: str = "facebook/m2m100_418M",
                 device=None,
                 ):
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_name_or_path)
        self.model_name_or_path = model_name_or_path
        self.model = M2M100ForConditionalGeneration.from_pretrained(model_name_or_path)
        if device is not None:
            self.model = self.model.to(device)
        self.model.config.max_length = max(self.model.config.max_length, self.model.config.max_position_embeddings - 4)

    def __str__(self):
        return self.model_name_or_path

    @property
    def requires_src_lang(self) -> bool:
        return True

    def _set_src_lang(self, src_lang: str):
        self.src_lang = src_lang
        self.tokenizer.src_lang = src_lang

    def _set_tgt_lang(self, tgt_lang: str):
        self.tgt_lang = tgt_lang
        self.tokenizer.tgt_lang = tgt_lang

    @torch.no_grad()
    def _translate(self,
                   source_sentences: List[str],
                   return_score: bool = False,
                   batch_size: int = 8,
                   num_beams: int = 5,
                   **kwargs,
                   ) -> Union[List[str], List[Tuple[str, float]]]:
        padding_strategy = PaddingStrategy.LONGEST if batch_size > 1 else PaddingStrategy.DO_NOT_PAD
        translations = []
        for src_sentences in tqdm(list(batch(source_sentences, batch_size)), disable=len(source_sentences) / batch_size < 10):
            inputs = self.tokenizer._batch_encode_plus(src_sentences, return_tensors="pt",
                                                       padding_strategy=padding_strategy)
            inputs = inputs.to(self.model.device)
            model_output = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.get_lang_id(self.tgt_lang),
                num_beams=num_beams,
                return_dict_in_generate=True,
                output_scores=return_score,
                **kwargs,
            )
            batch_translations = self.tokenizer.batch_decode(model_output.sequences, skip_special_tokens=True)
            if return_score:
                # Does not match our score method output for some reason; need to investigate further
                # scores = (2 ** model_output.sequences_scores).tolist()
                scores = [None for _ in batch_translations]
                assert len(batch_translations) == len(scores)
                batch_translations = list(zip(batch_translations, scores))
            translations += batch_translations
        return translations

    @torch.no_grad()
    def _score(self,
               source_sentences: List[str],
               hypothesis_sentences: List[str],
               batch_size: int = 8,
               **kwargs,
               ) -> List[float]:
        padding_strategy = PaddingStrategy.LONGEST if batch_size > 1 else PaddingStrategy.DO_NOT_PAD
        scores = []
        batch_iterator = zip(
            tqdm(batch(source_sentences, batch_size), disable=len(source_sentences) / batch_size < 10),
            batch(hypothesis_sentences, batch_size),
        )
        for src_sentences, tgt_sentences in batch_iterator:
            inputs = self.tokenizer._batch_encode_plus(src_sentences, return_tensors="pt", padding_strategy=padding_strategy)
            with self.tokenizer.as_target_tokenizer():
                # Hack: Append a second EOS token to make sure that one EOS is still there after shift_tokens_right
                tgt_sentences = [f"{sentence} {self.tokenizer.eos_token}" for sentence in tgt_sentences]
                labels = self.tokenizer._batch_encode_plus(tgt_sentences, return_tensors="pt", padding_strategy=padding_strategy)
            inputs = inputs.to(self.model.device)
            labels = labels.to(self.model.device)
            labels["input_ids"][labels["input_ids"] == self.tokenizer.pad_token_id] = -100
            inputs["decoder_input_ids"] = shift_tokens_right(labels["input_ids"], self.tokenizer.pad_token_id, self.model.config.decoder_start_token_id)
            output = self.model(**inputs)
            batch_scores = torch.zeros(len(src_sentences), device=self.model.device)
            for i in range(len(src_sentences)):
                loss = torch.nn.CrossEntropyLoss()(
                    output.logits[i][1:].view(-1, self.model.config.vocab_size),
                    labels["input_ids"][i][1:].view(-1),
                )
                batch_scores[i] = 2 ** (-loss)
            scores += batch_scores.tolist()
        return scores
