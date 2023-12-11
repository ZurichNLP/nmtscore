import logging
from typing import List, Union, Tuple

import torch
from tqdm import tqdm
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, TranslationPipeline
from transformers.file_utils import PaddingStrategy
from transformers.models.m2m_100.modeling_m2m_100 import shift_tokens_right

from nmtscore.models.tokenization_small100 import SMALL100Tokenizer
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
                 fp16: bool = True,
                 ):
        self.model_name_or_path = model_name_or_path
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.model.eval()
        device = device if device is not None else "cpu"
        if fp16 and device != "cpu":
            logging.info(f"Using {self} with half precision")
            self.model = self.model.half()
        self.pipeline = TranslationPipeline(self.model, self.tokenizer, device=device)

    def __str__(self):
        return self.model_name_or_path

    def _load_tokenizer(self):
        return M2M100Tokenizer.from_pretrained(self.model_name_or_path)

    def _load_model(self):
        return M2M100ForConditionalGeneration.from_pretrained(self.model_name_or_path)

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
        self.pipeline.batch_size = len(source_sentences)
        outputs = self.pipeline(
            source_sentences,
            src_lang=self.src_lang,
            tgt_lang=self.tgt_lang,
            num_beams=num_beams,
            **kwargs,
        )
        translations = [output['translation_text'] for output in outputs]
        if return_score:
            # Compute scores later because TranslationPipeline currently does not return those
            scores = [None for _ in translations]
            translations = list(zip(translations, scores))
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
            inputs = self.tokenizer(
                src_sentences,
                text_target=tgt_sentences,
                return_tensors="pt",
                padding=padding_strategy,
            )
            inputs = inputs.to(self.model.device)
            inputs["labels"][inputs["labels"] == self.tokenizer.pad_token_id] = -100
            inputs["decoder_input_ids"] = shift_tokens_right(inputs["labels"], self.tokenizer.pad_token_id, self.model.config.decoder_start_token_id)
            output = self.model(**inputs)
            batch_scores = torch.zeros(len(src_sentences), device=self.model.device)
            if isinstance(self.tokenizer, SMALL100Tokenizer):
                offset = 0  # No language token in tgt
            else:
                offset = 1
            for i in range(len(src_sentences)):
                loss = torch.nn.CrossEntropyLoss()(
                    output.logits[i][offset:].view(-1, self.model.config.vocab_size),
                    inputs["labels"][i][offset:].view(-1),
                )
                batch_scores[i] = 2 ** (-loss)
            scores += batch_scores.tolist()
        return scores
