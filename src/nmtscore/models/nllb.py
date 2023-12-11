
from nmtscore.models.m2m100 import M2M100Model


class NLLBModel(M2M100Model):

    def _validate_lang_code(self, lang_code: str):
        from transformers.models.nllb.tokenization_nllb import FAIRSEQ_LANGUAGE_CODES
        if lang_code not in FAIRSEQ_LANGUAGE_CODES:
            raise ValueError(f"{lang_code} is not a valid language code for {self}. "
                             f"Valid language codes are: {FAIRSEQ_LANGUAGE_CODES}")
