from nmtscore.models.m2m100 import M2M100Model
from nmtscore.models.tokenization_small100 import SMALL100Tokenizer


class SMALL100Model(M2M100Model):
    """
    Loads the model described in: SMaLL-100: Introducing Shallow Multilingual Machine Translation Model for
    Low-Resource Languages (Mohammadshahi et al., EMNLP 2022)
    """

    def __init__(self,
                 model_name_or_path: str = "alirezamsh/small100",
                 device=None,
                 ):
        super().__init__(model_name_or_path, device)
        self.pipeline.model.config.max_length = 200
        self.src_lang = "en"  # Not used

    def __str__(self):
        return self.model_name_or_path

    def _load_tokenizer(self):
        return SMALL100Tokenizer.from_pretrained(self.model_name_or_path)

    @property
    def requires_src_lang(self) -> bool:
        return False

    def _set_src_lang(self, src_lang: str):
        raise NotImplementedError
