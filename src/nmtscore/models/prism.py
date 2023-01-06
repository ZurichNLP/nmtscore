import copy
import sys
import tarfile
from pathlib import Path
from typing import List, Union, Dict, Iterator, Any, Tuple

import numpy as np
import sentencepiece as spm
import torch
from tqdm import tqdm
from fairseq import checkpoint_utils, utils
from fairseq.data import LanguagePairDataset
from transformers.utils import cached_file

from nmtscore.models import TranslationModel


class PrismModel(TranslationModel):
    """
    Loads a multilingual NMT model that is described in the following paper: Thompson, Brian, and Matt Post.
    "Automatic Machine Translation Evaluation in Many Languages via Zero-Shot Paraphrasing." Proceedings of the 2020
    Conference on Empirical Methods in Natural Language Processing (EMNLP). 2020.

    Incorporates some code from the reference implementation at https://github.com/thompsonb/prism
    Copyright (c) Brian Thompson

    In addition, incorporates from code from PyTorch (https://github.com/pytorch/fairseq/blob/main/fairseq/hub_utils.py)
    Copyright (c) Facebook, Inc. and its affiliates.
    """
    supported_languages = ['ar', 'bg', 'bn', 'ca', 'cs', 'da', 'de', 'el', 'en', 'es', 'et', 'eo', 'fi', 'fr', 'he',
                  'hr', 'hu', 'id', 'it', 'ja', 'kk', 'lt', 'lv', 'mk', 'nl', 'no', 'pl', 'pt', 'ro', 'ru',
                  'sk', 'sl', 'sq', 'sr', 'sv', 'tr', 'uk', 'vi', 'zh']

    def __init__(self,
                 model_dir: Union[Path, str] = None,
                 device=None,
                 ):
        self.model_dir = Path(model_dir) if model_dir else self._download_model()
        self.device = device
        self._load_model()
        for i in range(len(self.models)):
            if self.device is not None:
                self.models[i] = self.models[i].to(self.device)
            self.models[i].make_generation_fast_(
                beamable_mm_beam_size=None,
                need_attn=False,
            )

    def __str__(self):
        return "prism"

    @property
    def requires_src_lang(self) -> bool:
        return False

    def _set_tgt_lang(self, tgt_lang: str):
        assert tgt_lang in self.supported_languages
        self.tgt_lang = tgt_lang

    @torch.no_grad()
    def _translate(self,
                   source_sentences: List[str],
                   return_score: bool = False,
                   batch_size: int = 8,
                   beam=5,
                   verbose=False,
                   **kwargs,
                   ) -> Union[List[str], List[Tuple[str, float]]]:
        self.args.batch_size = batch_size
        return self.sample(source_sentences, return_score=return_score, batch_size=batch_size, beam=beam, verbose=verbose, **kwargs)

    @torch.no_grad()
    def _score(self,
               source_sentences: List[str],
               hypothesis_sentences: List[str],
               batch_size: int = 8,
               **kwargs,
               ) -> List[float]:
        self.args.batch_size = batch_size
        results = [None, ] * len(source_sentences)
        tokenized_source_sentences = [self.encode(sentence) for sentence in source_sentences]
        tokenized_hypothesis_sentences = [self.encode(sentence, prepend_target_id=True) for sentence in hypothesis_sentences]
        for batch in tqdm(self._build_batches(tokenized_source_sentences, tokenized_hypothesis_sentences, skip_invalid_size_inputs=False),
                          disable=len(source_sentences) / self.args.batch_size < 10):
            if self.device is not None:  # must be a better way
                batch['id'] = batch['id'].cuda()
                batch['net_input']['src_tokens'] = batch['net_input']['src_tokens'].to(self.device)
                batch['net_input']['src_lengths'] = batch['net_input']['src_lengths'].to(self.device)
                batch['net_input']['prev_output_tokens'] = batch['net_input']['prev_output_tokens'].to(self.device)
                batch['target'] = batch['target'].to(self.device)

            generator = SequenceScorer(self.task.target_dictionary)
            translations = self.task.inference_step(generator, self.models, batch)

            ids = batch['id'].cpu().numpy()

            tok_scores = [x[0]['positional_scores'].cpu().numpy() for x in translations]

            # [1:] to skip language tag log prob
            sent_scores = [np.mean(x[1:]) for x in tok_scores]

            for _id, sent_score, _tok_score in zip(ids, sent_scores, tok_scores):
                results[_id] = 2 ** float(sent_score)

        if None in results:
            raise Exception('Missing one or more sentence scores')
        return results

    def _download_model(self) -> Path:
        archive_path = cached_file(
            "Devrim/prism-default",  # Prism model on the Hugging Face Hub (uploaded by a third party)
            filename="m39v1.tar",
        )
        archive_path = Path(archive_path)
        assert archive_path.exists()
        assert tarfile.is_tarfile(archive_path)
        extracted_dir = archive_path.with_suffix(".extracted")
        with tarfile.open(archive_path) as tar_file:
            tar_file.extractall(extracted_dir)
        return Path(extracted_dir) / "m39v1"

    def _load_model(self):
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(str(self.model_dir / 'spm.model'))
        # sys.stdout = open(os.devnull, 'w')
        self.models, self.args, self.task = checkpoint_utils.load_model_ensemble_and_task(
            [str(self.model_dir / 'checkpoint.pt'), ],
            arg_overrides=dict(data=str(self.model_dir) + '/'),
        )
        # sys.stdout = sys.__stdout__

    def sample(
        self, sentences: List[str], return_score: bool = False, beam: int = 1, verbose: bool = False, **kwargs
    ) -> Union[List[str], List[Tuple[str, float]]]:
        tokenized_sentences = [self.encode(sentence) for sentence in sentences]
        batched_hypos = self.generate(tokenized_sentences, beam, verbose, **kwargs)
        samples = [self.decode(hypos[0]["tokens"]) for hypos in batched_hypos]
        if return_score:
            tok_scores = [hypos[0]['positional_scores'].tolist() for hypos in batched_hypos]
            # [1:] to skip language tag log prob
            sent_scores = [float(2 ** np.mean(x[1:])) for x in tok_scores]
            return list(zip(samples, sent_scores))
        else:
            return samples

    def generate(
            self,
            tokenized_sentences: List[torch.LongTensor],
            beam: int = 5,
            verbose: bool = False,
            skip_invalid_size_inputs=False,
            inference_step_args=None,
            **kwargs
    ) -> List[List[Dict[str, torch.Tensor]]]:
        if torch.is_tensor(tokenized_sentences) and tokenized_sentences.dim() == 1:
            return self.generate(
                tokenized_sentences.unsqueeze(0), beam=beam, verbose=verbose, **kwargs
            )[0]

        # build generator using current args as well as any kwargs
        gen_args = copy.copy(self.args)
        gen_args.beam = beam
        for k, v in kwargs.items():
            setattr(gen_args, k, v)
        generator = self.task.build_generator(self.models, gen_args)

        inference_step_args = inference_step_args or {}
        results = []
        for batch in tqdm(self._build_batches(tokenized_sentences, skip_invalid_size_inputs=skip_invalid_size_inputs),
                          disable=len(tokenized_sentences) / self.args.batch_size < 10):
            prefix_tokens = torch.tensor(batch["nsentences"] * [self.task.source_dictionary.index(f"<{self.tgt_lang}>")]).unsqueeze(-1)
            if self.device is not None:
                batch = utils.apply_to_sample(lambda t: t.to(self.device), batch)
                prefix_tokens = prefix_tokens.to(self.device)
            translations = self.task.inference_step(
                generator, self.models, batch, prefix_tokens=prefix_tokens, **inference_step_args
            )
            for id, hypos in zip(batch["id"].tolist(), translations):
                results.append((id, hypos))

        # sort output to match input order
        outputs = [hypos for _, hypos in sorted(results, key=lambda x: x[0])]
        return outputs

    def encode(self, sentence: str, prepend_target_id=False) -> torch.LongTensor:
        sent = ' '.join(self.tokenizer.EncodeAsPieces(sentence))
        if prepend_target_id:
            sent = f'<{self.tgt_lang}> ' + sent
        return self.binarize(sent)

    def decode(self, tokens: torch.LongTensor) -> str:
        pieces = self.string(tokens).split()
        return self.tokenizer.DecodePieces(pieces[1:])

    def binarize(self, sentence: str) -> torch.LongTensor:
        return self.task.source_dictionary.encode_line(sentence, add_if_not_exist=False).long()

    def string(self, tokens: torch.LongTensor) -> str:
        return self.task.target_dictionary.string(tokens)

    def _build_batches(self,
                       source_tokens: List[List[int]],
                       target_tokens: List[List[int]] = None,
                       skip_invalid_size_inputs: bool = False,
    ) -> Iterator[Dict[str, Any]]:
        source_lengths = torch.LongTensor([t.numel() for t in source_tokens])
        if target_tokens is not None:
            target_lengths = torch.LongTensor([t.numel() for t in target_tokens])
            dataset = LanguagePairDataset(source_tokens, source_lengths, self.task.source_dictionary,
                                        tgt=target_tokens, tgt_sizes=target_lengths,
                                        tgt_dict=self.task.target_dictionary)
        else:
            dataset = self.task.build_dataset_for_inference(source_tokens, source_lengths)

        batch_iterator = self.task.get_batch_iterator(
            dataset=dataset,
            max_tokens=self.args.max_tokens,
            max_sentences=self.args.batch_size,
            max_positions=self.args.max_source_positions,
            ignore_invalid_inputs=skip_invalid_size_inputs,
            disable_iterator_cache=True,
        ).next_epoch_itr(shuffle=False)
        return batch_iterator


class SequenceScorer(object):
    """
    Copy of https://github.com/pytorch/fairseq/blob/master/fairseq/sequence_scorer.py
    with softmax temperature control added
    MIT License

    Copyright (c) Facebook, Inc. and its affiliates.
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """

    def __init__(self, tgt_dict, softmax_batch=None, temperature=1.0):
        self.pad = tgt_dict.pad()
        self.eos = tgt_dict.eos()
        self.softmax_batch = softmax_batch or sys.maxsize
        self.temperature = temperature
        assert self.softmax_batch > 0

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        """Score a batch of translations."""
        net_input = sample['net_input']

        def batch_for_softmax(dec_out, target):
            # assumes decoder_out[0] is the only thing needed (may not be correct for future models!)
            first, rest = dec_out[0], dec_out[1:]
            bsz, tsz, dim = first.shape
            if bsz * tsz < self.softmax_batch:
                yield dec_out, target, True
            else:
                flat = first.contiguous().view(1, -1, dim)
                flat_tgt = target.contiguous().view(flat.shape[:-1])
                s = 0
                while s < flat.size(1):
                    e = s + self.softmax_batch
                    yield (flat[:, s:e],) + rest, flat_tgt[:, s:e], False
                    s = e

        def gather_target_probs(probs, target):
            probs = probs.gather(
                dim=2,
                index=target.unsqueeze(-1),
            )
            return probs

        orig_target = sample['target']

        # compute scores for each model in the ensemble
        avg_probs = None
        avg_attn = None
        for model in models:
            model.eval()
            decoder_out = model.forward(**net_input)
            attn = decoder_out[1]
            if type(attn) is dict:
                attn = attn.get('attn', None)

            batched = batch_for_softmax(decoder_out, orig_target)
            probs, idx = None, 0
            for bd, tgt, is_single in batched:
                sample['target'] = tgt

                # divide the logits by temperature prior to softmax
                # for example, see https://github.com/pytorch/fairseq/blob/master/fairseq/sequence_generator.py:
                #   decoder_out[0][:, -1:, :].div_(temperature)
                bd[0].div_(self.temperature)

                curr_prob = model.get_normalized_probs(bd, log_probs=len(models) == 1, sample=sample).data
                if is_single:
                    probs = gather_target_probs(curr_prob, orig_target)
                else:
                    if probs is None:
                        probs = curr_prob.new(orig_target.numel())
                    step = curr_prob.size(0) * curr_prob.size(1)
                    end = step + idx
                    tgt_probs = gather_target_probs(curr_prob.view(tgt.shape + (curr_prob.size(-1),)), tgt)
                    probs[idx:end] = tgt_probs.view(-1)
                    idx = end
                sample['target'] = orig_target

            probs = probs.view(sample['target'].shape)

            if avg_probs is None:
                avg_probs = probs
            else:
                avg_probs.add_(probs)
            if attn is not None and torch.is_tensor(attn):
                attn = attn.data
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        if len(models) > 1:
            avg_probs.div_(len(models))
            avg_probs.log_()
            if avg_attn is not None:
                avg_attn.div_(len(models))

        bsz = avg_probs.size(0)
        hypos = []
        start_idxs = sample['start_indices'] if 'start_indices' in sample else [0] * bsz
        for i in range(bsz):
            # remove padding from ref
            ref = utils.strip_pad(sample['target'][i, start_idxs[i]:], self.pad) \
                if sample['target'] is not None else None
            tgt_len = ref.numel()
            avg_probs_i = avg_probs[i][start_idxs[i]:start_idxs[i] + tgt_len]
            score_i = avg_probs_i.sum() / tgt_len
            if avg_attn is not None:
                avg_attn_i = avg_attn[i]
                alignment = utils.extract_hard_alignment(avg_attn_i, sample['net_input']['src_tokens'][i],
                                                         sample['target'][i], self.pad, self.eos)
            else:
                avg_attn_i = alignment = None
            hypos.append([{
                'tokens': ref,
                'score': score_i,
                'attention': avg_attn_i,
                'alignment': alignment,
                'positional_scores': avg_probs_i,
            }])
        return hypos
