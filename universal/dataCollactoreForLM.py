#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Pretraining the library models for denoising language modeling on a text file or a dataset.
Here is the full list of checkpoints on the hub that can be pretrained by this script:
https://huggingface.co/models?filter=bart
"""
# You can also adapt this script on your own denoising language modeling task. Pointers for this are left as comments.

import math
import random
from typing import Dict, List
import numpy as np
import torch

from transformers import (
    BatchEncoding,
    PreTrainedTokenizerBase,
    DefaultDataCollator,DataCollatorForLanguageModeling,DataCollatorWithPadding
)
from transformers.models.bart.modeling_bart import shift_tokens_right
class DataCollatorForBartDenoisingLM(DefaultDataCollator):
    """
    Data collator used for BART denoising language modeling. The code is largely copied from
    `<https://github.com/morganmcg1/rotobart/blob/main/data_collator.py#L223>`__.
    For more information on how BART denoising language modeling works, one can take a look
    at the `official paper <https://arxiv.org/pdf/1910.13461.pdf>`__
    or the `official code for preprocessing <https://github.com/facebookresearch/fairseq/blob/main/fairseq/data/denoising_dataset.py>`__ .
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data
        mask_ratio (:obj:`float`):
            The probability with which to (randomly) mask tokens in the input
        poisson_lambda (:obj:`float`):
            Mean parameter of Poisson distribution used to generate span-lengths to be masked
        permute_sentence_ratio (:obj:`float`):
            Ratio of sentences to be permuted in each document
        decoder_start_token_id: (:obj:`int):
            The decoder start token id of the model
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.mask_ratio = 0.35
        self.poisson_lambda = 3.5
        self.permute_sentence_ratio = 1.0
    def __post_init__(self):
        if self.tokenizer.mask_token is None or self.tokenizer.eos_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token or eos token token which is necessary for denoising"
                " language modeling. "
            )

    def __call__(self,examples):
        # convert list to dict and tensorize input
        batch = {k: np.array([examples[i][k] for i in range(len(examples))]) for k, v in examples[0].items()}

        batch["labels"] = batch["input_ids"].copy()
        batch["decoder_input_ids"] = np.roll(batch["input_ids"].copy(), 1, -1)
        do_permute = False
        if self.permute_sentence_ratio > 0.0:
            batch["input_ids"] = self.permute_sentences(batch["input_ids"])
            do_permute = True

        # masking span of tokens (text infilling in the paper)
        if self.mask_ratio:
            batch["input_ids"], batch["labels"] = self.span_mask_tokens(
                batch["input_ids"], batch["labels"], do_permute
            )

        # ignore pad tokens
        batch["attention_mask"] = (batch["input_ids"] != self.tokenizer.pad_token_id).astype(int)
        batch["decoder_attention_mask"] = (batch["decoder_input_ids"] != self.tokenizer.pad_token_id).astype(int)
        for k, v in batch.items():
            batch[k] = torch.tensor(v).long()
        return batch

    def permute_sentences(self, input_ids):
        """
        Shuffle sentences in each document.
        """
        results = input_ids

        # find end locations of sentences
        end_sentence_mask = np.array(input_ids) == self.tokenizer.pad_token_id
        sentence_ends = np.argwhere(end_sentence_mask)
        sentence_ends[:, 1] += 1
        example_has_multiple_sentences, num_sentences = np.unique(sentence_ends[:, 0], return_counts=True)
        num_sentences_map = dict(zip(example_has_multiple_sentences, num_sentences))

        num_to_permute = np.ceil(num_sentences * self.permute_sentence_ratio).astype(int)
        num_to_permute_map = dict(zip(example_has_multiple_sentences, num_to_permute))

        sentence_ends = np.split(sentence_ends[:, 1], np.unique(sentence_ends[:, 0], return_index=True)[1][1:])
        sentence_ends_map = dict(zip(example_has_multiple_sentences, sentence_ends))

        for i in range(input_ids.shape[0]):
            if i not in example_has_multiple_sentences:
                continue
            substitutions = np.random.permutation(num_sentences_map[i])[: num_to_permute_map[i]]
            ordering = np.arange(0, num_sentences_map[i])
            ordering[substitutions] = substitutions[np.random.permutation(num_to_permute_map[i])]
            # write shuffled sentences into results
            index = 0
            for j in ordering:
                sentence = input_ids[i, (sentence_ends_map[i][j - 1] if j > 0 else 0) : sentence_ends_map[i][j]]
                results[i, index : index + sentence.shape[0]] = sentence
                index += sentence.shape[0]
        return results
    def span_mask_tokens(self, input_ids, labels, do_permute):
        """
        Sampling text spans with span lengths drawn from a Poisson distribution and masking them.
        """
        special_tokens_mask_labels = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        special_tokens_mask_inputs = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in input_ids.tolist()
        ]
        special_tokens_mask_labels = np.array(special_tokens_mask_labels, dtype=bool)
        special_tokens_mask_inputs = np.array(special_tokens_mask_inputs, dtype=bool)

        # determine how many tokens we need to mask in total
        is_token_mask = ~(input_ids == self.tokenizer.pad_token_id) & ~special_tokens_mask_inputs
        num_tokens_to_mask = int(math.ceil(is_token_mask.astype(float).sum() * self.mask_ratio))
        if num_tokens_to_mask == 0:
            return input_ids, labels

        # generate a sufficient number of span lengths
        span_lengths = np.random.poisson(lam=self.poisson_lambda, size=(num_tokens_to_mask,))
        while np.cumsum(span_lengths, 0)[-1] < num_tokens_to_mask:
            span_lengths = np.concatenate(
                [span_lengths, np.random.poisson(lam=self.poisson_lambda, size=(num_tokens_to_mask,))]
            )

        # remove all spans of length 0
        # note that BART inserts additional mask tokens where length == 0,
        # which we do not implement for now as it adds additional complexity
        span_lengths = span_lengths[span_lengths > 0]

        # trim to about num_tokens_to_mask tokens
        cutoff_idx = np.argmin(np.abs(np.cumsum(span_lengths, 0) - num_tokens_to_mask)) + 1
        span_lengths = span_lengths[:cutoff_idx]

        # randomly choose starting positions for masking
        token_indices = np.argwhere(is_token_mask == 1)
        span_starts = np.random.permutation(token_indices.shape[0])[: span_lengths.shape[0]]
        # prepare mask
        masked_indices = np.array(token_indices[span_starts])
        mask = np.full_like(input_ids, fill_value=False)

        # mask starting positions
        for mi in masked_indices:
            mask[tuple(mi)] = True
        span_lengths -= 1

        # fill up spans
        max_index = input_ids.shape[1] - 1
        remaining = (span_lengths > 0) & (masked_indices[:, 1] < max_index)
        while np.any(remaining):
            masked_indices[remaining, 1] += 1
            for mi in masked_indices:
                mask[tuple(mi)] = True
            span_lengths -= 1
            remaining = (span_lengths > 0) & (masked_indices[:, 1] < max_index)

        # place the mask tokens
        mask[np.where(special_tokens_mask_inputs)] = False
        input_ids[np.where(mask)] = self.tokenizer.mask_token_id
        if not do_permute:
            labels[np.where(mask == 0)] = -100
        else:
            labels[np.where(special_tokens_mask_labels)] = -100

        # remove mask tokens that are not starts of spans
        to_remove = (mask == 1) & np.roll((mask == 1), 1, 1)
        new_input_ids = np.full_like(input_ids, fill_value=self.tokenizer.pad_token_id)
        for i, example in enumerate(input_ids):
            new_example = example[~to_remove[i]]
            new_input_ids[i, : new_example.shape[0]] = new_example

        return new_input_ids, labels
class DataCollatorForkMachineTranslation(DefaultDataCollator):

    def __init__(self, model, tokenizer, force_domain=None):
        self.model = model
        self.tokenizer = tokenizer
        self.masked_data_collator = DataCollatorForLanguageModeling(
            tokenizer = tokenizer, mlm=True, mlm_probability=0.15
        )
        self.padding_data_collator = DataCollatorWithPadding(tokenizer=tokenizer,max_length=128)
        self.force_domain = force_domain

    def back_translate(self, batch):
        batch = {k: [batch[i][k] for i in range(len(batch))] for k, v in batch[0].items()}
        batch = self.padding_data_collator(batch).to('cuda')
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        self.model.eval()
        final_output = []
        with torch.no_grad():
            for k,v in enumerate(input_ids):
                bt_outputs = self.model.generate(input_ids=[v],attention_mask=[attention_mask[k]], decoder_start_token_id = batch["tgt_lang"] ,force_words_ids=self.force_domain(batch["tgt_lang"]), max_length =128)
                bt_text = self.tokenizer.batch_decode(bt_outputs, skip_special_tokens=True)[0]
                self.tokenizer.src_lang = self.tokenizer.id_to_lang_code[batch["src_lang"][k][0][0].to("cpu").numpy().tolist()]
                # self.tokenizer.tgt_lang = self.tokenizer.id_to_lang_code[batch["tgt_lang"][k][0].to("cpu").numpy().tolist()]
                output_sample = self.tokenizer(bt_text,truncation=True, max_length=128)
                encodings = {
                    'input_ids': torch.tensor(output_sample['input_ids'], dtype=torch.long, device = 'cuda'),
                    'attention_mask': torch.tensor(output_sample['attention_mask'], dtype=torch.long, device = 'cuda'),
                    'labels':torch.tensor(v, dtype=torch.long, device = 'cuda'),
                }
                final_output.append(encodings)
        input_ids = torch.stack([example['input_ids'] for example in final_output])
        attention_mask = torch.stack([example['attention_mask'] for example in final_output])
        labels = torch.stack(example['labels'] for example in final_output)
        labels[labels[:, :] == 0] = -100
        returns = {"labels":labels,"input_ids":input_ids,"attention_mask":attention_mask}
        returns = self.padding_data_collator(returns)
        self.model.train()
        return returns

    def denoising(self, batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([example['input_ids'] for example in batch])
        input_ids, _ = self.masked_data_collator.mask_tokens(input_ids)
        lm_labels = torch.stack([example['target_ids'] for example in batch])
        lm_labels[lm_labels[:, :] == 0] = -100
        attention_mask = torch.stack([example['attention_mask'] for example in batch])
        decoder_attention_mask = torch.stack([example['target_attention_mask'] for example in batch])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'lm_labels': lm_labels,
            'decoder_attention_mask': decoder_attention_mask
        }

    def __call__(self, batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Take a list of samples from a Dataset and collate them into a batch.
        Returns:
            A dictionary of tensors
        """
        if self.model.training:
            return self.back_translate(batch)
        else:
            return self.padding_data_collator(batch)