# Name: Barid
#
import torch
from torch import nn
import os
from logging import getLogger
import torch
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import torch.nn.functional as F

logger = getLogger()


class CrossInit(object):

    def __init__(self, params):
        """
        Initialize trainer script.
        """
        self.emb_dim = params["emb_dim"]
        self.vocabulary = params["vocabulary"]
        self.multilingual_emb = nn.Embedding(self.vocabulary, self.emb_dim).to("cuda:0")
        nn.init.normal_(self.multilingual_emb.weight, mean=0, std=self.emb_dim**-0.5)
        self.span = params["span"]
        self.bs = params["bs"]
        self.rand_dict = params["ranking"]
        self.lang_num = len(self.rand_dict)
        self.params = params
        # optimizers
        self.adm_opt = torch.optim.Adam(
            self.multilingual_emb.parameters(), lr=params["lr"]
        )
        self.freq_num = params["freq_num"]
        self.tokenizer = params["tokenizer"]

    def get_pairs(self):
        lang_left, lang_right = np.random.randint(low=0, high=self.lang_num, size=2)
        span_seed = np.random.randint(self.freq_num, size=1)[0]
        left_lang = self.rand_dict[lang_left]
        right_lang = self.rand_dict[lang_right]
        right_lang_lenght = len(right_lang["input_ids"])
        left_lang_lenght = len(left_lang["input_ids"])
        if span_seed + self.span > right_lang_lenght:
            right_span_seed =  right_lang_lenght -  self.span
        else:
            right_span_seed = span_seed
        if span_seed + self.span > left_lang_lenght:
            span_seed = left_lang_lenght - self.span

        positive_pair_left = {
            "input_ids": left_lang["input_ids"][
                max(span_seed - self.span, 0) : min(
                    span_seed + self.span, self.freq_num
                )
                + 1
            ],
            "attention_mask": left_lang["attention_mask"][
                max(span_seed - self.span, 0) : min(
                    span_seed + self.span, self.freq_num
                )
                + 1
            ],
        }
        positive_pair_right = {
            "input_ids": right_lang["input_ids"][
                max(right_span_seed - self.span, 0) : min(
                    right_span_seed + self.span, self.freq_num
                )
                + 1
            ],
            "attention_mask": right_lang["attention_mask"][
                max(right_span_seed - self.span, 0) : min(
                    right_span_seed + self.span, self.freq_num
                )
                + 1
            ],
        }
        out_span = list(
            set(range(len(self.rand_dict[lang_right]["input_ids"]))).difference(
                range(
                    max(right_span_seed - self.span, 0),
                    min(right_span_seed + self.span, self.freq_num) + 1,
                )
            )
        )
        out_span = np.random.choice(out_span,self.span*2,replace=False)
        negative_pair_right = {
            "input_ids": [right_lang["input_ids"][i] for i in out_span],
            "attention_mask": [right_lang["attention_mask"][i] for i in out_span],
        }
        return (
            self.tokenizer.pad(
                positive_pair_left,
                padding="longest",
                return_tensors="pt",
            ).to("cuda:0"),
            self.tokenizer.pad(
                positive_pair_right,
                padding="longest",
                return_tensors="pt",
            ).to("cuda:0"),
            self.tokenizer.pad(
                negative_pair_right,
                padding="longest",
                return_tensors="pt",
            ).to("cuda:0"),
        )
    def _embedding_with_mask(self, inputs):
        em = self.multilingual_emb(inputs["input_ids"]).to("cuda:0")
        em  = em* torch.unsqueeze(
            inputs["attention_mask"], -1
        )
        em = torch.sum(em, dim=1) / torch.sum(
            inputs["attention_mask"], dim=1, keepdim=True
        )
        em = torch.mean(em, dim=0, keepdim=True)
        return em.to("cuda:0")

    def _contrastive_learning(self):
        positive_pair_left, positive_pair_right, negative_pair_right = self.get_pairs()
        positive_pair_left_emb = self._embedding_with_mask(positive_pair_left)
        positive_pair_right_emb = self._embedding_with_mask(positive_pair_right)
        negative_pair_right_emb = self._embedding_with_mask(negative_pair_right)
        positive_dot = torch.matmul(
            positive_pair_left_emb, torch.permute(positive_pair_right_emb, [-1, -2])
        )
        negative_dot = torch.matmul(
            positive_pair_left_emb, torch.permute(negative_pair_right_emb, [-1, -2])
        )
        return (
            torch.concatenate([positive_dot, negative_dot], dim=-1).to("cuda:0"),
            torch.Tensor([[1, 0]]).to("cuda:0"),
        )

    def contrastive_learning_step(self):
        batch_x = []
        batch_y = []
        for k in range(self.bs):
            (x, y) = self._contrastive_learning()
            if k == 0:
                batch_x = x.to("cuda:0")
                batch_y = y.to("cuda:0")
            else:
                batch_x = torch.concatenate([batch_x, x], dim=0)
                batch_y = torch.concatenate([batch_y, y], dim=0)
        loss = F.binary_cross_entropy(
            F.softmax(batch_x.view(self.bs, 2), -1),
            batch_y.to("cuda:0").view(self.bs, 2),
            reduction="mean",
        )
        self.adm_opt.zero_grad()
        loss.backward()
        self.adm_opt.step()
        return loss

    def export(self):
        save_path = os.path.join(
            self.params["output_path"], "vectors-%s.pth" % self.params["name"]
        )
        logger.info("Writing embeddings to %s ..." % save_path)
        torch.save({"embeddings": self.multilingual_emb}, save_path)
