# Name: Barid
#
import torch
from torch import nn
import os
from logging import getLogger
import scipy
import scipy.linalg
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
        self.emb_dim = params.emb_dim
        self.vocabulary = params.vocabulary
        self.multilingual_emb = nn.Embedding(self.vocabulary, self.emb_dim)
        self.multilingual_emb.to("cuda")
        self.span = params.span
        self.bs = params.bs
        self.rand_dict = params.ranking
        self.lang_num = len(self.rand_dict)
        # optimizers
        self.adm_opt = torch.optim.Adam(params.lr)
        self.freq_num = params.freq_num

    def get_pairs(self):
        lang_left,lang_right = np.random.randint(low=1, high=self.lang_num+1,size=2)
        span_seed = np.random.randint(self.freq_num,size=1)
        positive_span= list(range(max(span_seed - self.span, 0), min(span_seed+self.span,self.freq_num)))
        positive_pair_left = self.rand_dict[lang_left][positive_span]
        positive_pair_right = self.rand_dict[lang_right][positive_span]
        negative_pair_righ = np.random.choice(self.rand_dict[lang_right] - positive_span)
        return (positive_pair_left,positive_pair_right), (positive_pair_left,negative_pair_righ)

    def _contrastive_learning(self):
       (positive_pair_left,positive_pair_right), (positive_pair_left,negative_pair_righ) = self.get_pairs()
       positive_pair_left_emb = torch.mean(self.multilingual_emb(positive_pair_left),dim=0)
       positive_pair_right_emb = torch.mean(self.multilingual_emb(positive_pair_right),dim=0)
       negative_pair_righ_emb = torch.mean(self.multilingual_emb(negative_pair_righ),dim=0)
       positive_dot = torch.matmul(positive_pair_left_emb,positive_pair_right_emb.T)
       negative_dot = torch.matmul(positive_pair_left_emb,negative_pair_righ_emb.T)
       return ((positive_dot,negative_dot),(1,0))
    def contrastive_learning_step(self):
        batch_x = []
        batch_y = []
        for _ in range(self.bs):
            (x,y) = self._contrastive_learning()
            batch_x.append(x)
            batch_y.append(y)
        loss = F.binary_cross_entropy(batch_x.view(self.bs, 2), batch_y.view(self.bs,2), reduction='mean')
        self.adm_opt.zero_grad()
        loss.backward()
        self.adm_opt.step()
        return loss

    def export(self):
        save_path = os.path.join(self.params.output_path, 'vectors-%s.pth' % self.params.name)
        logger.info('Writing embeddings to %s ...' % save_path)
        torch.save({'embeddings': self.multilingual_emb}, save_path)