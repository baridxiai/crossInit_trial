# -*- coding: utf-8 -*-
# code warrior: Barid

import torch

# import sys
# import os
# import numpy as np

lang1 = {"id1":("King", "Man"), "id2":("Queen","Woman")}
lang2 = {"id1":("König", "Mann"), "id2":("Königin","Frau")}
cos_fn = torch.nn.CosineSimilarity(dim=-1)

def word_rep(word,model,tokenizer):
    return (torch.mean(model(tokenizer.encode(word,add_special_tokens=False,return_tensors="pt"))[0],0), word)

def kingManWomanQueen(model, tokenizer, lang1, lang2):
    sum = 0
    for k, v in lang1.items():
        w1,w2 = lang1[k]
        w_1,w_2 = lang2[k]
        w1 = word_rep(w1,model,tokenizer)
        w2 = word_rep(w2,model,tokenizer)
        w_1 = word_rep(w_1,model,tokenizer)
        w_2 = word_rep(w_2,model,tokenizer)
        w_1_cos = cos_fn(w1[0] - w2[0] + w_2[0], w_1[0]).detach().numpy().tolist()
        w1_cos = cos_fn(w_1[0] - w_2[0] + w2[0], w1[0]).detach().numpy().tolist()
        print(
            "|"
            + w1[1]
            + "-"
            + w2[1]
            + "+"
            + w_2[1]
            + "="
            + w_1[1]
            + "|"
            + str(round(round(w_1_cos, 2), 2))
        )
        print(
            "|"
            + w_1[1]
            + "-"
            + w_2[1]
            + "+"
            + w2[1]
            + "="
            + w1[1]
            + "|"
            + str(round(round(w1_cos, 2), 2))
        )
        sum += round(w_1_cos, 2)
        sum += round(w1_cos, 2)
    print(sum/(len(lang1)))


# kingManWomanQueen(model.embedding_softmax_layer,data_manager.encode)
# kingManWomanQueen(
#     model.embedding_softmax_layer,
#     data_manager.encode,
#     model.lang_encoding([1]),
#     model.lang_encoding([2]),
# )
