# -*- coding: utf-8 -*-
# code warrior: Barid
from itertools import chain
import numpy as np
from transformers import BatchEncoding

def temperature_sampling(path,alpha=0.5):
    corpora_info, total_alpha = read_corpora_info_langIdPathCount(path,alpha)
    sampling = []
    prob = []
    for lang in corpora_info:
        prob.append(float(lang[2]))
    prob = np.array(prob)
    prob /= prob.sum()
    prob = np.array([p ** alpha for p in prob])
    prob /= prob.sum()
    for index,lang in enumerate(corpora_info):
        sampling.append([lang[0],lang[1],prob[index]])
    return sampling
def read_corpora_info_langIdPathCount(path,alpha):
    corpora_info = []
    total = 0
    with open(path) as file:
        for _,line in enumerate(file.readlines()):
            lang = line.strip().split("@@")
            total += int(lang[-1])
            corpora_info.append(lang)
        file.close()

    return corpora_info, total


def xlm_dataset_streamer(data,tokenizer,stream_length):
    examples = tokenizer(data)
    concatenated_examples = list(chain(*examples["input_ids"]))
    concatenated_examples =  list(filter((tokenizer.bos_token_id).__ne__, concatenated_examples))
    total_length = len(concatenated_examples)
    seq_length = stream_length -2
    if total_length >= seq_length:
        total_length = (total_length // seq_length) * seq_length

    result = {
        "input_ids": [[tokenizer.bos_token_id] +concatenated_examples[i : i + seq_length] + [tokenizer.sep_token_id] for i in range(0, total_length, seq_length)]
    }
    return result
def xlm_streamer(example, tokenizer,stanza_pipeline,key_name="text", stream_length=256,langs=None, remove=True,require_lang=True):
    # example = sentence_split_function(example[key_name],tokenizer,stanza_pipeline)
    re = xlm_dataset_streamer(example[key_name],tokenizer,stream_length)
    if require_lang:
        language_id = tokenizer.lang2id[langs]
        re.update({"langs": (np.ones_like(re["input_ids"])*language_id).tolist()})
    if "token_type_ids" in re and remove:
        re.pop("token_type_ids")
        re.pop("attention_mask")
    return re
def mbart_dataset_streamer(data,tokenizer,stream_length):
    examples = tokenizer(data)
    concatenated_examples = list(chain(*examples["input_ids"]))
    concatenated_examples =  list(filter((tokenizer.cur_lang_code_id).__ne__, concatenated_examples))
    total_length = len(concatenated_examples)
    seq_length = stream_length -1
    if total_length >= seq_length:
        total_length = (total_length // seq_length) * seq_length

    result = {
        "input_ids": [concatenated_examples[i : i + seq_length] + [tokenizer.cur_lang_code_id] for i in range(0, total_length, seq_length)]
    }
    result['attention_mask'] = np.ones_like(result["input_ids"])
    return result

def sentence_split_function(example, tokenizer,stanza_pipeline):
    def _sentence_splits(sen):
        return   (" " +  tokenizer.pad_token + " ").join([s.text for s in sen]) + " " +  tokenizer.pad_token + " "
    sentences = [stanza_pipeline(exa).sentences for exa in example]
    sentences_split = [_sentence_splits(sen) for sen in sentences]
    # use pad token as end of sentence indicator
    return sentences_split
def mBart_streamer(example, tokenizer,stanza_pipeline,key_name="text", stream_length=512):
    example = sentence_split_function(example[key_name],tokenizer,stanza_pipeline)
    re = mbart_dataset_streamer(example,tokenizer,stream_length)
    return re


def mBart_wmt16ENRO(example, tokenizer):
    tokenizer.src_lang="en_XX"
    tokenizer.tgt_lang="ro_RO"
    # final_returns = {"input_ids":[],"labels":[],"attention_mask":[],"decoder_start_token_id":[]}
    example = example["translation"]
    returns = tokenizer(example['en'],target_text=example['ro'])
    returns["decoder_start_token_id"] = [tokenizer.lang_code_to_id['ro_RO']]
    # final_returns["input_ids"].append(returns["input_ids"])
    # final_returns["labels"].append(returns["labels"])
    # final_returns["attention_mask"].append(returns["attention_mask"])
    return returns
def mBart_floreENNE(example, tokenizer):
    tokenizer.src_lang="en_XX"
    tokenizer.tgt_lang="ne_NP"
    # final_returns = {"input_ids":[],"labels":[],"attention_mask":[],"decoder_start_token_id":[]}
    # example = example["translation"]
    returns = tokenizer(example['sentence_eng_Latn'],target_text=example['sentence_npi_Deva'])
    returns["decoder_start_token_id"] = [tokenizer.lang_code_to_id['ne_NP']]
    # final_returns["input_ids"].append(returns["input_ids"])
    # final_returns["labels"].append(returns["labels"])
    # final_returns["attention_mask"].append(returns["attention_mask"])
    return returns

def UNMT_sample(example, tokenizer,stream_length=256,src_lang="en_XX", tgt_lang="en_XX"):
    returns = {"input_ids":[], "src_lang":[],"tgt_lang":[],"src_text":[],'attention_mask':[]}
    raw_inputs = tokenizer(example)
    for k,v in enumerate(raw_inputs["input_ids"]):
        if len(v) <= stream_length-1 and len(v)>=8:
            returns["input_ids"].append(v)
            returns['src_lang'].append([tokenizer.lang_code_to_id[src_lang]])
            returns['tgt_lang'].append([tokenizer.lang_code_to_id[tgt_lang]])
            returns['attention_mask'].append(raw_inputs['attention_mask'][k])
    raw_inputs.update(returns)

    return returns
def mBart_UNMT(example, tokenizer,model,stanza_pipeline,key_name="text", stream_length=256,src_lang="en_XX", tgt_lang="ro_RO",lang_vocab_dict=None):
    example = sentence_split_function(example[key_name],tokenizer,stanza_pipeline)
    example = UNMT_sample(example,tokenizer,stream_length,src_lang,tgt_lang)
    return example
