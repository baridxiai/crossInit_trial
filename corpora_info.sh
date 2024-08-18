#!/bin/bash

SRC_CORPORA="en fr es de el bg ru tr ar vi th zh hi sw ur"
# SRC_CORPORA=" el bg tr th hi ur"
# English, Spanish, French, German, Italian, Portuguese, Dutch, Arabic, Polish, Egyptian, Japanese, Russian, Cebuano, Swedish, Ukrainian, Vietnamese, Chinese, Waray, Afrikaans & Swahili
# CORPORA="en fr de ru zh sw ur"
# CORPORA="ur"
ROOT=${HOME}/massive_data/data/xnli15/txt
# CORPORA_PATH=${HOME}/massive_data/data/txt
STREAM_SIZE=256
i=0
# for l in ${SRC_CORPORA}; do
#     i=$((i+1))
#     words= ${HOME}/miniconda3/bin/python textStream.py --path ${CORPORA_PATH}/${l}.all --language ${l} --tokenizer xlm-mlm-xnli15-1024 --output ${ROOT}/${l}_xlmStream --stream_size ${STREAM_SIZE}
#     echo ${l}_src@@${ROOT}/${l}_src_${STREAM_SIZE}@@${words} >> ${ROOT}/corpora_info.txt
# done
for l in ${SRC_CORPORA}; do
    echo ${l}@@${ROOT}/${l}.all@@$(du -k ${ROOT}/${l}.all | cut -f1) >> ${ROOT}/corpora_info_xnli15.txt
done
