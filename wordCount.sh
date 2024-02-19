#!/bin/bash

CORPORA="en fr es de el bg ru tr ar vi th zh hi sw ur"
# CORPORA="en fr de ru zh sw ur"
# CORPORA="ur"
ROOT=${HOME}/massive_data/data/wiki/wiki_xlm_gpt
for l in ${CORPORA}; do
    perl -0777 -lape's/\s+/\n/g' ${l}_src_256 | sort | uniq -c | sort -nr >> ${l}_src_256_freq
done