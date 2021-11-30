#!/bin/bash

#PJM -g jh170036a
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=10:00:00
#PJM --fs /work,/data
#PJM -N smallbbox
#PJM -o extract_small_bbox.txt
#PJM -j
#PJM --mail-list suzuki-takahiro596@g.ecc.u-tokyo.ac.jp
#PJM -m b
#PJM -m e

#module load cuda/10.2
module load cuda/11.1

cd "${PJM_O_WORKDIR}" || exit

#pipenv shell #activate the virtual environment
dataPath='AbnormalDir' #'AbnormalDir5880' #data poolなので広めであればよい
dataBbox='exist_under150smallabnormal_fromalldata_bboxInfo.csv'

numSamples=200 #50
maxsize=150 #150*150

saveDir="/work/gk36/k77012/M2/smallbbox/maxsize_${maxsize}/"
mkdir -p ${saveDir}
mkdir -p "${saveDir}/Original_images"
mkdir -p "${saveDir}/Images_with_gt"

#python ../codes/train.py ${trainPath} ${validPath} ${testPath} ${trainBbox} ${validBbox} ${testBbox} ${model} ${epoch} ${batch_size} ${numSamples} ${pretrained}>> ../train_log/${trainPath}_${validPath}_${testPath}_${model}_epoch${epoch}_batchsize${batch_size}_${pretrained}.txt
#pipenv run python ../WisteriaCodes/extract_small_bbox.py ${trainPath} ${validPath} ${testPath} ${trainBbox} ${validBbox} ${testBbox} ${model} ${epoch} ${batch_size} ${numSamples} ${pretrained} >> ../train_log/${trainPath}_${validPath}_${testPath}_${model}_epoch${epoch}_batchsize${batch_size}_${pretrained}.txt
pipenv run python ../WisteriaCodes/extract_small_bbox.py ${dataPath} ${dataBbox} ${numSamples} ${maxsize} ${saveDir}
echo 'finish_extract_small_bbox'
