#!/bin/bash

#PJM -g jh170036a
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=30:00:00
#PJM --fs /work,/data
#PJM -N training
#PJM -o train.txt
#PJM -j
#PJM --mail-list suzuki-takahiro596@g.ecc.u-tokyo.ac.jp
#PJM -m b
#PJM -m e

#module load cuda/10.2
module load cuda/11.1

cd "${PJM_O_WORKDIR}" || exit

#pipenv shell #activate the virtual environment

i=1 #1,2,3,4,5
trainPath="sim${i}_1000"
trainBbox="simDataInfo/bboxInfo/bboxInfo${i}_1000.csv"
validPath='AllDataDir' #'AbnormalDir'
validBboxName='rare_small_bboxInfo_20_1_withNormal'
#validBbox="${validBboxName}.csv"

testPath='AllDataDir' #'AbnormalDir'
testBboxName='rare_small_bboxInfo_81_1_withNormal'
#testBbox="${testBboxName}.csv"

modelPath="/work/gk36/k77012/M2/model/"
saveFROCPath="/work/gk36/k77012/M2/FROC/"

epoch=40
batch_size=64
numSamples=50

#optimizerはAdamでいいか

model='SSD'
pretrained="ImageNet" #'BigBbox' #"ImageNet" as default

saveDir="/work/gk36/k77012/M2/result/${trainPath}_${validBboxName}_${testBboxName}_${model}_batch${batch_size}_epoch${epoch}_pretrained${pretrained}/"
mkdir -p ${saveDir} #for saving Dir
mkdir -p "${saveDir}/train"
mkdir -p "${saveDir}/valid"
mkdir -p "${saveDir}/test"

pipenv run python ../WisteriaCodes/train.py ${trainPath} ${validPath} ${testPath} ${trainBbox} ${validBboxName} ${testBboxName} ${modelPath} ${model} ${epoch} ${batch_size} ${numSamples} ${pretrained} ${saveDir} ${saveFROCPath} ${i} >> ../train_log/${trainPath}_${validBboxName}_${testBboxName}_${model}_epoch${epoch}_batchsize${batch_size}_${pretrained}_${i}.txt
echo 'training_finished'
