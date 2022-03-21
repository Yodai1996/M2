#!/bin/bash

#PJM -g jh170036a
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=30:00:00
#PJM --fs /work,/data
#PJM -N udr_training
#PJM -o udr_train.txt
#PJM -j
#PJM --mail-list suzuki-takahiro596@g.ecc.u-tokyo.ac.jp
#PJM -m b
#PJM -m e

#module load cuda/10.2
module load cuda/11.1

cd "${PJM_O_WORKDIR}" || exit

#pipenv shell #activate the virtual environment

#change here
fold=5 #1,2,3,4,5. 2 is not tried yet.
optimizer="Adam" #"Adam" #"VSGD" #"Adam" #"SGD"
m=1000 #200

i="udr" #無駄だが一応つけておく。

trainPath="udr_${m}"
trainBbox="simDataInfo/bboxInfo/bboxInfo_udr_${m}.csv"
validPath='AllDataDir' #'AbnormalDir'
validBboxName="rare_small_bboxInfo_20_${fold}_withNormal"
#validBbox="${validBboxName}.csv"

testPath='AllDataDir' #'AbnormalDir'
testBboxName="rare_small_bboxInfo_81_${fold}_withNormal"
#testBbox="${testBboxName}.csv"

modelPath="/work/gu14/k77012/M2/model/udr/"
saveFROCPath="/work/gu14/k77012/M2/FROC/udr/"
mkdir -p ${modelPath}
mkdir -p ${saveFROCPath}

epoch=120 #300
batch_size=64
numSamples=50
model='SSD'
pretrained="ImageNet" #'BigBbox' #"ImageNet" as default

saveDir="/work/gu14/k77012/M2/result/${trainPath}_${validBboxName}_${testBboxName}_${model}_batch${batch_size}_epoch${epoch}_pretrained${pretrained}_${optimizer}/"
mkdir -p ${saveDir} #for saving Dir
mkdir -p "${saveDir}/train"
mkdir -p "${saveDir}/valid"
mkdir -p "${saveDir}/test"

pipenv run python ../WisteriaCodes/train.py ${trainPath} ${validPath} ${testPath} ${trainBbox} ${validBboxName} ${testBboxName} ${modelPath} ${model} ${epoch} ${batch_size} ${numSamples} ${pretrained} ${saveDir} ${saveFROCPath} ${i} ${optimizer} >> /work/gu14/k77012/M2/train_log/${trainPath}_${validBboxName}_${testBboxName}_${model}_epoch${epoch}_batchsize${batch_size}_${pretrained}_${optimizer}.txt
echo 'training_finished'