#!/bin/bash

#PJM -g jh170036a
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=30:00:00
#PJM --fs /work,/data
#PJM -N tryinfering
#PJM -o try.txt
#PJM -j
#PJM --mail-list suzuki-takahiro596@g.ecc.u-tokyo.ac.jp
#PJM -m b
#PJM -m e

#module load cuda/10.2
module load cuda/11.1

cd "${PJM_O_WORKDIR}" || exit

#pipenv shell #activate the virtual environment

trainPath='AllDataDir' #'AbnormalDir'
validPath='AllDataDir' #'AbnormalDir'
trainBboxName='nonSmall_bboxInfo_655' #'nonSmall_bboxInfo_655_withNormal'
#validBboxName='nonSmall_bboxInfo_164_withNormal'
validBboxName='rare_small_bboxInfo_20_1_withNormal'

#trainBbox="${trainBboxName}.csv"
#validBbox="${validBboxName}.csv"

loadModelPath="/work/gk36/k77012/M2/model/model_nonSmall_bboxInfo_655_nonSmall_bboxInfo_164_withNormal_VSGD_0.01_120"
saveFROCPath="/work/gk36/k77012/M2/FROC/"

optimizer='VSGD'
variability='0.005' #starting epsilon
#ignoreBigBbox="True" #ignore

epoch=0 #120 #40
batch_size=64
numSamples=50

model='SSD'
pretrained='pretrained'  #'unpretrained'
saveDir="/work/gk36/k77012/M2/result/${trainBboxName}_${validBboxName}_${model}_batch${batch_size}_epoch${epoch}_${pretrained}_${optimizer}_${variability}/"
mkdir -p ${saveDir} #for saving Dir
mkdir -p "${saveDir}/train"
mkdir -p "${saveDir}/valid"

pipenv run python ../WisteriaCodes/try.py ${trainPath} ${validPath} ${trainBboxName} ${validBboxName} ${loadModelPath} ${model} ${epoch} ${batch_size} ${numSamples} ${pretrained} ${saveDir} ${saveFROCPath} ${optimizer} ${variability} >> ../train_log/${trainBboxName}_${validBboxName}_${model}_epoch${epoch}_batchsize${batch_size}_${pretrained}_${optimizer}_${variability}.txt
echo 'training_finished'