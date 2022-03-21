#!/bin/bash

#PJM -g jh170036a
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=30:00:00
#PJM --fs /work,/data
#PJM -N resulting
#PJM -o result_minmin_pretrained.txt
#PJM -j
#PJM --mail-list suzuki-takahiro596@g.ecc.u-tokyo.ac.jp
#PJM -m b
#PJM -m e

#module load cuda/10.2
module load cuda/11.1

cd "${PJM_O_WORKDIR}" || exit

#pipenv shell #activate the virtual environment

#modify here

fold=1 #modify here. 5-fold CV.
iter=39 #21
preEpoch=400 #120 #120 #400
preOptimizer='SGD' #"Adam" #"Adam" #'SGD'
pretrain="model_nonSmall_bboxInfo_655_nonSmall_bboxInfo_164_withNormal_${preOptimizer}_0.01_${preEpoch}" #"ImageNet" #'BigBbox' #"ImageNet"
#metric="FAUC_Pretrained${pretrain}"

optimizer="SGD" #"Adam" #"SGD" #"Adam" #"VSGD" #epsilon=0.01
epoch=120 #40だと少なかった

#test data
dataPath='AllDataDir'
dataBboxName="rare_small_bboxInfo_81_${fold}_withNormal"

Path="minmin_FAUC_Pretrained${pretrain}_${optimizer}_epoch${epoch}/model_version${iter}_rare_small_bboxInfo_20_${fold}_withNormal_pretrained${pretrain}_epoch${epoch}"

loadModelPath="/work/gu14/k77012/M2/model/${Path}"

saveFROCPath="/work/gu14/k77012/M2/FROC/${Path}/${dataBboxName}/" #saveFROCPath="/work/gk36/k77012/M2/FROC/${Path}/${dataBboxName}/"
mkdir -p ${saveFROCPath}

#numSamples=50
saveDir="/work/gu14/k77012/M2/result/${Path}/${dataBboxName}/" #saveDir="/work/gk36/k77012/M2/result/${Path}/${dataBboxName}/"
mkdir -p ${saveDir} #for saving Dir
#mkdir -p "${saveDir}"

LogDir="/work/gu14/k77012/M2/log/${Path}/"
mkdir -p ${LogDir}

pipenv run python ../WisteriaCodes/result.py ${dataPath} ${dataBboxName} ${Path} ${loadModelPath} ${saveDir} ${saveFROCPath} >> "${LogDir}/result_${dataBboxName}.txt"
echo 'resulting_finished'