#!/bin/bash

#PJM -g gu15
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=30:00:00
#PJM --fs /work,/data
#PJM -N visualize
#PJM -o visualize.txt
#PJM -j
#PJM --mail-list suzuki-takahiro596@g.ecc.u-tokyo.ac.jp
#PJM -m b
#PJM -m e

#module load cuda/10.2
module load cuda/11.1

cd "${PJM_O_WORKDIR}" || exit

#pipenv shell #activate the virtual environment

#modify here
fold=1
score="0.6"
iter=5
pretrain="ImageNet"
epoch=120

#acceptable FPs per Image
FPsI="0.2"

#test data
dataPath='AllDataDir'
validBboxName="rare_small_bboxInfo_20_${fold}_withNormal"
testBboxName="rare_small_bboxInfo_81_${fold}_withNormal"

Path="curriculumBO/rare_small_bboxInfo_20_${fold}_withNormal/start${score}_decay1.0_VSGD_FAUC_Pretrained${pretrain}_variability0.01_decay1.0_t1000_v200_iter30_inf30_epoch${epoch}/model${iter}"
#Path="pretrain/model_nonSmall_bboxInfo_655_nonSmall_bboxInfo_164_withNormal_VSGD_0.01_120"

loadModelPath="/work/jh170036a/k77012/M2/model/${Path}"
#loadModelPath="/work/gk36/k77012/M2/model/${Path}"

saveDir="/work/jh170036a/k77012/M2/visualize/${Path}/" #saveDir="/work/gk36/k77012/M2/result/${Path}/${dataBboxName}/"
mkdir -p ${saveDir} #for saving Dir
mkdir -p "${saveDir}/${validBboxName}/"
mkdir -p "${saveDir}/${testBboxName}/"

LogDir="/work/jh170036a/k77012/M2/log/${Path}/"
mkdir -p ${LogDir}

pipenv run python ../WisteriaCodes/visualize.py ${dataPath} ${validBboxName} ${testBboxName} ${loadModelPath} ${saveDir} ${FPsI} >> "${LogDir}/thres_FPsI${FPsI}.txt"
echo 'visualizing_finished'