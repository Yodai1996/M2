#!/bin/bash

#PJM -g gk36
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=30:00:00
#PJM --fs /work,/data
#PJM -N resulting
#PJM -o calcMetric.txt
#PJM -j
#PJM --mail-list suzuki-takahiro596@g.ecc.u-tokyo.ac.jp
#PJM -m b
#PJM -m e

#module load cuda/10.2
module load cuda/11.1

cd "${PJM_O_WORKDIR}" || exit

#pipenv shell #activate the virtual environment

#modify here
fold=5
score="0.5"
iter=8
pretrain="ImageNet" #"BigBbox" #"ImageNet"
epoch=120

#test data
dataPath='AllDataDir'
validBboxName="rare_small_bboxInfo_20_${fold}_withNormal200"
testBboxName="rare_small_bboxInfo_81_${fold}_withNormal810" #"rare_small_bboxInfo_81_${fold}_withNormal"

Path="curriculumBO/rare_small_bboxInfo_20_${fold}_withNormal/start${score}_decay1.0_VSGD_FAUC_Pretrained${pretrain}_variability0.01_decay1.0_t1000_v200_iter30_inf30_epoch${epoch}/model${iter}"
#Path="pretrain/model_nonSmall_bboxInfo_655_nonSmall_bboxInfo_164_withNormal_VSGD_0.01_120"

loadModelPath="/work/jh170036a/k77012/M2/model/${Path}"
#loadModelPath="/work/gk36/k77012/M2/model/${Path}"

LogDir="/work/jh170036a/k77012/M2/log_10:1/${Path}/"
mkdir -p ${LogDir}

pipenv run python ../WisteriaCodes/calcMetric.py ${dataPath} ${validBboxName} ${testBboxName} ${Path} ${loadModelPath} ${pretrain} >> "${LogDir}/result_${testBboxName}.txt"
echo 'resulting_finished'