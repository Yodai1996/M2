#!/bin/bash

#PJM -g gu15
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=30:00:00
#PJM --fs /work,/data
#PJM -N finetuning
#PJM -o finetuning.txt
#PJM -j
#PJM --mail-list suzuki-takahiro596@g.ecc.u-tokyo.ac.jp
#PJM -m b
#PJM -m e

#module load cuda/10.2
module load cuda/11.1

cd "${PJM_O_WORKDIR}" || exit

#chage here
fold=1 #modify here. 5-fold CV.
iter=5
curriStart='0.6' #'0.4' #'0.3' #'0.5' #'1.0' #ここ調整。
GDRoptimizer='VSGD' #'Adam' #'VSGD' ###################################################要注意。ここ変えたので。
GDRepoch=120
GDRvariability='0.01' #starting epsilon

optimizer="VSGD" #"Adam" #'VSGD'
variability='0.01' #starting epsilon
epoch=120 #300 #40
batch_size=64
model='SSD'

Path="rare_small_bboxInfo_20_${fold}_withNormal/start${curriStart}_decay1.0_${GDRoptimizer}_FAUC_PretrainedImageNet_variability${GDRvariability}_decay1.0_t1000_v200_iter30_inf30_epoch${GDRepoch}/"
loadModelPath="/work/gu14/k77012/M2/model/curriculumBO/${Path}/model${iter}"
#loadModelPath="/work/gu14/k77012/M2/model/curriculumBO/rare_small_bboxInfo_20_${fold}_withNormal/start${curriStart}_decay1.0_${GDRoptimizer}_FAUC_PretrainedImageNet_variability${GDRvariability}_decay1.0_t1000_v200_iter30_inf30_epoch${GDRepoch}/model${iter}"

#for finetuning
trainPath='AllDataDir' #'AbnormalDir'
validPath='AllDataDir' #'AbnormalDir'
trainBboxName='nonSmall_bboxInfo_655' #'nonSmall_bboxInfo_655_withNormal'
validBboxName='nonSmall_bboxInfo_164_withNormal'

saveModelDir="/work/gu14/k77012/M2/model/finetune/${trainBboxName}_${validBboxName}_${optimizer}_${variability}_epoch${epoch}/${Path}"
mkdir -p ${saveModelDir}
saveModelPath="${saveModelDir}/model${iter}"
trainLog="/work/gu14/k77012/M2/train_log/finetune/${trainBboxName}_${validBboxName}_${optimizer}_${variability}_epoch${epoch}/${Path}"
mkdir -p ${trainLog}

pipenv run python ../WisteriaCodes/finetune.py ${trainPath} ${validPath} ${trainBboxName} ${validBboxName} ${loadModelPath} ${saveModelPath} ${model} ${epoch} ${batch_size} ${optimizer} ${variability} >> ${trainLog}/model${iter}.txt
echo 'finished'






