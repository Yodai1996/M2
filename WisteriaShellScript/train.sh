#!/bin/bash

#PJM -g jh170036a
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=10:00:00
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

#trainPath='AbnormalDir1000'
#validPath='AbnormalDir4880'
#trainBbox='abnormal1000_bboxinfo.csv'
#validBbox='abnormal4880_bboxinfo.csv'

i=3 #1,2,3,4,5
trainPath="sim${i}_1000"
trainBbox="simDataInfo/bboxInfo/bboxInfo${i}_1000.csv"
validPath='AbnormalDir10'
validBbox='abnormal10_bboxinfo.csv'


#trainPath='sim2_abnormal1000' #'train1/1_abnormal1000_1'
#validPath='sim2_abnormal200' #'train1/1_abnormal200_1'
#trainBbox='simDataInfo/bboxInfo/bboxInfo_2.csv'
#validBbox='simDataInfo/bboxInfo/bboxInfo2_200.csv'

#trainPath='sim5_1000' #'train1/1_abnormal1000_1'
#validPath='sim5_200' #'train1/1_abnormal200_1'
#trainBbox='simDataInfo/bboxInfo/bboxInfo5_1000.csv' #'simDataInfo/bboxInfo/bboxInfo_1.csv'
#validBbox='simDataInfo/bboxInfo/bboxInfo5_200.csv'

#testPath='AbnormalDir'
#testBbox='abnormal_bboxinfo.csv'

#trainPath='sim5_1000' #'train1/1_abnormal1000_1'
#trainBbox='simDataInfo/bboxInfo/bboxInfo5_1000.csv' #'simDataInfo/bboxInfo/bboxInfo_1.csv'
#validPath='AbnormalDir10'
#validBbox='abnormal10_bboxinfo.csv'

#testPath='AbnormalDir5880'
testPath='AbnormalDir' #大は小を兼ねるはず
testBbox='abnormal5880_bboxinfo.csv'

modelPath="/work/gk36/k77012/M2/model/"

epoch=40
batch_size=64
numSamples=50

model='SSD'
pretrained='pretrained'  #'unpretrained'
mkdir -p "/work/gk36/k77012/M2/result/${trainPath}_${validPath}_${model}_batch${batch_size}_epoch${epoch}_${pretrained}_Dice/" #for saving Dir
mkdir -p "/work/gk36/k77012/M2/result/${trainPath}_${validPath}_${model}_batch${batch_size}_epoch${epoch}_${pretrained}_Dice/train"
mkdir -p "/work/gk36/k77012/M2/result/${trainPath}_${validPath}_${model}_batch${batch_size}_epoch${epoch}_${pretrained}_Dice/valid"
mkdir -p "/work/gk36/k77012/M2/result/${trainPath}_${validPath}_${model}_batch${batch_size}_epoch${epoch}_${pretrained}_Dice/test"

#python ../codes/train.py ${trainPath} ${validPath} ${testPath} ${trainBbox} ${validBbox} ${testBbox} ${model} ${epoch} ${batch_size} ${numSamples} ${pretrained}>> ../train_log/${trainPath}_${validPath}_${testPath}_${model}_epoch${epoch}_batchsize${batch_size}_${pretrained}.txt
pipenv run python ../WisteriaCodes/train.py ${trainPath} ${validPath} ${testPath} ${trainBbox} ${validBbox} ${testBbox} ${modelPath} ${model} ${epoch} ${batch_size} ${numSamples} ${pretrained} ${i}>> ../train_log/${trainPath}_${validPath}_${testPath}_${model}_epoch${epoch}_batchsize${batch_size}_${pretrained}_Dice.txt
echo 'training_finished'