#!/bin/bash

#PJM -g jh170036a
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=40:00:00
#PJM --fs /work,/data
#PJM -N minmin-training
#PJM -o minmin.txt
#PJM -j
#PJM --mail-list suzuki-takahiro596@g.ecc.u-tokyo.ac.jp
#PJM -m b
#PJM -m e

module load cuda/11.1

cd "${PJM_O_WORKDIR}" || exit

metric='Dice'

#for simulation
boText="bo_minmin_${metric}.txt"
bufText="buf_minmin_${metric}.txt"

m=1000
normalDir="/work/gk36/k77012/M2/data/NormalDir${m}/"
normalIdList="/work/gk36/k77012/M2/normalIdList${m}.csv"

#for training
epoch=40
batch_size=64
numSamples=50
model='SSD'
pretrained='pretrained'  #'unpretrained'

validPath='AbnormalDir10'
validBbox='abnormal10_bboxinfo.csv'
testPath='AbnormalDir' #大は小を兼ねるはず #'AbnormalDir4880' #'AbnormalDir5012'
testBbox='abnormal5870_bboxinfo.csv' #'abnormal4880_bboxinfo.csv' #'abnormal5012_bboxinfo.csv'

#make dir for model and log
modelPath="/work/gk36/k77012/M2/model/minmin_${metric}/"
mkdir -p ${modelPath}
mkdir -p "/work/gk36/k77012/M2/train_log/minmin_${metric}/"

for i in 6
do
  cd ../bo_io
  ./build/suggest --hm --ha --hpopt -a ei --md 7 --mi ./in/${boText} >> ../${bufText}

  trainPath="sim${i}_${m}"
  abnormalDir="/work/gk36/k77012/M2/data/minmin_${metric}/${trainPath}/"
  segMaskDir="/work/gk36/k77012/M2/SegmentationMask/minmin_${metric}/mask${i}_${m}/"
  paraDir="/work/gk36/k77012/M2/simDataInfo/paraInfo/minmin_${metric}/"
  bboxDir="/work/gk36/k77012/M2/simDataInfo/bboxInfo/minmin_${metric}/"
  paraPath="${paraDir}/parameterInfo${i}_${m}.csv"
  bboxPath="${bboxDir}/bboxInfo${i}_${m}.csv"

  mkdir -p ${abnormalDir} #親ディレクトリが無い場合は全て作る、＆、フォルダがすでに存在する場合は何もしない
  mkdir -p ${segMaskDir}
  mkdir -p ${paraDir}
  mkdir -p ${bboxDir}

  savePath="minmin_${metric}/${trainPath}_${validPath}_${model}_batch${batch_size}_epoch${epoch}_${pretrained}"

  mkdir -p "/work/gk36/k77012/M2/result/${savePath}/" #for saving Dir
  mkdir -p "/work/gk36/k77012/M2/result/${savePath}/train"
  mkdir -p "/work/gk36/k77012/M2/result/${savePath}/valid"
  mkdir -p "/work/gk36/k77012/M2/result/${savePath}/test"

  pipenv run python ../WisteriaCodes/minmin.py $i ${boText} ${bufText} ${normalIdList} ${normalDir} ${abnormalDir} ${segMaskDir} ${paraPath} ${bboxPath} ${validPath} ${testPath} ${savePath} ${validBbox} ${testBbox} ${model} ${pretrained} ${epoch} ${batch_size} ${numSamples} ${modelPath} ${metric} >> ../train_log/minmin_${metric}/sim${i}_${trainPath}_${validPath}_${testPath}_${model}_epoch${epoch}_batchsize${batch_size}_${pretrained}.txt
  echo $i'_finished'

done