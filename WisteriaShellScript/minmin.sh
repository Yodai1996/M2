#!/bin/bash

#PJM -g jh170036a
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=47:00:00
#PJM --fs /work,/data
#PJM -N minmin-training
#PJM -o minmin.txt
#PJM -j
#PJM --mail-list suzuki-takahiro596@g.ecc.u-tokyo.ac.jp
#PJM -m b
#PJM -m e

module load cuda/11.1

cd "${PJM_O_WORKDIR}" || exit

fold=2 #modify here. 5-fold CV.

pretrained='BigBbox' #"ImageNet"
metric="FAUC_Pretrained${pretrained}"

#for simulation
boText="bo_minmin_${metric}_${fold}.txt"
bufText="buf_minmin_${metric}_${fold}.txt"

m=1000
normalDir="/work/gk36/k77012/M2/data/NormalDir/" #データプールなら何でもよい。
normalIdList="/work/gk36/k77012/M2/normalFiles${m}.csv"

#for training
epoch=40
batch_size=64
numSamples=50
model='SSD'

validPath='AllDataDir' #'AbnormalDir'
validBboxName="rare_small_bboxInfo_20_${fold}_withNormal"
#validBbox="${validBboxName}.csv"

testPath='AllDataDir' #'AbnormalDir'
testBboxName="rare_small_bboxInfo_81_${fold}_withNormal"
#testBbox="${testBboxName}.csv"

#make dir for model and log
modelPath="/work/gk36/k77012/M2/model/minmin_${metric}/"
mkdir -p ${modelPath}
mkdir -p "/work/gk36/k77012/M2/train_log/minmin_${metric}/"

for i in 6
do
  cd ../bo_io
  ./build/suggest --hm --ha --hpopt -a ei --md 5 --mi ./in/${boText} >> ../${bufText}

  trainPath="sim${i}_${m}"
  abnormalDir="/work/gk36/k77012/M2/data/minmin_${metric}_${fold}/${trainPath}/"
  segMaskDir="/work/gk36/k77012/M2/SegmentationMask/minmin_${metric}_${fold}/mask${i}_${m}/"
  paraDir="/work/gk36/k77012/M2/simDataInfo/paraInfo/minmin_${metric}_${fold}/"
  bboxDir="/work/gk36/k77012/M2/simDataInfo/bboxInfo/minmin_${metric}_${fold}/"
  paraPath="${paraDir}/parameterInfo${i}_${m}.csv"
  bboxPath="${bboxDir}/bboxInfo${i}_${m}.csv"

  mkdir -p ${abnormalDir} #親ディレクトリが無い場合は全て作る、＆、フォルダがすでに存在する場合は何もしない
  mkdir -p ${segMaskDir}
  mkdir -p ${paraDir}
  mkdir -p ${bboxDir}

  savePath="minmin_${metric}/${trainPath}_${validBboxName}_${testBboxName}_${model}_batch${batch_size}_epoch${epoch}_pretrained${pretrained}"

  mkdir -p "/work/gk36/k77012/M2/result/${savePath}/" #for saving Dir
  mkdir -p "/work/gk36/k77012/M2/result/${savePath}/train"
  mkdir -p "/work/gk36/k77012/M2/result/${savePath}/valid"
  mkdir -p "/work/gk36/k77012/M2/result/${savePath}/test"

  pipenv run python ../WisteriaCodes/minmin.py $i ${boText} ${bufText} ${normalIdList} ${normalDir} ${abnormalDir} ${segMaskDir} ${paraPath} ${bboxPath} ${validPath} ${testPath} ${savePath} ${validBboxName} ${testBboxName} ${model} ${pretrained} ${epoch} ${batch_size} ${numSamples} ${modelPath} ${metric} >> ../train_log/minmin_${metric}/sim${i}_${trainPath}_${validBboxName}_${testBboxName}_${model}_epoch${epoch}_batchsize${batch_size}_${pretrained}.txt
  echo $i'_finished'

done