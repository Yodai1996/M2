#!/bin/bash

#PBS -q l-regular
#PBS -l select=1
#PBS -W group_list=gk36
#PBS -l walltime=59:00:00
#PBS -o minmin.txt
#PBS -j oe
#PBS -m abe
#PBS -M suzuki-takahiro596@g.ecc.u-tokyo.ac.jp

cd "${PBS_O_WORKDIR}" || exit

. /lustre/gk36/k77012/anaconda3/bin/activate pytorch2

#for simulation
boText="bo_minmin.txt"
bufText="buf_minmin.txt"

m=1000
normalDir="/lustre/gk36/k77012/M2/data/NormalDir${m}/"
normalIdList="/lustre/gk36/k77012/M2/normalIdList${m}.csv"

#for training
epoch=40
batch_size=64
numSamples=50
model='SSD'
pretrained='pretrained'  #'unpretrained'

validPath='AbnormalDir10'
validBbox='abnormal10_bboxinfo.csv'
testPath='AbnormalDir5012'
testBbox='abnormal5012_bboxinfo.csv'

for i in 21 22 23
do
  cd ../bo_io
  ./build/suggest --hm --ha --hpopt -a ei --md 7 --mi ./in/${boText} >> ../${bufText}

  #../bo_io/build/suggest --hm --ha --hpopt -a ei --md 7 --mi ../bo_io/in/${boText} >> ../${bufText}

  trainPath="sim${i}_${m}" #abnormalDirと同一なため、引数として渡す必要は無い.
  #trainBbox="simDataInfo/bboxInfo/minmin/bboxInfo${i}_${m}.csv" #saveBboxPathと同一なため本来は不要だが、実装上これも引数として渡すことにする。(これは引数としても不要そう。)

  abnormalDir="/lustre/gk36/k77012/M2/data/minmin/${trainPath}/"
  segMaskDir="/lustre/gk36/k77012/M2/SegmentationMask/minmin/mask${i}_${m}/"
  saveParaPath="/lustre/gk36/k77012/M2/simDataInfo/paraInfo/minmin/parameterInfo${i}_${m}.csv"
  saveBboxPath="/lustre/gk36/k77012/M2/simDataInfo/bboxInfo/minmin/bboxInfo${i}_${m}.csv"

  mkdir -p ${abnormalDir} #親ディレクトリが無い場合は全て作る、＆、フォルダがすでに存在する場合は何もしない
  mkdir -p ${segMaskDir}

  savePath="minmin/${trainPath}_${validPath}_${model}_batch${batch_size}_epoch${epoch}_${pretrained}"

  mkdir -p "/lustre/gk36/k77012/M2/result/${savePath}/" #for saving Dir
  mkdir -p "/lustre/gk36/k77012/M2/result/${savePath}/train"
  mkdir -p "/lustre/gk36/k77012/M2/result/${savePath}/valid"
  mkdir -p "/lustre/gk36/k77012/M2/result/${savePath}/test"

  python ../codes/minmin.py $i ${boText} ${bufText} ${normalIdList} ${normalDir} ${abnormalDir} ${segMaskDir} ${saveParaPath} ${saveBboxPath} ${validPath} ${testPath} ${savePath} ${validBbox} ${testBbox} ${model} ${pretrained} ${epoch} ${batch_size} ${numSamples} >> ../train_log/minmin/sim${i}_${trainPath}_${validPath}_${testPath}_${model}_epoch${epoch}_batchsize${batch_size}_${pretrained}.txt
  echo $i'_finished'

done