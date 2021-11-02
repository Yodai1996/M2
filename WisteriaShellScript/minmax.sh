#!/bin/bash

#PBS -q l-regular
#PBS -l select=1
#PBS -W group_list=gk36
#PBS -l walltime=59:00:00
#PBS -o minmax.txt
#PBS -j oe
#PBS -m abe
#PBS -M suzuki-takahiro596@g.ecc.u-tokyo.ac.jp

cd "${PBS_O_WORKDIR}" || exit

. /lustre/gk36/k77012/anaconda3/bin/activate pytorch2

#for simulation
boText="bo_minmax.txt"
bufText="buf_minmax.txt"

numTrain=1000
normalDir="/lustre/gk36/k77012/M2/data/NormalDir${numTrain}/"
normalIdList="/lustre/gk36/k77012/M2/normalIdList${numTrain}.csv"

#for validation
numValid=200
normalDir2="/lustre/gk36/k77012/M2/data/NormalDir${numValid}/"
normalIdList2="/lustre/gk36/k77012/M2/normalIdList${numValid}.csv"


#for training
epoch=40
batch_size=64
numSamples=50
model='SSD'
pretrained='pretrained'  #'unpretrained'

testPath='AbnormalDir5012' #'AbnormalDir'でもよい
testBbox='abnormal5012_bboxinfo.csv' #'abnormal_bboxinfo.csv'

for i in 10 11 12
do
  cd ../bo_io
  ./build/suggest --hm --ha --hpopt -a ei --md 7 --mi ./in/${boText} >> ../${bufText}

  #../bo_io/build/suggest --hm --ha --hpopt -a ei --md 7 --mi ../bo_io/in/${boText} >> ../${bufText}

  trainPath="sim${i}_${numTrain}" #abnormalDirと同一なため、引数として渡す必要は無い.
  abnormalDir="/lustre/gk36/k77012/M2/data/minmax/${trainPath}/"
  segMaskDir="/lustre/gk36/k77012/M2/SegmentationMask/minmax/mask${i}_${numTrain}/"
  saveParaPath="/lustre/gk36/k77012/M2/simDataInfo/paraInfo/minmax/parameterInfo${i}_${numTrain}.csv" #これらのDirは事前に作っておく必要がある。
  saveBboxPath="/lustre/gk36/k77012/M2/simDataInfo/bboxInfo/minmax/bboxInfo${i}_${numTrain}.csv"

  mkdir -p ${abnormalDir} #親ディレクトリが無い場合は全て作る、＆、フォルダがすでに存在する場合は何もしない
  mkdir -p ${segMaskDir}

  #same for the validation data
  validPath="sim${i}_${numValid}" #abnormalDir2と同一なため、引数として渡す必要は無い.
  abnormalDir2="/lustre/gk36/k77012/M2/data/minmax/${validPath}/"
  segMaskDir2="/lustre/gk36/k77012/M2/SegmentationMask/minmax/mask${i}_${numValid}/"
  saveParaPath2="/lustre/gk36/k77012/M2/simDataInfo/paraInfo/minmax/parameterInfo${i}_${numValid}.csv"
  saveBboxPath2="/lustre/gk36/k77012/M2/simDataInfo/bboxInfo/minmax/bboxInfo${i}_${numValid}.csv"

  mkdir -p ${abnormalDir2} #親ディレクトリが無い場合は全て作る、＆、フォルダがすでに存在する場合は何もしない
  mkdir -p ${segMaskDir2}

  savePath="minmax/${trainPath}_${validPath}_${model}_batch${batch_size}_epoch${epoch}_${pretrained}"

  mkdir -p "/lustre/gk36/k77012/M2/result/${savePath}/" #for saving Dir
  mkdir -p "/lustre/gk36/k77012/M2/result/${savePath}/train"
  mkdir -p "/lustre/gk36/k77012/M2/result/${savePath}/valid"
  mkdir -p "/lustre/gk36/k77012/M2/result/${savePath}/test"

  python ../codes/minmax.py $i ${boText} ${bufText} ${normalIdList} ${normalDir} ${abnormalDir} ${segMaskDir} ${saveParaPath} ${saveBboxPath} ${normalIdList2} ${normalDir2} ${abnormalDir2} ${segMaskDir2} ${saveParaPath2} ${saveBboxPath2} ${testPath} ${savePath} ${testBbox} ${model} ${pretrained} ${epoch} ${batch_size} ${numSamples} >> ../train_log/minmax/sim${i}_${trainPath}_${validPath}_${testPath}_${model}_epoch${epoch}_batchsize${batch_size}_${pretrained}.txt
  echo $i'_finished'

done