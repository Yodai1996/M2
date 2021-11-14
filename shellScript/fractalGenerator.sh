#!/bin/bash

#PBS -q l-regular
#PBS -l select=2
#PBS -W group_list=gk36
#PBS -l walltime=19:00:00
#PBS -o fractalGenerator.txt
#PBS -j oe
#PBS -m abe
#PBS -M suzuki-takahiro596@g.ecc.u-tokyo.ac.jp

cd "${PBS_O_WORKDIR}" || exit

. /lustre/gk36/k77012/anaconda3/bin/activate pytorch

m=200 #1000
normalDir="/lustre/gk36/k77012/M2/data/NormalDir${m}/" #"/lustre/gk36/k77012/M2/data/NormalDir1000/0_normal1000/"
normalIdList="/lustre/gk36/k77012/M2/normalIdList${m}.csv"

for i in 5 #4 #3 #1 #2 #6
do
  abnormalDir="/lustre/gk36/k77012/M2/data/sim${i}_${m}/" #"/lustre/gk36/k77012/M2/data/train_${i}/1_abnormal${m}_${i}/"
  segMaskDir="/lustre/gk36/k77012/M2/SegmentationMask/mask${i}_${m}/" #"/lustre/gk36/k77012/M2/data/Mask/mask${i}_${m}/"
  saveParaPath="/lustre/gk36/k77012/M2/simDataInfo/paraInfo/parameterInfo${i}_${m}.csv"
  saveBboxPath="/lustre/gk36/k77012/M2/simDataInfo/bboxInfo/bboxInfo${i}_${m}.csv"

  mkdir -p ${abnormalDir} #train_${i}が作られていること前提ではない。親ディレクトリが無い場合は全て作る、＆、フォルダがすでに存在する場合は何もしない
  mkdir -p ${segMaskDir}

  python ../codes/fractalGenerator.py ${normalIdList} ${normalDir} ${abnormalDir} ${segMaskDir} ${saveParaPath} ${saveBboxPath} 40 130 5 5 0.7 2 0.7 0.6 #80 200 4 5 0.3 4 1 0.2 #50 190 4 5 0.5 3 0.6 0.5 #60 150 3 5 0.6 3 0.5 0.4 #70 180 4 5 0.5 3 0.3 0.7   #50 120 2 5 0.4 3 0.4 0.7\
  echo $i'_finished'
done
