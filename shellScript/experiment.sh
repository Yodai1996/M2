#!/bin/bash

#PBS -q l-regular
#PBS -l select=4
#PBS -W group_list=gk36
#PBS -l walltime=59:00:00
#PBS -o experiment.txt
#PBS -j oe
#PBS -m abe
#PBS -M suzuki-takahiro596@g.ecc.u-tokyo.ac.jp

cd "${PBS_O_WORKDIR}" || exit

. /lustre/gk36/k77012/anaconda3/bin/activate pytorch2

boText="bo_fasterRCNN.txt"
bufText="buf.txt"
epoch=2
batch_size=10
numSamples=30
scoreThres=0.3  #used in inference, you can set as you like.

for i in 8
do
  cd ../bo_io
  ./build/suggest --hm --ha --hpopt -a ei --md 7 --mi ./in/${boText} >> ../${bufText}
  cd ..
  cd ./data
  mkdir Mask/mask_$i
  mkdir train_$i
  mkdir train_$i/1_abnormal1000_$i
  cd ..
  cd results
  mkdir train_$i
  mkdir train_$i/train
  mkdir train_$i/valid
  cd ..
  python ./codes/main.py $i ${boText} ${bufText} ${epoch} ${batch_size} ${numSamples} ${scoreThres}>> ./results/log_$i.txt
  echo $i'_finished'
done