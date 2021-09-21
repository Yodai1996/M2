#!/bin/bash

#PBS -q h-regular
#PBS -l select=1
#PBS -W group_list=gk36
#PBS -l walltime=10:00:00
#PBS -o train_FN_indexOutput.txt
#PBS -j oe
#PBS -m abe
#PBS -M suzuki-takahiro596@g.ecc.u-tokyo.ac.jp

cd "${PBS_O_WORKDIR}" || exit

. /lustre/gk36/k77012/anaconda3/bin/activate pytorch2

#change here
ver=5 #ver \in [1,5]

trainPath='AbnormalDir5880' #data poolなので広めであればよい
trainBbox="abnormal4704_bboxinfo_${ver}.csv"
validPath='AbnormalDir5880' #data poolなので広めであればよい
validBbox="abnormal1176_bboxinfo_${ver}.csv"

#使わないので適当でよい。
testPath='AbnormalDir5012'
testBbox='abnormal5012_bboxinfo.csv'

epoch=40
batch_size=64
numSamples=50

model='SSD'
#model='fasterRCNN'
pretrained='pretrained'  #'unpretrained'
mkdir -p "/lustre/gk36/k77012/M2/result/${trainPath}_${validPath}_${model}_batch${batch_size}_epoch${epoch}_${pretrained}_${ver}/" #for saving Dir
mkdir -p "/lustre/gk36/k77012/M2/result/${trainPath}_${validPath}_${model}_batch${batch_size}_epoch${epoch}_${pretrained}_${ver}/train"
mkdir -p "/lustre/gk36/k77012/M2/result/${trainPath}_${validPath}_${model}_batch${batch_size}_epoch${epoch}_${pretrained}_${ver}/valid"
mkdir -p "/lustre/gk36/k77012/M2/result/${trainPath}_${validPath}_${model}_batch${batch_size}_epoch${epoch}_${pretrained}_${ver}/original"


python ../codes/train_FN_indexOutput.py ${trainPath} ${validPath} ${testPath} ${trainBbox} ${validBbox} ${testBbox} ${model} ${epoch} ${batch_size} ${numSamples} ${pretrained} ${ver} >> ../train_log/${trainPath}_${validPath}_${testPath}_${model}_epoch${epoch}_batchsize${batch_size}_${pretrained}_${ver}.txt
python ../codes/extract_FN_samples.py ${ver}
echo 'training_finished'