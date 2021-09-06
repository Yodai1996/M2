#!/bin/bash

#PBS -q h-regular
#PBS -l select=16
#PBS -W group_list=gk36
#PBS -l walltime=10:00:00
#PBS -o main.txt
#PBS -j oe
#PBS -m abe
#PBS -M suzuki-takahiro596@g.ecc.u-tokyo.ac.jp

cd "${PBS_O_WORKDIR}" || exit

. /lustre/gk36/k77012/anaconda3/bin/activate pytorch2

trainPath='AbnormalDir1000'
validPath='AbnormalDir5012'
trainBbox='abnormal1000_bboxinfo.csv'
validBbox='abnormal5012_bboxinfo.csv'
#testPath='AbnormalDir'
#testBbox='abnormal_bboxinfo.csv'


epoch=5
batch_size=16
numSamples=50

model='SSD'
#model='fasterRCNN'
pretrained='pretrained'  #'unpretrained'
mkdir -p "/lustre/gk36/k77012/M2/result/${trainPath}_${validPath}_${model}_batch${batch_size}_epoch${epoch}_${pretrained}/" #for saving Dir
mkdir -p "/lustre/gk36/k77012/M2/result/${trainPath}_${validPath}_${model}_batch${batch_size}_epoch${epoch}_${pretrained}/train"
mkdir -p "/lustre/gk36/k77012/M2/result/${trainPath}_${validPath}_${model}_batch${batch_size}_epoch${epoch}_${pretrained}/valid"
#mkdir -p "/lustre/gk36/k77012/M2/result/${trainPath}_${validPath}_${model}_batch${batch_size}_epoch${epoch}_${pretrained}/test"


#python ../codes/train.py ${trainPath} ${validPath} ${testPath} ${trainBbox} ${validBbox} ${testBbox} ${model} ${epoch} ${batch_size} ${numSamples} ${pretrained}>> ../train_log/log_${trainPath}_${validPath}_${testPath}_${model}_epoch${epoch}_batchsize${batch_size}_${pretrained}.txt
python ../codes/train.py ${trainPath} ${validPath} ${trainBbox} ${validBbox} ${model} ${epoch} ${batch_size} ${numSamples} ${pretrained}>> ../train_log/log_${trainPath}_${validPath}_${model}_epoch${epoch}_batchsize${batch_size}_${pretrained}.txt
echo 'training_finished'
