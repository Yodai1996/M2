#!/bin/bash

#PBS -q h-regular
#PBS -l select=1
#PBS -W group_list=gk36
#PBS -l walltime=10:00:00
#PBS -o train_continual.txt
#PBS -j oe
#PBS -m abe
#PBS -M suzuki-takahiro596@g.ecc.u-tokyo.ac.jp

cd "${PBS_O_WORKDIR}" || exit

. /lustre/gk36/k77012/anaconda3/bin/activate pytorch2

#trainPath='AbnormalDir1000'
#validPath='AbnormalDir5012'
#trainBbox='abnormal1000_bboxinfo.csv'
#validBbox='abnormal5012_bboxinfo.csv'

i=1
trainPath="sim${i}_1000" #'train1/1_abnormal1000_1'
validPath="sim${i}_200" #'train1/1_abnormal200_1'
trainBbox="simDataInfo/bboxInfo/bboxInfo${i}_1000.csv" #'simDataInfo/bboxInfo/bboxInfo_1.csv'
validBbox="simDataInfo/bboxInfo/bboxInfo${i}_200.csv"

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


testPath='AbnormalDir5012'
testBbox='abnormal5012_bboxinfo.csv'


epoch=40
batch_size=64
numSamples=50

model='SSD'
#pretrained='pretrained'  #'unpretrained'
mkdir -p "/lustre/gk36/k77012/M2/result/continual/${trainPath}_${validPath}_${model}_batch${batch_size}_epoch${epoch}/" #for saving Dir
mkdir -p "/lustre/gk36/k77012/M2/result/continual/${trainPath}_${validPath}_${model}_batch${batch_size}_epoch${epoch}/train"
mkdir -p "/lustre/gk36/k77012/M2/result/continual/${trainPath}_${validPath}_${model}_batch${batch_size}_epoch${epoch}/valid"
mkdir -p "/lustre/gk36/k77012/M2/result/continual/${trainPath}_${validPath}_${model}_batch${batch_size}_epoch${epoch}/test"


python ../codes/train_continual.py ${trainPath} ${validPath} ${testPath} ${trainBbox} ${validBbox} ${testBbox} ${model} ${epoch} ${batch_size} ${numSamples} ${i}>> ../train_log/model${version}_${trainPath}_${validPath}_${testPath}_${model}_epoch${epoch}_batchsize${batch_size}.txt
#python ../codes/train.py ${trainPath} ${validPath} ${trainBbox} ${validBbox} ${model} ${epoch} ${batch_size} ${numSamples} ${pretrained}>> ../train_log/${trainPath}_${validPath}_${model}_epoch${epoch}_batchsize${batch_size}_${pretrained}.txt
echo 'training_finished'