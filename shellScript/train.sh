#!/bin/bash

#PBS -q h-regular
#PBS -l select=16
#PBS -W group_list=gk36
#PBS -l walltime=10:00:00
#PBS -o train.txt
#PBS -j oe
#PBS -m abe
#PBS -M suzuki-takahiro596@g.ecc.u-tokyo.ac.jp

cd "${PBS_O_WORKDIR}" || exit

. /lustre/gk36/k77012/anaconda3/bin/activate pytorch2

#version=6
#epoch=3
#batch_size=10
#numSamples=50
#
#python ../codes/fasterRcnn.py ${version} ${epoch} ${batch_size} ${numSamples}>> ../results/log_${version}.txt
#echo 'faster-RCNN_finished'

trainPath='AbnormalDir1000'
validPath='AbnormalDir5012'
trainBbox='abnormal1000_bboxinfo.csv'
validBbox='abnormal5012_bboxinfo.csv'
#trainPath='AbnormalDir5012'
#validPath='AbnormalDir1000'
#trainBbox='abnormal5012_bboxinfo.csv'
#validBbox='abnormal1000_bboxinfo.csv'

#trainPath='sim1_abnormal1000' #'train1/1_abnormal1000_1'
#validPath='sim1_abnormal200' #'train1/1_abnormal200_1'
#trainBbox='simDataInfo/bboxInfo/bboxInfo_1.csv'
#validBbox='simDataInfo/bboxInfo/bboxInfo1_200.csv'

#trainPath='sim2_abnormal1000' #'train1/1_abnormal1000_1'
#validPath='sim2_abnormal200' #'train1/1_abnormal200_1'
#trainBbox='simDataInfo/bboxInfo/bboxInfo_2.csv'
#validBbox='simDataInfo/bboxInfo/bboxInfo2_200.csv'

#testも用意したので
testPath='AbnormalDir'
testBbox='abnormal_bboxinfo.csv'

epoch=12
batch_size=16
numSamples=50

#mkdir -p "/lustre/gk36/k77012/M2/results/${trainPath}_${validPath}_batch${batch_size}_epoch${epoch}/" #for saving Dir
#mkdir -p "/lustre/gk36/k77012/M2/results/${trainPath}_${validPath}_batch${batch_size}_epoch${epoch}/train"
#mkdir -p "/lustre/gk36/k77012/M2/results/${trainPath}_${validPath}_batch${batch_size}_epoch${epoch}/valid"
#mkdir -p "/lustre/gk36/k77012/M2/results/${trainPath}_${validPath}_batch${batch_size}_epoch${epoch}/test"

model='SSD'
#mkdir -p "/lustre/gk36/k77012/M2/results/${trainPath}_${validPath}_${model}_batch${batch_size}_epoch${epoch}/" #for saving Dir
#mkdir -p "/lustre/gk36/k77012/M2/results/${trainPath}_${validPath}_${model}_batch${batch_size}_epoch${epoch}/train"
#mkdir -p "/lustre/gk36/k77012/M2/results/${trainPath}_${validPath}_${model}_batch${batch_size}_epoch${epoch}/valid"
#mkdir -p "/lustre/gk36/k77012/M2/results/${trainPath}_${validPath}_${model}_batch${batch_size}_epoch${epoch}/test"
mkdir -p "/lustre/gk36/k77012/M2/result/${trainPath}_${validPath}_${model}_batch${batch_size}_epoch${epoch}/" #for saving Dir
mkdir -p "/lustre/gk36/k77012/M2/result/${trainPath}_${validPath}_${model}_batch${batch_size}_epoch${epoch}/train"
mkdir -p "/lustre/gk36/k77012/M2/result/${trainPath}_${validPath}_${model}_batch${batch_size}_epoch${epoch}/valid"
mkdir -p "/lustre/gk36/k77012/M2/result/${trainPath}_${validPath}_${model}_batch${batch_size}_epoch${epoch}/test"


#python ../codes/fasterRcnn.py ${trainPath} ${validPath} ${trainBbox} ${validBbox} ${epoch} ${batch_size} ${numSamples}>> ../results/log_${trainPath}_${validPath}_epoch${epoch}_batchsize${batch_size}.txt
#python ../codes/fasterRcnn.py ${trainPath} ${validPath} ${testPath} ${trainBbox} ${validBbox} ${testBbox} ${epoch} ${batch_size} ${numSamples}>> ../results/log_${trainPath}_${validPath}_${testPath}_epoch${epoch}_batchsize${batch_size}.txt
#python ../codes/ssd.py ${trainPath} ${validPath} ${testPath} ${trainBbox} ${validBbox} ${testBbox} ${model} ${epoch} ${batch_size} ${numSamples}>> ../results/log_${trainPath}_${validPath}_${testPath}_${model}_epoch${epoch}_batchsize${batch_size}.txt
python ../codes/ssd.py ${trainPath} ${validPath} ${testPath} ${trainBbox} ${validBbox} ${testBbox} ${model} ${epoch} ${batch_size} ${numSamples}>> ../train_log/log_${trainPath}_${validPath}_${testPath}_${model}_epoch${epoch}_batchsize${batch_size}.txt
echo 'faster-RCNN_finished'
