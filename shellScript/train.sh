#!/bin/bash

#PBS -q h-regular
#PBS -l select=1
#PBS -W group_list=gk36
#PBS -l walltime=10:00:00
#PBS -o train.txt
#PBS -j oe
#PBS -m abe
#PBS -M suzuki-takahiro596@g.ecc.u-tokyo.ac.jp

cd "${PBS_O_WORKDIR}" || exit

. /lustre/gk36/k77012/anaconda3/bin/activate pytorch2

trainPath='SmallBboxesAbnormalDir_SizeUnder130_except100' #'AbnormalDir1000'
validPath='AbnormalDir4880'
trainBbox='abnormal_under130_smallbboxinfo_forHanaoka.csv' #'abnormal1000_bboxinfo.csv'
validBbox='abnormal4880_bboxinfo.csv'

#trainPath='sim1_abnormal1000' #'train1/1_abnormal1000_1'
#validPath='sim1_abnormal200' #'train1/1_abnormal200_1'
#trainBbox='simDataInfo/bboxInfo/bboxInfo1_1000.csv' #'simDataInfo/bboxInfo/bboxInfo_1.csv'
#validBbox='simDataInfo/bboxInfo/bboxInfo1_200.csv'

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

#trainPath='sim5_1000' #'train1/1_abnormal1000_1'
#trainBbox='simDataInfo/bboxInfo/bboxInfo5_1000.csv' #'simDataInfo/bboxInfo/bboxInfo_1.csv'
#validPath='AbnormalDir10'
#validBbox='abnormal10_bboxinfo.csv'

testPath='AbnormalDir5880'
testBbox='abnormal5880_bboxinfo.csv'


epoch=0 #40
batch_size=64
numSamples=100 #50

model='SSD'
#model='fasterRCNN'
pretrained='pretrained'  #'unpretrained'
mkdir -p "/lustre/gk36/k77012/M2/result/${trainPath}_${validPath}_${model}_batch${batch_size}_epoch${epoch}_${pretrained}/" #for saving Dir
mkdir -p "/lustre/gk36/k77012/M2/result/${trainPath}_${validPath}_${model}_batch${batch_size}_epoch${epoch}_${pretrained}/train"
mkdir -p "/lustre/gk36/k77012/M2/result/${trainPath}_${validPath}_${model}_batch${batch_size}_epoch${epoch}_${pretrained}/valid"
mkdir -p "/lustre/gk36/k77012/M2/result/${trainPath}_${validPath}_${model}_batch${batch_size}_epoch${epoch}_${pretrained}/test"


python ../codes/train.py ${trainPath} ${validPath} ${testPath} ${trainBbox} ${validBbox} ${testBbox} ${model} ${epoch} ${batch_size} ${numSamples} ${pretrained}>> ../train_log/${trainPath}_${validPath}_${testPath}_${model}_epoch${epoch}_batchsize${batch_size}_${pretrained}.txt
#python ../codes/train.py ${trainPath} ${validPath} ${trainBbox} ${validBbox} ${model} ${epoch} ${batch_size} ${numSamples} ${pretrained}>> ../train_log/${trainPath}_${validPath}_${model}_epoch${epoch}_batchsize${batch_size}_${pretrained}.txt
echo 'training_finished'