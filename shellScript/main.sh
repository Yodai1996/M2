#!/bin/bash

#PBS -q h-regular
#PBS -l select=16
#PBS -W group_list=gk36
#PBS -l walltime=10:00:00
#PBS -o main.txt
#PBS -j oe
#PBS -m abe
#PBS -M suzuki-takahiro596@g.ecc.u-tokyo.ac.jp


trainPath='AbnormalDir1000'
validPath='AbnormalDir5012'
trainBbox='abnormal1000_bboxinfo.csv'
validBbox='abnormal5012_bboxinfo.csv'
#testPath='AbnormalDir'
#testBbox='abnormal_bboxinfo.csv'

epoch=2
batch_size=32
numSamples=50

model='SSD'
#model='fasterRCNN'
pretrained='pretrained'  #'unpretrained'
mkdir -p "/lustre/gk36/k77012/M2/result/${trainPath}_${validPath}_${model}_batch${batch_size}_epoch${epoch}_${pretrained}/" #for saving Dir
mkdir -p "/lustre/gk36/k77012/M2/result/${trainPath}_${validPath}_${model}_batch${batch_size}_epoch${epoch}_${pretrained}/train"
mkdir -p "/lustre/gk36/k77012/M2/result/${trainPath}_${validPath}_${model}_batch${batch_size}_epoch${epoch}_${pretrained}/valid"
#mkdir -p "/lustre/gk36/k77012/M2/result/${trainPath}_${validPath}_${model}_batch${batch_size}_epoch${epoch}_${pretrained}/test"


NODES=($(cat "${PBS_NODEFILE}" | uniq))
for((i=0;i<${#NODES[@]};i++))
do
    NODE=${NODES[$i]}
    ssh "${NODE}" -T "
    export OMP_NUM_THREADS=4
    . /lustre/gk36/k77012/anaconda3/bin/activate pytorch2
    cd ${PBS_O_WORKDIR} || exit
    python -m torch.distributed.launch \
          --nnodes=16 --nproc_per_node=2 \
          --master_addr=${HOSTNAME} --master_port=9999 \
          --node_rank=${i} \
          ../codes/train.py --tp ${trainPath} --vp ${validPath} --tb ${trainBbox} --vb ${validBbox} --model ${model} --epoch ${epoch} --bsz ${batch_size} --ns ${numSamples} --pret ${pretrained} >> ../train_log/log_${trainPath}_${validPath}_${model}_epoch${epoch}_batchsize${batch_size}_${pretrained}.txt
    " &
done
wait

echo 'training_finished'