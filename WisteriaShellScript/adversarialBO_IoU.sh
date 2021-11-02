#!/bin/bash

#PBS -q l-regular
#PBS -l select=1
#PBS -W group_list=gk36
#PBS -l walltime=59:30:00
#PBS -o adversarialBO_IoU.txt
#PBS -j oe
#PBS -m abe
#PBS -M suzuki-takahiro596@g.ecc.u-tokyo.ac.jp

cd "${PBS_O_WORKDIR}" || exit

. /lustre/gk36/k77012/anaconda3/bin/activate pytorch2


numIter=30 #4 #30 #4 #30 #4 #iteration for BO
numInfer=30 #how many images are generated by the simulator for inference
optimizer='Adam' #default
metric='IoU' #'Dice'

bufText="/lustre/gk36/k77012/M2/buf_adversarialBO_${optimizer}_${metric}.txt" #use the same text as it is just a buffer
bestText="/lustre/gk36/k77012/M2/best_adversarialBO_${optimizer}_${metric}.txt"

numTrain=1000 #30 #1000 #30
normalDir="/lustre/gk36/k77012/M2/data/NormalDir${numTrain}/"
normalIdList="/lustre/gk36/k77012/M2/normalIdList${numTrain}.csv"

numValid=200
normalDir2="/lustre/gk36/k77012/M2/data/NormalDir${numValid}/"
normalIdList2="/lustre/gk36/k77012/M2/normalIdList${numValid}.csv"

testPath='AbnormalDir4880' #'AbnormalDir5012'
testBbox='abnormal4880_bboxinfo.csv' #'abnormal5012_bboxinfo.csv'

#for training
epoch=40
batch_size=64
numSamples=50
model='SSD'

#make dir for model and log
modelPath="/lustre/gk36/k77012/M2/model/adversarialBO/${optimizer}_${metric}/"
mkdir -p ${modelPath}
mkdir -p "/lustre/gk36/k77012/M2/train_log/adversarialBO/${optimizer}_${metric}"


#first, change directory
cd ../bo_io

for i in 1 2
do
  normalDirForInfer="/lustre/gk36/k77012/M2/data/NormalDir${numInfer}/"
  normalIdListForInfer="/lustre/gk36/k77012/M2/normalIdList${numInfer}.csv"

  boDir="/lustre/gk36/k77012/M2/bo_io/in/adversarialBO/${optimizer}_${metric}"
  mkdir -p ${boDir}
  boText="${boDir}/iter${i}.txt"

  #inference for the first 5 intial points
  for j in 1 2 3 4 5
  do
    abnormalDir="/lustre/gk36/k77012/M2/data/sim${j}_${numInfer}/"
    bboxPath="/lustre/gk36/k77012/M2/simDataInfo/bboxInfo/bboxInfo${j}_${numInfer}.csv"
    python ../codes/initialInference.py $i $j ${abnormalDir} ${bboxPath} ${boText} ${modelPath} ${metric} #i is used to identify the model to load, j is used to determine the X values. 結果をboTextに格納.
  done

  #inference for the next numIter points using BO module
  for j in $(seq 1 ${numIter}) #[1,numIter]
  do
    ./build/suggest --hm --ha --hpopt -a ei --md 7 --mi ${boText} >> ${bufText}  #run the BO module

    #save information
    abnormalDir="/lustre/gk36/k77012/M2/data/adversarialBO/${optimizer}_${metric}/ver${i}/iter${j}_${numInfer}/"
    segMaskDir="/lustre/gk36/k77012/M2/SegmentationMask/adversarialBO/${optimizer}_${metric}/ver${i}/mask${j}_${numInfer}/" #まあ無くてもよさそう
    paraDir="/lustre/gk36/k77012/M2/simDataInfo/paraInfo/adversarialBO/${optimizer}_${metric}/ver${i}/"
    bboxDir="/lustre/gk36/k77012/M2/simDataInfo/bboxInfo/adversarialBO/${optimizer}_${metric}/ver${i}/"
    paraPath="${paraDir}/parameterInfo${j}_${numInfer}.csv" #まあ無くてもよさそう
    bboxPath="${bboxDir}/bboxInfo${j}_${numInfer}.csv"

    mkdir -p ${abnormalDir} #train_${i}が作られていること前提ではない。親ディレクトリが無い場合は全て作る、＆、フォルダがすでに存在する場合は何もしない
    mkdir -p ${segMaskDir}
    mkdir -p ${paraDir}
    mkdir -p ${bboxDir}

    python ../codes/fractalGenerator.py ${normalIdListForInfer} ${normalDirForInfer} ${abnormalDir} ${segMaskDir} ${paraPath} ${bboxPath} ${bufText} #bufText is used to know the X values.

    python ../codes/inference.py $i ${bufText} ${abnormalDir} ${bboxPath} ${boText} ${modelPath} ${metric} #i is used to identify the model to load, bufText is used to determine the X values. 結果をboTextに格納.

  done

  #bufferに出力するというよりかは、ここでboTextを基に、best値を判定し、buf同様の形式で出力すればよい。
  python ../codes/findBest.py ${boText} >> ${bestText}

  ### generate training data
  #save information
  trainPath="ver${i}_${numTrain}"
  abnormalDir="/lustre/gk36/k77012/M2/data/adversarialBO/${optimizer}_${metric}/${trainPath}/"
  segMaskDir="/lustre/gk36/k77012/M2/SegmentationMask/adversarialBO/${optimizer}_${metric}/ver${i}_${numTrain}/"
  paraPath="/lustre/gk36/k77012/M2/simDataInfo/paraInfo/adversarialBO/${optimizer}_${metric}/parameterInfo${i}_${numTrain}.csv"
  bboxPath="/lustre/gk36/k77012/M2/simDataInfo/bboxInfo/adversarialBO/${optimizer}_${metric}/bboxInfo${i}_${numTrain}.csv"

  mkdir -p ${abnormalDir} #train_${i}が作られていること前提ではない。親ディレクトリが無い場合は全て作る、＆、フォルダがすでに存在する場合は何もしない
  mkdir -p ${segMaskDir}

  python ../codes/fractalGenerator.py ${normalIdList} ${normalDir} ${abnormalDir} ${segMaskDir} ${paraPath} ${bboxPath} ${bestText} #bestText is used to know the X values.

  ### generate tvalidation data
  #save information
  validPath="ver${i}_${numValid}"
  abnormalDir2="/lustre/gk36/k77012/M2/data/adversarialBO/${optimizer}_${metric}/${validPath}/"
  segMaskDir2="/lustre/gk36/k77012/M2/SegmentationMask/adversarialBO/${optimizer}_${metric}/ver${i}_${numValid}/"
  paraPath2="/lustre/gk36/k77012/M2/simDataInfo/paraInfo/adversarialBO/${optimizer}_${metric}/parameterInfo${i}_${numValid}.csv"
  bboxPath2="/lustre/gk36/k77012/M2/simDataInfo/bboxInfo/adversarialBO/${optimizer}_${metric}/bboxInfo${i}_${numValid}.csv"

  mkdir -p ${abnormalDir2} #train_${i}が作られていること前提ではない。親ディレクトリが無い場合は全て作る、＆、フォルダがすでに存在する場合は何もしない
  mkdir -p ${segMaskDir2}

  python ../codes/fractalGenerator.py ${normalIdList2} ${normalDir2} ${abnormalDir2} ${segMaskDir2} ${paraPath2} ${bboxPath2} ${bestText} #bestText is used to know the X values.

  savePath="adversarialBO/${optimizer}_${metric}/${trainPath}/${validPath}_test${testPath}/${model}_batch${batch_size}_epoch${epoch}"
  mkdir -p "/lustre/gk36/k77012/M2/result/${savePath}/" #for saving Dir
  mkdir -p "/lustre/gk36/k77012/M2/result/${savePath}/train"
  mkdir -p "/lustre/gk36/k77012/M2/result/${savePath}/valid"
  mkdir -p "/lustre/gk36/k77012/M2/result/${savePath}/test"

  #training the model
  python ../codes/train_continual.py ${abnormalDir} ${abnormalDir2} ${testPath} ${bboxPath} ${bboxPath2} ${testBbox} ${modelPath} ${model} ${epoch} ${batch_size} ${numSamples} ${i} ${savePath} ${optimizer} ${metric} >> ../train_log/adversarialBO/${optimizer}_${metric}/model${i}_${trainPath}_${validPath}_${testPath}_${model}_epoch${epoch}_batchsize${batch_size}.txt
  echo $i'_finished'

done