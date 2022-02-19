#!/bin/bash

#PJM -g jh170036a
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=30:00:00
#PJM --fs /work,/data
#PJM -N resulting
#PJM -o result_curriculum.txt
#PJM -j
#PJM --mail-list suzuki-takahiro596@g.ecc.u-tokyo.ac.jp
#PJM -m b
#PJM -m e

#module load cuda/10.2
module load cuda/11.1

cd "${PJM_O_WORKDIR}" || exit

#pipenv shell #activate the virtual environment

#modify here
fold=5
score="0.6"
iter=6
pretrain="ImageNet" #"BigBbox" #"ImageNet"
epoch=120

optimizer="VSGD" #"SAM" #"SGD" #"Adam" #"VSGD"
variability="0.01" #"0.05" #when you use SAM, use an appropriate value of rho, e.g., 0.05. Otherwise, use 0.01.

#test data
dataPath='AllDataDir'
dataBboxName="rare_small_bboxInfo_81_${fold}_withNormal"

#used model. change here
#"ImageNet_${dataBboxName}"
#"pretrain/model_nonSmall_bboxInfo_655_nonSmall_bboxInfo_164_withNormal_VSGD_0.01_120"
#"curriculumBO/rare_small_bboxInfo_20_1_withNormal/start0.3_decay1.0_VSGD_FAUC_PretrainedBigBbox_variability0.01_decay1.0_t1000_v200_iter30_inf30_epoch40/model7"
#"minmin_FAUC_PretrainedBigBbox/model_version20_rare_small_bboxInfo_20_1_withNormal_pretrainedBigBbox_epoch40"
#"minmin_FAUC_PretrainedImageNet/model_version27_rare_small_bboxInfo_20_1_withNormal_pretrainedImageNet_epoch40"
#"minmin_FAUC_PretrainedImageNet_VSGD_epoch120/model_version30_rare_small_bboxInfo_20_1_withNormal_pretrainedImageNet_epoch120"
#"curriculumBO/rare_small_bboxInfo_20_1_withNormal/start0.5_decay1.0_VSGD_FAUC_PretrainedImageNet_variability0.01_decay1.0_t1000_v200_iter30_inf30_epoch40/model7"

Path="curriculumBO/rare_small_bboxInfo_20_${fold}_withNormal/start${score}_decay1.0_${optimizer}_FAUC_Pretrained${pretrain}_variability${variability}_decay1.0_t1000_v200_iter30_inf30_epoch${epoch}/model${iter}"
#Path="curriculumBO/rare_small_bboxInfo_20_${fold}_withNormal/start${score}_decay1.0_${optimizer}_FAUC_Pretrained${pretrain}_variability0.01_decay1.0_t1000_v200_iter30_inf30_epoch${epoch}/model${iter}"
#Path="curriculumBO/rare_small_bboxInfo_20_${fold}_withNormal/start${score}_decay1.0_VSGD_FAUC_Pretrained${pretrain}_variability0.01_decay1.0_t1000_v200_iter30_inf30_epoch${epoch}/model${iter}"
#Path="pretrain/model_nonSmall_bboxInfo_655_nonSmall_bboxInfo_164_withNormal_VSGD_0.01_120"

loadModelPath="/work/jh170036a/k77012/M2/model/${Path}"
#loadModelPath="/work/gk36/k77012/M2/model/${Path}"


saveFROCPath="/work/jh170036a/k77012/M2/FROC/${Path}/${dataBboxName}/" #saveFROCPath="/work/gk36/k77012/M2/FROC/${Path}/${dataBboxName}/"
mkdir -p ${saveFROCPath}

#numSamples=50
saveDir="/work/jh170036a/k77012/M2/result/${Path}/${dataBboxName}/" #saveDir="/work/gk36/k77012/M2/result/${Path}/${dataBboxName}/"
mkdir -p ${saveDir} #for saving Dir
#mkdir -p "${saveDir}"

LogDir="/work/jh170036a/k77012/M2/log/${Path}/"
mkdir -p ${LogDir}

pipenv run python ../WisteriaCodes/result.py ${dataPath} ${dataBboxName} ${Path} ${loadModelPath} ${saveDir} ${saveFROCPath} >> "${LogDir}/result_${dataBboxName}.txt"
echo 'resulting_finished'