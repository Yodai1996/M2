#!/bin/bash

#PJM -g jh170036a
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=30:00:00
#PJM --fs /work,/data
#PJM -N froc
#PJM -o froc.txt
#PJM -j
#PJM --mail-list suzuki-takahiro596@g.ecc.u-tokyo.ac.jp
#PJM -m b
#PJM -m e

#module load cuda/10.2
module load cuda/11.1

cd "${PJM_O_WORKDIR}" || exit

#pipenv shell #activate the virtual environment

#pretrain="ImageNet"

#endpoint to draw FROC
end=1

prefixgk36="/work/gk36/k77012/M2/model"
prefix="/work/jh170036a/k77012/M2/model"


#ModelPath
nonrare="${prefixgk36}/pretrain/model_nonSmall_bboxInfo_655_nonSmall_bboxInfo_164_withNormal_VSGD_0.01_120"

#minmin Path
#change here
Path1="minmin_FAUC_PretrainedImageNet_VSGD_epoch120/model_version30_rare_small_bboxInfo_20_1_withNormal_pretrainedImageNet_epoch120"
Path2="minmin_FAUC_PretrainedImageNet_VSGD_epoch120/model_version19_rare_small_bboxInfo_20_2_withNormal_pretrainedImageNet_epoch120"
Path3="minmin_FAUC_PretrainedImageNet_VSGD_epoch120/model_version21_rare_small_bboxInfo_20_3_withNormal_pretrainedImageNet_epoch120"
Path4="minmin_FAUC_PretrainedImageNet_VSGD_epoch120/model_version23_rare_small_bboxInfo_20_4_withNormal_pretrainedImageNet_epoch120"
Path5="minmin_FAUC_PretrainedImageNet_VSGD_epoch120/model_version32_rare_small_bboxInfo_20_5_withNormal_pretrainedImageNet_epoch120"

BayRn1="${prefixgk36}/${Path1}"
BayRn2="${prefix}/${Path2}"
BayRn3="${prefix}/${Path3}"
BayRn4="${prefix}/${Path4}"
BayRn5="${prefix}/${Path5}"

#CDR Path
#change here
Path1="curriculumBO/rare_small_bboxInfo_20_1_withNormal/start0.5_decay1.0_VSGD_FAUC_PretrainedImageNet_variability0.01_decay1.0_t1000_v200_iter30_inf30_epoch40/model7"
Path2="curriculumBO/rare_small_bboxInfo_20_2_withNormal/start0.6_decay1.0_VSGD_FAUC_PretrainedImageNet_variability0.01_decay1.0_t1000_v200_iter30_inf30_epoch40/model9"
Path3="curriculumBO/rare_small_bboxInfo_20_3_withNormal/start0.6_decay1.0_VSGD_FAUC_PretrainedImageNet_variability0.01_decay1.0_t1000_v200_iter30_inf30_epoch40/model10"
Path4="curriculumBO/rare_small_bboxInfo_20_4_withNormal/start0.5_decay1.0_VSGD_FAUC_PretrainedImageNet_variability0.01_decay1.0_t1000_v200_iter30_inf30_epoch40/model6"
Path5="curriculumBO/rare_small_bboxInfo_20_5_withNormal/start0.5_decay1.0_VSGD_FAUC_PretrainedImageNet_variability0.01_decay1.0_t1000_v200_iter30_inf30_epoch40/model5"

CDR1="${prefixgk36}/${Path1}"
CDR2="${prefix}/${Path2}"
CDR3="${prefix}/${Path3}"
CDR4="${prefix}/${Path4}"
CDR5="${prefix}/${Path5}"

saveFROCPath="/work/jh170036a/k77012/M2/FROC/All_ImageNetPretrain/" #saveFROCPath="/work/gk36/k77012/M2/FROC/${Path}/${dataBboxName}/"
mkdir -p ${saveFROCPath}

pipenv run python ../WisteriaCodes/froc.py ${saveFROCPath} ${end} ${nonrare} ${BayRn1} ${BayRn2} ${BayRn3} ${BayRn4} ${BayRn5} ${CDR1} ${CDR2} ${CDR3} ${CDR4} ${CDR5}
echo 'froc_finished'