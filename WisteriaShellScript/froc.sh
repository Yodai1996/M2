#!/bin/bash

#PJM -g jh170036a
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=03:00:00
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
#prefix="/work/jh170036a/k77012/M2/model"
prefix="/work/gu14/k77012/M2/model"

#ModelPath
#nonrare="${prefixgk36}/pretrain/model_nonSmall_bboxInfo_655_nonSmall_bboxInfo_164_withNormal_VSGD_0.01_120"

#minmin Path
#change here
#Path1="minmin_FAUC_PretrainedImageNet_Adam_epoch120/model_version40_rare_small_bboxInfo_20_1_withNormal_pretrainedImageNet_epoch120"
#Path2="minmin_FAUC_PretrainedImageNet_Adam_epoch120/model_version14_rare_small_bboxInfo_20_2_withNormal_pretrainedImageNet_epoch120"
#Path3="minmin_FAUC_PretrainedImageNet_Adam_epoch120/model_version40_rare_small_bboxInfo_20_3_withNormal_pretrainedImageNet_epoch120"
#Path4="minmin_FAUC_PretrainedImageNet_Adam_epoch120/model_version17_rare_small_bboxInfo_20_4_withNormal_pretrainedImageNet_epoch120"
#Path5="model_version3_sim3_1000_rare_small_bboxInfo_20_5_withNormal_ImageNet_Adam_epoch120"
Path1="minmin_FAUC_PretrainedImageNet_Adam_epoch120_ver1/model_version40_rare_small_bboxInfo_20_1_withNormal_pretrainedImageNet_epoch120"
Path2="minmin_FAUC_PretrainedImageNet_Adam_epoch120_ver1/model_version14_rare_small_bboxInfo_20_2_withNormal_pretrainedImageNet_epoch120"
Path3="minmin_FAUC_PretrainedImageNet_Adam_epoch120_ver1/model_version40_rare_small_bboxInfo_20_3_withNormal_pretrainedImageNet_epoch120"
Path4="minmin_FAUC_PretrainedImageNet_Adam_epoch120/model_version17_rare_small_bboxInfo_20_4_withNormal_pretrainedImageNet_epoch120"
Path5="model_version3_sim3_1000_rare_small_bboxInfo_20_5_withNormal_ImageNet_Adam_epoch120"

#BayRn1="${prefixgk36}/${Path1}"
BayRn1="${prefix}/${Path1}"
BayRn2="${prefix}/${Path2}"
BayRn3="${prefix}/${Path3}"
BayRn4="${prefix}/${Path4}"
BayRn5="${prefixgk36}/${Path5}"

#UDR
Path1="udr/model_udr_1000_rare_small_bboxInfo_20_1_withNormal_ImageNet_Adam_epoch120"
Path2="udr/model_udr_1000_rare_small_bboxInfo_20_2_withNormal_ImageNet_Adam_epoch120"
Path3="udr/model_udr_1000_rare_small_bboxInfo_20_3_withNormal_ImageNet_Adam_epoch120"
Path4="udr/model_udr_1000_rare_small_bboxInfo_20_4_withNormal_ImageNet_Adam_epoch120"
Path5="udr/model_udr_1000_rare_small_bboxInfo_20_5_withNormal_ImageNet_Adam_epoch120"

UDR1="${prefix}/${Path1}"
UDR2="${prefix}/${Path2}"
UDR3="${prefix}/${Path3}"
UDR4="${prefix}/${Path4}"
UDR5="${prefix}/${Path5}"

#GDR Path
Path1="curriculumBO/rare_small_bboxInfo_20_1_withNormal/start0.6_decay1.0_VSGD_FAUC_PretrainedImageNet_variability0.01_decay1.0_t1000_v200_iter30_inf30_epoch120/model5"
Path2="curriculumBO/rare_small_bboxInfo_20_2_withNormal/start0.6_decay1.0_VSGD_FAUC_PretrainedImageNet_variability0.01_decay1.0_t1000_v200_iter30_inf30_epoch120/model2"
Path3="curriculumBO/rare_small_bboxInfo_20_3_withNormal/start0.6_decay1.0_VSGD_FAUC_PretrainedImageNet_variability0.01_decay1.0_t1000_v200_iter30_inf30_epoch120/model5"
Path4="curriculumBO/rare_small_bboxInfo_20_4_withNormal/start0.7_decay1.0_VSGD_FAUC_PretrainedImageNet_variability0.01_decay1.0_t1000_v200_iter30_inf30_epoch120/model4"
Path5="curriculumBO/rare_small_bboxInfo_20_5_withNormal/start0.4_decay1.0_VSGD_FAUC_PretrainedImageNet_variability0.01_decay1.0_t1000_v200_iter30_inf30_epoch120/model9"

CDR1="${prefix}/${Path1}"
CDR2="${prefix}/${Path2}"
CDR3="${prefix}/${Path3}"
CDR4="${prefix}/${Path4}"
CDR5="${prefix}/${Path5}"

#easy2hard-1
Path1="curriculumBO/rare_small_bboxInfo_20_1_withNormal/start1.0_decay0.1_VSGD_FAUC_PretrainedImageNet_variability0.01_decay1.0_t1000_v200_iter30_inf30_epoch120/model8"
Path2="curriculumBO/rare_small_bboxInfo_20_2_withNormal/start1.0_decay0.1_VSGD_FAUC_PretrainedImageNet_variability0.01_decay1.0_t1000_v200_iter30_inf30_epoch120/model4"
Path3="curriculumBO/rare_small_bboxInfo_20_3_withNormal/start1.0_decay0.1_VSGD_FAUC_PretrainedImageNet_variability0.01_decay1.0_t1000_v200_iter30_inf30_epoch120/model5"
Path4="curriculumBO/rare_small_bboxInfo_20_4_withNormal/start1.0_decay0.1_VSGD_FAUC_PretrainedImageNet_variability0.01_decay1.0_t1000_v200_iter30_inf30_epoch120/model5"
Path5="curriculumBO/rare_small_bboxInfo_20_5_withNormal/start1.0_decay0.1_VSGD_FAUC_PretrainedImageNet_variability0.01_decay1.0_t1000_v200_iter30_inf30_epoch120/model6"

easy2hard1_1="${prefix}/${Path1}"
easy2hard1_2="${prefix}/${Path2}"
easy2hard1_3="${prefix}/${Path3}"
easy2hard1_4="${prefix}/${Path4}"
easy2hard1_5="${prefix}/${Path5}"

#easy2hard-2
Path1="curriculumBO/rare_small_bboxInfo_20_1_withNormal/start0.75_decay0.05_VSGD_FAUC_PretrainedImageNet_variability0.01_decay1.0_t1000_v200_iter30_inf30_epoch120/model9"
Path2="curriculumBO/rare_small_bboxInfo_20_2_withNormal/start0.75_decay0.05_VSGD_FAUC_PretrainedImageNet_variability0.01_decay1.0_t1000_v200_iter30_inf30_epoch120/model5"
Path3="curriculumBO/rare_small_bboxInfo_20_3_withNormal/start0.75_decay0.05_VSGD_FAUC_PretrainedImageNet_variability0.01_decay1.0_t1000_v200_iter30_inf30_epoch120/model5"
Path4="curriculumBO/rare_small_bboxInfo_20_4_withNormal/start0.75_decay0.05_VSGD_FAUC_PretrainedImageNet_variability0.01_decay1.0_t1000_v200_iter30_inf30_epoch120/model4"
Path5="curriculumBO/rare_small_bboxInfo_20_5_withNormal/start0.75_decay0.05_VSGD_FAUC_PretrainedImageNet_variability0.01_decay1.0_t1000_v200_iter30_inf30_epoch120/model7"

easy2hard2_1="${prefix}/${Path1}"
easy2hard2_2="${prefix}/${Path2}"
easy2hard2_3="${prefix}/${Path3}"
easy2hard2_4="${prefix}/${Path4}"
easy2hard2_5="${prefix}/${Path5}"


saveFROCPath="/work/gu14/k77012/M2/FROC/" #"/work/gu14/k77012/M2/FROC/All_ImageNetPretrain/" #saveFROCPath="/work/gk36/k77012/M2/FROC/${Path}/${dataBboxName}/"
mkdir -p ${saveFROCPath}

####change below####
fold=5
UDR=${UDR5}
BayRn=${BayRn5}
CDR=${CDR5}
easy2hard1=${easy2hard1_5}
easy2hard2=${easy2hard2_5}
####change above####

#pipenv run python ../WisteriaCodes/froc.py ${saveFROCPath} ${end} ${UDR1} ${UDR2} ${UDR3} ${UDR4} ${UDR5} ${BayRn1} ${BayRn2} ${BayRn3} ${BayRn4} ${BayRn5} ${CDR1} ${CDR2} ${CDR3} ${CDR4} ${CDR5} ${easy2hard1_1} ${easy2hard1_2} ${easy2hard1_3} ${easy2hard1_4} ${easy2hard1_5} ${easy2hard2_1} ${easy2hard2_2} ${easy2hard2_3} ${easy2hard2_4} ${easy2hard2_5}
pipenv run python ../WisteriaCodes/froc.py ${saveFROCPath} ${end} ${UDR} ${BayRn} ${CDR} ${easy2hard1} ${easy2hard2} ${fold}
echo 'froc_finished'