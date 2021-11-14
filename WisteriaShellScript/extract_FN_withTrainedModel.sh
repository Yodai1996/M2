#!/bin/bash

#PJM -g jh170036a
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=10:00:00
#PJM --fs /work,/data
#PJM -N extract_FN_withTM
#PJM -o extract_FN_withTM.txt
#PJM -j
#PJM --mail-list suzuki-takahiro596@g.ecc.u-tokyo.ac.jp
#PJM -m b
#PJM -m e

#module load cuda/10.2
module load cuda/11.1

cd "${PJM_O_WORKDIR}" || exit

#change here
ver=5 #ver \in [1,5]
dataPath='AbnormalDir' #'AbnormalDir5880' #data poolなので広めであればよい
dataBbox="abnormal1176_bboxinfo_${ver}.csv"

#thres used to decide FN indices
thres="0.15"
lower_thres="0"

saveFnDir="/work/gk36/k77012/M2/FN_samples/FN_samples_${thres}_lowerThres${lower_thres}/" #common to each ver.
mkdir -p ${saveFnDir}
mkdir -p "${saveFnDir}/Images_with_gt_and_inference/"
mkdir -p "${saveFnDir}/Images_with_gt/"
mkdir -p "${saveFnDir}/Original_images/"

saveDir="/work/gk36/k77012/M2/result/FN_samples/abnormal1176_${ver}/" #_thres${thres}/"
mkdir -p ${saveDir}
mkdir -p "${saveDir}/Images_with_gt_and_inference"
mkdir -p "${saveDir}/Images_with_gt"

pipenv run python ../WisteriaCodes/extract_FN_withTrainedModel.py ${ver} ${dataPath} ${dataBbox} ${thres} ${lower_thres} ${saveDir} ${saveFnDir}
echo 'extract_FN_finished'