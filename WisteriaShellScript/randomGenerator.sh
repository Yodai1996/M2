#!/bin/bash

#PJM -g jh170036a
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=30:00:00
#PJM --fs /work,/data
#PJM -N randomGenerator
#PJM -o randomGenerator.txt
#PJM -j
#PJM --mail-list suzuki-takahiro596@g.ecc.u-tokyo.ac.jp
#PJM -m b
#PJM -m e

#module load cuda/10.2
module load cuda/11.1

cd "${PJM_O_WORKDIR}" || exit

#pipenv shell #activate the virtual environment

m=1000 #200
normalDir="/work/gk36/k77012/M2/data/NormalDir/" #データプールなら何でもよい。
normalIdList="/work/gk36/k77012/M2/normalFiles${m}.csv"


abnormalDir="/work/gk36/k77012/M2/data/udr_${m}/"
segMaskDir="/work/gk36/k77012/M2/SegmentationMask/udr_${m}/"
saveParaPath="/work/gk36/k77012/M2/simDataInfo/paraInfo/parameterInfo_udr_${m}.csv"
saveBboxPath="/work/gk36/k77012/M2/simDataInfo/bboxInfo/bboxInfo_udr_${m}.csv"

mkdir -p ${abnormalDir} #train_${i}が作られていること前提ではない。親ディレクトリが無い場合は全て作る、＆、フォルダがすでに存在する場合は何もしない
mkdir -p ${segMaskDir}
mkdir -p "/work/gk36/k77012/M2/simDataInfo/paraInfo/"
mkdir -p "/work/gk36/k77012/M2/simDataInfo/bboxInfo/"

pipenv run python ../WisteriaCodes/randomGenerator.py ${normalIdList} ${normalDir} ${abnormalDir} ${segMaskDir} ${saveParaPath} ${saveBboxPath}
echo 'finished'