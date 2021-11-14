#!/bin/bash

#PBS -q h-regular
#PBS -l select=1
#PBS -W group_list=gk36
#PBS -l walltime=10:00:00
#PBS -o extract_FN_samples.txt
#PBS -j oe
#PBS -m abe
#PBS -M suzuki-takahiro596@g.ecc.u-tokyo.ac.jp

cd "${PBS_O_WORKDIR}" || exit

. /lustre/gk36/k77012/anaconda3/bin/activate pytorch2

ver=2

python ../codes/extract_FN_samples.py ${ver}
echo 'finished'