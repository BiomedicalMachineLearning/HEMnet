#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 20
#SBATCH --mem=80000
#SBATCH -o out_%x_%j.txt
#SBATCH -e error_%x_%j.txt
#SBATCH --gres=gpu:tesla:2
#SBATCH --job-name HEMnet


MY_DIR=/clusterdata/s4463993
cd $MY_DIR

module load cuda/10.0.130
module load gnu7
module load openmpi3
module load anaconda/3.6

source activate /scratch/imb/Xiao/.conda/envs/tensorflow_2.0.0

BASE_DIR=/scratch/imb/Xiao/HE_test/10x/

OUT_DIR=HEMnet_16_01_2020

python /scratch/imb/Xiao/HEMnet/HEMnet/train.py -b $BASE_DIR \
-t train_dataset_10x_19_12_19_strict_Reinhard/tiles_10x/ -l valid_Reinhard/tiles_10x \
-o $OUT_DIR -g 2 -e 10 -s -m vgg16 -a 64 -w -v

python /scratch/imb/Xiao/HEMnet/HEMnet/test.py  -b $BASE_DIR \
-t 1957_T_Reinhard/tiles_10x -o $OUT_DIR -w ${OUT_DIR}/training_results/trained_model.h5 \
-m vgg16 -g 2 -v

python /scratch/imb/Xiao/HEMnet/HEMnet/visualisation.py -b $BASE_DIR \
-t ${OUT_DIR}/training_results -p ${OUT_DIR}/prediction_results -o ${OUT_DIR} -i 1957_T
