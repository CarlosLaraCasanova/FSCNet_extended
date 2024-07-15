#!/bin/bash

# Job name
#SBATCH --job-name TFGCARLOS

# Assign job to a queue
#SBATCH --partition dios
#SACTH --nodelist Titan

# Use GPU
#SBATCH --gres=gpu:1

# Default configs for NGPU
export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/clara/TFG_Carlos_ENV
export TFHUB_CACHE_DIR=.


printf "$(date '+%H:%M:%S')\n"

python entrenamiento.py --tosave ./resultados --cv split_train_cvfolds.json --gt ./FORENSE/data_curated_bbox.csv --boxes 3d_boxes.npy --ver ver_lst.npy --opt_mat m_sol_dict_train_cv.json --workers 2 --torch_workers 16 --w_out 224 --e_quantile 0.9 --model modelo2  --dataset midataset_facebox --pretrained  --epochs 100 --batch 16 --lr 0.0004 --label modelo2_100_16_00004

python entrenamiento.py --tosave ./resultados --cv split_train_cvfolds.json --gt ./FORENSE/data_curated_bbox.csv --boxes 3d_boxes.npy --ver ver_lst.npy --opt_mat m_sol_dict_train_cv.json --workers 2 --torch_workers 16 --w_out 224 --e_quantile 0.9 --model modelo2  --dataset midataset_facebox --pretrained  --epochs 100 --batch 8 --lr 0.0004 --label modelo2_100_8_00004



python entrenamiento.py --tosave ./resultados --cv split_train_cvfolds.json --gt ./FORENSE/data_curated_bbox.csv --boxes 3d_boxes.npy --ver ver_lst.npy --opt_mat m_sol_dict_train_cv.json --workers 2 --torch_workers 16 --w_out 224 --e_quantile 0.9 --model modelo2  --dataset midataset_facebox --pretrained  --epochs 100 --batch 16 --lr 0.0001 --label modelo2_100_16_00001

python entrenamiento.py --tosave ./resultados --cv split_train_cvfolds.json --gt ./FORENSE/data_curated_bbox.csv --boxes 3d_boxes.npy --ver ver_lst.npy --opt_mat m_sol_dict_train_cv.json --workers 2 --torch_workers 16 --w_out 224 --e_quantile 0.9 --model modelo2  --dataset midataset_facebox --pretrained  --epochs 100 --batch 8 --lr 0.0001 --label modelo2_100_8_00001







printf "$(date '+%H:%M:%S)')\n"
printf "\n\nFINALIZADO\n\n"
