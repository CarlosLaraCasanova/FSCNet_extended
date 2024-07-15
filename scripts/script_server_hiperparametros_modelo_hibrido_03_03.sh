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





python entrenamiento_hibrido.py --tosave ./resultados --cv split_train_test.json --gt ./FORENSE/data_curated_bbox.csv --boxes 3d_boxes.npy --ver ver_lst.npy --opt_mat m_sol_dict_train_final.json --workers 2 --torch_workers 16 --w_out 32 --e_quantile 0.9 --model modelo_hibrido1  --dataset midataset_roi_hibrido --pretrained  --epochs 100 --batch 16 --w_factor=5 --lr 0.0001 --alfa 7  --beta 3 --label modelo_hibrido1_7_3

python entrenamiento_hibrido.py --tosave ./resultados --cv split_train_test.json --gt ./FORENSE/data_curated_bbox.csv --boxes 3d_boxes.npy --ver ver_lst.npy --opt_mat m_sol_dict_train_final.json --workers 2 --torch_workers 16 --w_out 32 --e_quantile 0.9 --model modelo_hibrido1  --dataset midataset_roi_hibrido --pretrained  --epochs 100 --batch 16 --w_factor=5 --lr 0.0001 --alfa 0.7 --beta 0.3 --label modelo_hibrido1_07_03


printf "$(date '+%H:%M:%S)')\n"
printf "\n\nFINALIZADO\n\n"
