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

python entrenamiento.py --tosave ./resultados --cv split_train_cvfolds.json --gt ./FORENSE/data_curated_bbox.csv --boxes 3d_boxes.npy --ver ver_lst.npy --opt_mat m_sol_dict_train_cv.json --workers 2 --torch_workers 16 --w_out 32 --e_quantile 0.9 --model modelo1  --dataset midataset_roi --pretrained  --epochs 200 --batch 8 --w_factor=5 --lr 0.0004 --max_translation 0.02 --max_angle 5 --random_flip --label modelofreeze_002_5_flip

python entrenamiento.py --tosave ./resultados --cv split_train_cvfolds.json --gt ./FORENSE/data_curated_bbox.csv --boxes 3d_boxes.npy --ver ver_lst.npy --opt_mat m_sol_dict_train_cv.json --workers 2 --torch_workers 16 --w_out 32 --e_quantile 0.9 --model modelo1  --dataset midataset_roi --pretrained  --epochs 200 --batch 8 --w_factor=5 --lr 0.0004 --max_translation 0.02 --max_angle 15 --random_flip --label modelofreeze_002_15_flip

python entrenamiento.py --tosave ./resultados --cv split_train_cvfolds.json --gt ./FORENSE/data_curated_bbox.csv --boxes 3d_boxes.npy --ver ver_lst.npy --opt_mat m_sol_dict_train_cv.json --workers 2 --torch_workers 16 --w_out 32 --e_quantile 0.9 --model modelo1  --dataset midataset_roi --pretrained  --epochs 200 --batch 8 --w_factor=5 --lr 0.0004 --max_translation 0.02 --max_angle 25 --random_flip --label modelofreeze_002_25_flip

python entrenamiento.py --tosave ./resultados --cv split_train_cvfolds.json --gt ./FORENSE/data_curated_bbox.csv --boxes 3d_boxes.npy --ver ver_lst.npy --opt_mat m_sol_dict_train_cv.json --workers 2 --torch_workers 16 --w_out 32 --e_quantile 0.9 --model modelo1  --dataset midataset_roi --pretrained  --epochs 200 --batch 8 --w_factor=5 --lr 0.0004 --max_translation 0.02 --max_angle 5 --label modelofreeze_002_5

python entrenamiento.py --tosave ./resultados --cv split_train_cvfolds.json --gt ./FORENSE/data_curated_bbox.csv --boxes 3d_boxes.npy --ver ver_lst.npy --opt_mat m_sol_dict_train_cv.json --workers 2 --torch_workers 16 --w_out 32 --e_quantile 0.9 --model modelo1  --dataset midataset_roi --pretrained  --epochs 200 --batch 8 --w_factor=5 --lr 0.0004 --max_translation 0.02 --max_angle 15 --label modelofreeze_002_15

python entrenamiento.py --tosave ./resultados --cv split_train_cvfolds.json --gt ./FORENSE/data_curated_bbox.csv --boxes 3d_boxes.npy --ver ver_lst.npy --opt_mat m_sol_dict_train_cv.json --workers 2 --torch_workers 16 --w_out 32 --e_quantile 0.9 --model modelo1  --dataset midataset_roi --pretrained  --epochs 200 --batch 8 --w_factor=5 --lr 0.0004 --max_translation 0.02 --max_angle 25 --label modelofreeze_002_25


python entrenamiento.py --tosave ./resultados --cv split_train_cvfolds.json --gt ./FORENSE/data_curated_bbox.csv --boxes 3d_boxes.npy --ver ver_lst.npy --opt_mat m_sol_dict_train_cv.json --workers 2 --torch_workers 16 --w_out 32 --e_quantile 0.9 --model modelo1  --dataset midataset_roi --pretrained  --epochs 200 --batch 8 --w_factor=5 --lr 0.0004 --max_translation 0.11 --max_angle 5 --random_flip --label modelofreeze_011_5_flip

python entrenamiento.py --tosave ./resultados --cv split_train_cvfolds.json --gt ./FORENSE/data_curated_bbox.csv --boxes 3d_boxes.npy --ver ver_lst.npy --opt_mat m_sol_dict_train_cv.json --workers 2 --torch_workers 16 --w_out 32 --e_quantile 0.9 --model modelo1  --dataset midataset_roi --pretrained  --epochs 200 --batch 8 --w_factor=5 --lr 0.0004 --max_translation 0.11 --max_angle 15 --random_flip --label modelofreeze_011_15_flip

python entrenamiento.py --tosave ./resultados --cv split_train_cvfolds.json --gt ./FORENSE/data_curated_bbox.csv --boxes 3d_boxes.npy --ver ver_lst.npy --opt_mat m_sol_dict_train_cv.json --workers 2 --torch_workers 16 --w_out 32 --e_quantile 0.9 --model modelo1  --dataset midataset_roi --pretrained  --epochs 200 --batch 8 --w_factor=5 --lr 0.0004 --max_translation 0.11 --max_angle 25 --random_flip --label modelofreeze_011_25_flip

python entrenamiento.py --tosave ./resultados --cv split_train_cvfolds.json --gt ./FORENSE/data_curated_bbox.csv --boxes 3d_boxes.npy --ver ver_lst.npy --opt_mat m_sol_dict_train_cv.json --workers 2 --torch_workers 16 --w_out 32 --e_quantile 0.9 --model modelo1  --dataset midataset_roi --pretrained  --epochs 200 --batch 8 --w_factor=5 --lr 0.0004 --max_translation 0.11 --max_angle 5 --label modelofreeze_011_5

python entrenamiento.py --tosave ./resultados --cv split_train_cvfolds.json --gt ./FORENSE/data_curated_bbox.csv --boxes 3d_boxes.npy --ver ver_lst.npy --opt_mat m_sol_dict_train_cv.json --workers 2 --torch_workers 16 --w_out 32 --e_quantile 0.9 --model modelo1  --dataset midataset_roi --pretrained  --epochs 200 --batch 8 --w_factor=5 --lr 0.0004 --max_translation 0.11 --max_angle 15 --label modelofreeze_011_15

python entrenamiento.py --tosave ./resultados --cv split_train_cvfolds.json --gt ./FORENSE/data_curated_bbox.csv --boxes 3d_boxes.npy --ver ver_lst.npy --opt_mat m_sol_dict_train_cv.json --workers 2 --torch_workers 16 --w_out 32 --e_quantile 0.9 --model modelo1  --dataset midataset_roi --pretrained  --epochs 200 --batch 8 --w_factor=5 --lr 0.0004 --max_translation 0.11 --max_angle 25 --label modelofreeze_011_25


printf "$(date '+%H:%M:%S)')\n"
printf "\n\nFINALIZADO\n\n"
