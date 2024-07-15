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

python entrenamiento.py --data ./FORENSE/ --gt ./FORENSE/data_curated_bbox.csv --tosave ./resultados --cv cv_folds.json --boxes 3d_boxes.npy --ver ver_lst.npy --opt_mat m_sol_dict_all_test.json --pre_mode 3ddfa --batch |_| --epochs 100 --lr |_| --workers 2 --torch_workers 16 --w_factor |_| --w_out 32 --e_quantile 0.9 --pretrained --max_translation 0.11 --max_angle 5 --reduce_factor 0.15  --model condresnet18pwise --label modelo1_|_| --n_landmarks 30 --dataset midataset_roi 




printf "$(date '+%H:%M:%S)')\n"
printf "\n\nFINALIZADO\n\n"
