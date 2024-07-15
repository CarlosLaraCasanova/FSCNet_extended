import json
import pandas as pd
import numpy as np
import argparse

from scipy.spatial.distance import cdist

from dltrain.data import *
from dltrain.models import *
from dltrain.utils import LM_NAMES

MODEL_CHOICES = dict(resnet18=get_custom_resnet18, vgg11bn=get_custom_vgg11_bn,
                     condresnet18=get_conditional_resnet18_embedding,
                     condresnet18conv=get_conditional_resnet18_convolution,
                     condresnet18pwise=get_conditional_resnet18_pointwise,
                     resnet18fed=get_custom_resnet18, mhresnet18=get_multihead_resnet18,
                     modelo1=get_modelo_cabeza_custom_resnet18_entrenada,
                     modelo2=get_modelo_completoresnet18, 
                     modelo_hibrido1=get_conditional_resnet18_convolution_hibrido,
                     modelo3=get_modelo_completoresnet18_con_normal)

DATASET_CHOICES = dict(torch=SingleLandmarkDataset, cv2=SingleLandmarkDatasetCV2,
                       augm=SingleLandmarkDatasetCV2Augmented, jitter=SingleLandmarkDatasetCV2Augmented,
                       all_noaug=AllLandmarkDatasetCV2Augmented,
                       all_augm=AllLandmarkDatasetCV2Augmented, all_jitter=AllLandmarkDatasetCV2Augmented,
                       midataset_roi=AllLandmarkDatasetCV2Augmented_clasificacion,
                       midataset_facebox=AllLandmarkDatasetCV2Augmented_clasificacion_caracompleta,
                       midataset_roi_hibrido=AllLandmarkDatasetCV2Augmented_hibrido,
                       midataset_facebox_normal=AllLandmarkDatasetCV2Augmented_clasificacion_caracompleta_con_normal)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--alfa', type=float, default=10, help='multiplicador error regresión entrenamiento híbrido')    
    parser.add_argument('--beta', type=float, default=1, help='multiplicador error clasificación entrenamiento híbrido')
    
    parser.add_argument('-d', '--data', type=str, help='images path')
    parser.add_argument('--gt', type=str, help='ground truth csv (size: [N, 1(filename) + 30*2(x,y)])')

    parser.add_argument('--tosave', type=str, help='Path to save models')
    parser.add_argument('--eval_out', type=str, help='Path to save evaluation results')
    parser.add_argument('--save_eval_images', action='store_true', help='Save evaluation images')
    parser.add_argument('--cv', type=str, help='5-fold train/test indices json path (size: [5, 2, _])')

    # All-purpose boxes
    parser.add_argument('--boxes', type=str, help='3DDFA boxes numpy path (size: [N, 4])')

    # 3ddfa predictions
    parser.add_argument('--ver', type=str, help='3DDFA output numpy path (size: [N, 3, 38365])')
    parser.add_argument('-m', '--opt_mat', type=str,
                        help='Optimization transformation matrices path (size: [5, n_landmarks, (M, opt_sol)])')
    parser.add_argument('--opt_maxiter', type=int, default=1000,
                        help='Optimization transformation max iterations (default: 1000)')

    # hrnet predictions
    # choice argument
    parser.add_argument('--pre_mode', choices=['3ddfa', 'hrnet'], default='3ddfa', help='3DDFA or HRNet prediction')
    parser.add_argument('--hrnet_pred', type=str, default=None,
                        help='HRNet output numpy path (size: [5, N, n_landmarks, 2])')

    parser.add_argument('-b', '--batch', type=int, default=5, help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=700, help='Epochs')
    parser.add_argument('--lr', type=float, default=0.0005, help='Model learning rate')
    parser.add_argument('--workers', type=int, default=0, help="Dataloader's num workers (0: faster, uses cache)")
    parser.add_argument('--torch_workers', type=int, default=torch.get_num_threads(), help="Torch intraop parallelism")

    parser.add_argument('--w_factor', type=float, default=2.5, help="Window's size factor")
    parser.add_argument('--w_out', type=int, help="Window's output size")
    parser.add_argument('--e_quantile', type=float, default=0.9, help="Window's size error quantile")
    parser.add_argument('--single_max', action='store_true')

    # Training options
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--freeze_layers', type=int, default=3, help='Freeze layers')
    parser.add_argument('--dropout', action='store_true')

    # Augmentation params
    parser.add_argument('--max_translation', type=float, default=0.06, help='Max translation')
    parser.add_argument('--max_angle', type=int, default=25, help='Max rotation in degrees')
    parser.add_argument('--jitter_brightness', type=float, default=0.3, help='Jitter brightness')
    parser.add_argument('--jitter_contrast', type=float, default=0.2, help='Jitter contrast')
    parser.add_argument('--jitter_saturation', type=float, default=0.9, help='Jitter saturation')
    parser.add_argument('--jitter_hue', type=float, default=0.0, help='Jitter saturation')
    parser.add_argument('--random_flip', action='store_true', help='Random flip')


    # LR scheduler
    parser.add_argument('--reduce_times', type=int, default=1, help='LR scheduler reduce times')
    parser.add_argument('--reduce_factor', type=float, default=0.15, help='LR scheduler reduce factor')



    parser.add_argument('--only_maxerror', action='store_true')
    parser.add_argument('--train_all', action='store_true')

    parser.add_argument('--merge_every', type=int, help='Federated: Merge every N epochs', default=1)
    parser.add_argument('--custom_layers', help='Federated: Custom layers', nargs="+", default=['fc'])
    parser.add_argument('--weighted', help='Federated: Weighted aggregation', action='store_true')

    parser.add_argument('--model', choices=MODEL_CHOICES.keys())
    parser.add_argument('--cmap', type=int,default=-1)
    parser.add_argument('--clip_nan', action='store_true')

    parser.add_argument('-l', '--label', type=str, help="Experiment's label")

    parser.add_argument('--gpu_id', type=int, help='GPU Device id', default=0)
    parser.add_argument('--n_landmarks', type=int, help='GPU Device id', default=len(LM_NAMES))

    do_split_in = ['torch', 'cv2', 'augm', 'jitter']
    parser.add_argument('--dataset', choices=DATASET_CHOICES.keys())

    # Validation arguments
    parser.add_argument('--save_correction', action='store_true', help='Save correction')
    parser.add_argument('--save_visibility', action='store_true', help='Save visibility')
    parser.add_argument('--save_boxes_df', action='store_true', help='Save Dataframe with boxes')

    parser.add_argument('--lm_agreement', type=str, help='Path to dispersion agreement JSON')
    parser.add_argument('--optuna_storage', type=str, help='Path to optuna storage')
    parser.add_argument('--optuna_study_name', type=str, help='Optuna study name')
    parser.add_argument('--optuna_n_trials', type=int, help='Optuna n trials')


    args = parser.parse_args()

    train_method = 'federated' if args.model == 'resnet18fed' else 'split' if args.dataset in do_split_in else 'joint'
    args.train_method = train_method

    return args

def read_pts_lst(args):
    # Load data
    df1 = pd.read_csv(args.gt)
    filenames = df1.iloc[:, 0]
    
    np_dataset_pts_lst = df1.iloc[:, 1:].to_numpy().reshape((-1, 30, 2))
    return filenames, np_dataset_pts_lst

def get_correction(boxes):
    dataset_correction = (boxes[:, [2, 3]] - boxes[:, [0, 1]]).max(1)
    return dataset_correction

def read_boxes(args):
    # DATOS DE LAS FACE BOXES DE CADA IMAGEN DE ENTRENAMIENTO
    np_dataset_boxes = np.load(args.boxes)
    # dataset_correction = (np_dataset_boxes[:, [2, 3]] - np_dataset_boxes[:, [0, 1]]).max(1)
    # CALCULO DE LA CORRECCIÓN
    dataset_correction = get_correction(np_dataset_boxes)
    return np_dataset_boxes, dataset_correction

def read_ver_and_dist(args):
    assert args.ver is not None

    # Lista de las (X,Y) de los landamarks para todos las imagenes de entrenamiento
    np_dataset_pts_lst = read_pts_lst(args)[1]
    # Lista de las correciones para cada imagen de las de entrenamiento
    dataset_correction = read_boxes(args)[1]
    
    # Lista de los (X,Y,Z) de los vértices de cada máscara para cada imagen de entrada
    np_dataset_ver_lst = np.load(args.ver)
    # Compute every 3DFFA vertex to Ground-truth distance
    dataset_dist_lst = []
    # pts_lst: LISTA DE (X,Y) DE LANDMARKS DE UNA IMAGEN DE TRAIN
    # ver_lst: LISTA DE (X,Y,Z) DE VÉRTICES DELA MÁSCARA DE UNA IMAGEN DE TRAIN
    for ver_lst, pts_lst in zip(np_dataset_ver_lst, np_dataset_pts_lst):
        # dist_matrix: DISTANCIA ENRE CADA LANDMARK Y VÉRTICE DE LA MÁSCARA (PROYECTADA)
        dist_matrix = cdist(ver_lst[:2].T, pts_lst)
        dataset_dist_lst.append(dist_matrix)
    # dataset_dist_lst: PARA CADA IMAGEN DE ENTRADA, LA DISTANCIA ENTRE SUS LANDMARKS
    #                  Y LOS VÉRTICES DE SU MÁSCARA
    # np_dataset_dist_lst_norm: SE NORMALIZA DICHA DISTANCIA
    np_dataset_dist_lst_norm = np.array(dataset_dist_lst) / dataset_correction[:, None, None]
    return np_dataset_ver_lst, np_dataset_dist_lst_norm

def read_kfold(args):
    with open(args.cv) as f:
        kfold_idxs = json.load(f)
    return kfold_idxs
    
# EMPIEZA CÓDIGO DE CARLOS LARA CASANOVA    
    
def read_visiblity_lst(args):
    # Load data
    df = pd.read_csv(args.gt)
    filenames = df.iloc[:, 0]
    
    np_dataset_puntos_list = df.iloc[:, 1:].to_numpy().reshape((-1, 30, 2))
    
    np_dataset_visibility_list = ~np.isnan(np_dataset_puntos_list)[:,:,0]
   
    return filenames, np_dataset_visibility_list
    
# TERMINA CÓDIGO DE CARLOS LARA CASANOVA
