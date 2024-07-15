import os

import json
import torch
import yaml
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from dltrain.arguments import parse_arguments, DATASET_CHOICES, MODEL_CHOICES, get_correction
from dltrain.data import flatten_and_index
from dltrain.matrix_optim import MatrixBuilder
from dltrain.utils import list_files, read_image, LM_NAMES
from dltrain.culling import FaceCulling



#define all images valid extensions
VALID_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp']

def calculo_normales():
    images_folder = "FORENSE"
    device = torch.device('cuda',0) if torch.cuda.is_available() else torch.device('cpu')

    OUT_DIR = "./"

    df = pd.read_csv("FORENSE/data_curated_bbox.csv")
    filenames = df.iloc[:, 0]

    # Config TDDFA
    tddfa_config = yaml.load(open('configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)
    tddfa = TDDFA(gpu_mode=torch.cuda.is_available(), **tddfa_config, gpu_id=0)
    face_boxes = FaceBoxes()

    # Config FaceCulling
    face_culling = FaceCulling(tddfa.tri)
    np_dataset_ver_lst = np.load("ver_lst.npy")


    ver_lst_all = np.array(np_dataset_ver_lst).squeeze()
    # Get normales
    normales = np.array([face_culling.get_normales(_ver) for _ver in ver_lst_all])
    
    normales = normales.transpose(0, 2, 1)

    np.save("normales_lst.npy",normales)


def mostrar_normales():
    df = pd.read_csv("FORENSE/data_curated_bbox.csv")
    filenames = df.iloc[:, 0]
    np_dataset_ver_lst = np.load("ver_lst.npy")
    normales = np.load("normales_lst.npy")
    
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from mpl_toolkits.mplot3d import Axes3D


    imagen_idx = 0

    # Cargar la imagen 2D
    img = mpimg.imread("FORENSE/" + filenames[imagen_idx])

    # Crear la figura
    fig = plt.figure()

    # Mostrar la imagen 2D
    ax1 = fig.add_subplot(111)
    ax1.imshow(img)
    ax1.axis('off')  # Ocultar los ejes de la imagen 2D

    ld_idx=0
    for ld_idx in range(np_dataset_ver_lst[imagen_idx].shape[1]):
        if (ld_idx % 30 != 0):
            continue
        recta_x = normales[imagen_idx][0][ld_idx]
        recta_y = normales[imagen_idx][1][ld_idx]
        recta_z = normales[imagen_idx][2][ld_idx]
        recta = np.array([recta_x,recta_y,recta_z])
        recta = recta/np.linalg.norm(recta)
    
        p_inicio_x = np_dataset_ver_lst[imagen_idx][0][ld_idx]
        p_inicio_y = np_dataset_ver_lst[imagen_idx][1][ld_idx]
        p_inicio_z = np_dataset_ver_lst[imagen_idx][2][ld_idx]
        inicio = np.array([p_inicio_x,p_inicio_y,p_inicio_z])

        # Coordenadas de la línea 3D
        t = np.linspace(0,5,50)
        x = inicio[0] + recta[0] * t
        y = inicio[1] + recta[1] * t


        # Dibujar la línea 3D
        ax1.plot(x, y, color='red')
        ax1.scatter(inicio[0],inicio[1],color="blue")





    # Mostrar la figura
    plt.show()

calculo_normales()
