import numpy as np
import os
import PIL.Image
from scipy.stats import f
import torch
import json
import matplotlib.pyplot as plt
import cv2

LM_NAMES = ["Menton", "Gnathion", "Pogonion", "Prosthion", "Labiale Superius", "Subnasale", "Nasion", "Glabella",
            "Vertex", "Left Gonion", "Right Gonion", "Left Zygion", "Right Zygion", "Left Alare", "Right Alare",
            "Left Endocanthion", "Right Endocanthion", "Left Exocanthlon", "Right Exocanthlon", "Left Tragion",
            "Right Tragion", "Infradentale", "Trichion", "Supramentale", "Left Frontotemporale",
            "Right Frontotemporale", "Left Frontozygomaticus", "Right Frontozygomaticus", "Left Midsurpaorbital",
            "Right Midsupraorbital", ]
LM_FLIP_IDX = np.arange(len(LM_NAMES))
LM_FLIP_IDX[['Left' in x for x in LM_NAMES]] += 1
LM_FLIP_IDX[['Right' in x for x in LM_NAMES]] -= 1

def list_files(p, valid_ext=None):
    files = next(os.walk(p))[2]
    if valid_ext is not None:
        files = [x for x in files if os.path.splitext(x)[1].lower() in valid_ext]
    return files


def get_pts(full_image_path, return_nans=False):
    with open(os.path.splitext(full_image_path)[0] + '.pts', 'r') as ifile:
        pts = ifile.read().splitlines()[3:-1]
        pts_np = np.array([x.split() for x in pts]).astype(float)
        if return_nans:
            pts_np[pts_np == -1] = np.nan
    return pts_np


def pts_bb(full_impath):
    return np.nanpercentile(get_pts(full_impath, return_nans=True), (0, 100), axis=0).reshape(-1)


def read_image(impath):
    im = np.array(
        PIL.Image.open(
            impath
        ).convert('RGB'))
    is_grayscale = len(im.shape) == 2
    if is_grayscale:
        im = np.tile(im[..., None], (1, 1, 3))
    return im


def iou_f(bboxes1, bboxes2):
    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    return iou


def crop_3ddfa(img, bbox):
    left, top, right, bottom = bbox[:4]
    old_size = (right - left + bottom - top) / 2
    center_x = right - (right - left) / 2.0
    center_y = bottom - (bottom - top) / 2.0 + old_size * 0.14
    size = int(old_size * 1.58)

    roi_box = [0] * 2
    roi_box[0] = center_x - size / 2
    roi_box[1] = center_y - size / 2

    s = 120 / size
    M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    M[:, 2] -= roi_box

    # sigma = 14e-4*size
    # img_t = cv2.GaussianBlur(img,(0,0),sigma)

    img = cv2.warpAffine(img, M * s, dsize=(120, 120), flags=cv2.INTER_LINEAR)
    return img

def crop_img(img, img_bb):
    x0, y0, x1, y1 = img_bb.astype(int)
    x_span, y_span = (x1 - x0), (y1 - y0)
    x0, y0, x1, y1 = np.clip((img_bb + [-x_span, -y_span, x_span, y_span]), 0, 1e+100).astype(int)
    img = img[y0:y1, x0:x1]
    return img, (x0, y0)

def HotellingT2(X,mu):
    n=len(X)
    mean=X.mean(0)
    S=(X-mean).T.dot(X-mean)/(n-1)
    return n*(mean-mu).dot(np.linalg.inv(S)).dot((mean-mu).T)

def t2_test(Xpop):
    mupop = np.array([0,0])
    t2 = HotellingT2(Xpop,mupop)
    n,p = Xpop.shape
    statistic = t2 * (n-p)/(p*(n-1))
    F = f(p, n-p)
    p_value = 1 - F.cdf(statistic)
    return p_value

def filter_nan(X,axis):
    return X[~np.isnan(X).any(axis)]

def mse_corrected(y,yp,correction=None):
    correction = np.ones_like(y)  if correction is None else correction
    return np.nanmean((((y-yp))**2).sum(1)/(correction**2))

class MSELossNaN(torch.nn.MSELoss):
    def forward(self, input, target):
        mask = ~torch.isnan(target)
        return super().forward(input[mask],target[mask])

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_M_json(M_sol_dict, M_sol_path):
    with open(M_sol_path,'w') as ofile:
        json.dump(M_sol_dict,ofile,cls=NumpyEncoder)

def load_M_json(m_sol_path):
    with open(m_sol_path,'r') as ifile:
        M_sol_dict_j = {
            int(cv_idx):{
                int(lm_idx): (np.array(m_tuple[0]),m_tuple[1])
                for lm_idx,m_tuple
                in e.items()
            }
            for cv_idx,e
            in json.load(ifile).items()
        }
    return M_sol_dict_j

def plot_history(train_history,epochs,title=None):
    fig,ax = plt.subplots(1)
    for k in train_history.keys():
        y_plot = train_history[k]
        x_plot = np.linspace(0,epochs,len(y_plot))
        ax.plot(x_plot,y_plot)
    ax.legend(train_history.keys())
    ax.set_yscale('log')
    if title is not None:
        ax.set_title(title)
    return fig

from tqdm import tqdm
from FaceBoxes import FaceBoxes
def get_correction(boxes):
    boxes = boxes.reshape(-1,5)
    dataset_correction = (boxes[:, [2, 3]] - boxes[:, [0, 1]]).max(1)
    return dataset_correction

def read_boxes(boxes_path):
    np_dataset_boxes = np.load(boxes_path)
    # dataset_correction = (np_dataset_boxes[:, [2, 3]] - np_dataset_boxes[:, [0, 1]]).max(1)
    # dataset_correction = get_correction(np_dataset_boxes)
    return np_dataset_boxes#, dataset_correction
def get_filelist_boxes(filenames,images_folder):
    face_boxes = FaceBoxes()
    # Get Bounding Boxes
    dataset_boxes = []
    empty_box = [np.nan]*5

    for img_filename in tqdm(filenames, desc='Loading images'):
        full_impath = os.path.join(images_folder, img_filename)
        # [[x0,y0,x1,y1]]
        cache_path = os.path.join(images_folder, img_filename + '.boxes.json')
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                boxes = json.load(f)
        else:
            img = read_image(full_impath)
            boxes = face_boxes(img)
            boxes = [[float(v) for v in b] for b in boxes]
            with open(cache_path, 'w') as f:
                json.dump(boxes, f)

        if len(boxes) == 0:
            print('No face detected in {}'.format(img_filename))
            boxes = [empty_box]
        elif len(boxes) > 1:
            print('More than one face detected on image {}'.format(img_filename))
            boxes = [empty_box]
        dataset_boxes.append(boxes[0])
    return dataset_boxes
import yaml
from TDDFA import TDDFA
def get_filelist_vertices(dataset_boxes, filenames, images_folder, gpu_mode=True, gpu_id=0):
    tddfa_config = yaml.load(open('configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)
    tddfa = TDDFA(gpu_mode=gpu_mode, **tddfa_config, gpu_id=gpu_id)

    # Get 3d vertices
    ver_lst_all = []
    for img_filename, boxes in tqdm(zip(filenames, dataset_boxes), total=len(filenames),
                                         desc='Loading vertices'):
        cache_path = os.path.join(images_folder, img_filename + '.vertices.npy')
        if os.path.exists(cache_path):
            ver_lst = np.load(cache_path)
        else:
            img = read_image(os.path.join(images_folder, img_filename))
            # Get 3d vertices
            param_lst, roi_box_lst = tddfa(img, [boxes])
            ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)
            np.save(cache_path, ver_lst)
        ver_lst_all.append(ver_lst)
    ver_lst_all = np.array(ver_lst_all).squeeze()
    return ver_lst_all

from scipy.spatial.distance import cdist
def get_ver_dist_list(np_dataset_pts_lst, np_dataset_ver_lst, dataset_correction):
    # Compute every 3DFFA vertex to Ground-truth distance
    dataset_dist_lst = []
    for ver_lst, pts_lst in zip(np_dataset_ver_lst, np_dataset_pts_lst):
        dist_matrix = cdist(ver_lst[:2].T, pts_lst)
        dataset_dist_lst.append(dist_matrix)

    np_dataset_dist_lst_norm = np.array(dataset_dist_lst) / dataset_correction[:, None, None]
    return np_dataset_dist_lst_norm