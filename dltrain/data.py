from .utils import read_image, LM_FLIP_IDX

from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize, ColorJitter
from torchvision.transforms.functional import crop
from torchvision.ops import roi_align

import numpy as np
import torch
import PIL
import math

import cv2

import torchvision


class UnNormalize(torchvision.transforms.Normalize):
    def __init__(self, mean, std, *args, **kwargs):
        new_mean = [-m / s for m, s in zip(mean, std)]
        new_std = [1 / s for s in std]
        super().__init__(new_mean, new_std, *args, **kwargs)


imagenet_norm = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class SingleLandmarkDatasetBase(Dataset):
    def __init__(self, landmark_idx, path_list, center_list, correction_list, q_maxerror, window_factor,
                 window_outsize, ground_truth=None, do_cache=False, **kwargs):

        self.lm_idx = landmark_idx

        self.path_list = path_list
        self.center_list = center_list
        self.ground_truth = ground_truth if ground_truth is not None else center_list
        self.correction_list = correction_list

        assert (len(path_list) == len(center_list))
        assert (len(path_list) == len(correction_list))

        self.q_maxerror = q_maxerror
        self.window_factor = window_factor
        self.window_outsize = window_outsize

        self.f = Compose([ToTensor(),
                          Normalize(**imagenet_norm)])

        self.cache = {}
        self.do_cache = do_cache

    def __len__(self):
        return len(self.path_list)

    def __split_image__(self, img, pred, xyoffset):
        raise NotImplementedError

    def __getitem__(self, idx):
        if self.do_cache and (idx in self.cache):
            return self.cache[idx]

        img_filename = self.path_list[idx]
        img = read_image(img_filename)

        correction = self.correction_list[idx]
        pred = self.center_list[idx]
        xyoffset = self.q_maxerror[:, None] * correction * self.window_factor

        gt = ((self.ground_truth[idx] - pred) / xyoffset)[self.lm_idx]

        # Begin custom code
        img_splits = self.__split_image__(img, pred, xyoffset)

        result = img_splits, torch.tensor(gt).float()

        if self.do_cache:
            self.cache[idx] = result

        return result

    def inverse_lms(self, lms):
        assert (len(lms) == len(self.center_list))
        correction = self.correction_list
        pred = self.center_list[:, self.lm_idx]
        xyoffset = self.q_maxerror[self.lm_idx] * correction * self.window_factor

        # gt = ((self.ground_truth[idx]-pred)/xyoffset)[self.lm_idx]
        inv = (lms * xyoffset[:, None]) + pred
        return inv


class SingleLandmarkDataset(SingleLandmarkDatasetBase):
    def __split_image__(self, img, pred, xyoffset):
        boxes = np.concatenate((pred - xyoffset, pred + xyoffset), axis=1)
        boxes_tensor = [torch.tensor(boxes).float()[self.lm_idx:self.lm_idx + 1]]

        img_tensor = self.f(img)
        img_splits = roi_align(img_tensor[None, :], boxes_tensor, output_size=self.window_outsize)[0]
        return img_splits


class SingleLandmarkDatasetCV2(SingleLandmarkDatasetBase):
    def __split_image__(self, img, pred, xyoffset):
        s = 1 / xyoffset[self.lm_idx].item() * self.window_outsize / 2
        t = -pred[self.lm_idx] * s + self.window_outsize / 2

        M = np.diag([s, s, 1.]).astype(np.float32)[:2]
        M[:, -1] = t

        im_t = img.astype(np.float32) / 255.
        img = cv2.warpAffine(im_t, M, dsize=(self.window_outsize, self.window_outsize))
        img_tensor = self.f(img)

        return img_tensor


class SingleLandmarkDatasetCV2Augmented(Dataset):
    def __init__(self, landmark_idx, path_list, center_list, correction_list, q_maxerror, window_factor,
                 window_outsize, ground_truth=None, do_cache=False, do_augmentation=False, max_angle=20,
                 max_translation=0.25,
                 do_jitter=False, **kwargs):

        self.lm_idx = landmark_idx

        self.path_list = path_list
        self.center_list = center_list
        self.ground_truth = ground_truth if ground_truth is not None else center_list
        self.correction_list = correction_list

        assert (len(path_list) == len(center_list))
        assert (len(path_list) == len(correction_list))

        self.q_maxerror = q_maxerror
        self.window_factor = window_factor
        self.window_outsize = window_outsize

        f_steps = [ToTensor()] + \
                  ([ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)] if do_jitter else []) + \
                  [Normalize(**imagenet_norm)]
        self.f = Compose(f_steps)

        self.do_cache = do_cache
        self.cache = {}

        self.do_augmentation = do_augmentation
        self.max_angle = max_angle
        self.max_translation = max_translation * window_outsize

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        if idx not in self.cache:
            img_filename = self.path_list[idx]
            img = read_image(img_filename)

            if self.do_cache:
                self.cache[idx] = img
        else:
            img = self.cache[idx]

        correction = self.correction_list[idx]
        pred = self.center_list[idx, self.lm_idx]
        xyoffset = (self.q_maxerror[self.lm_idx] * correction * self.window_factor).item()

        s = 1 / xyoffset * self.window_outsize / 2
        t = -pred * s + self.window_outsize / 2

        M = np.diag([s, s, 1.])
        M[:2, -1] = t

        if self.do_augmentation:
            half_window = self.window_outsize / 2
            a = np.random.rand() * self.max_angle * 2 - self.max_angle
            tx, ty = np.random.rand(2) * self.max_translation * 2 - self.max_translation

            # image rotation matrix
            rot = cv2.getRotationMatrix2D((half_window, half_window), a, 1)
            rot[:2, -1] += [tx, ty]

            # ground-truth rotation matrix
            rot_ref = cv2.getRotationMatrix2D((0, 0), a, 1)
            rot_ref[:2, -1] += [tx / half_window, ty / half_window]

            M = np.matmul(rot, M)

        img = cv2.warpAffine(img, M[:2], dsize=(self.window_outsize, self.window_outsize))
        img_tensor = self.f(img)

        gt = ((self.ground_truth[idx] - pred) / xyoffset)[self.lm_idx]

        if self.do_augmentation:
            gt = np.matmul(rot_ref[:2], np.concatenate([gt, [1]]))

        result = img_tensor, torch.tensor(gt).float()
        return result

    def inverse_lms(self, lms):
        assert (len(lms) == len(self.center_list))
        assert (not self.do_augmentation)

        correction = self.correction_list
        pred = self.center_list[:, self.lm_idx]
        xyoffset = self.q_maxerror[self.lm_idx] * correction * self.window_factor

        # gt = ((self.ground_truth[idx]-pred)/xyoffset)[self.lm_idx]
        inv = (lms * xyoffset[:, None]) + pred
        return inv


class AllLandmarkDatasetCV2Augmented(Dataset):
    def __init__(self, path_list, center_list, correction_list, q_maxerror, window_factor,
                 window_outsize, ground_truth=None, do_cache=False, do_augmentation=False, max_angle=25,
                 max_translation=0.06, jitter_brightness=None, jitter_contrast=None, jitter_saturation=None,
                 jitter_hue=None, do_jitter=False, random_flip=False, **kwargs):

        self.path_list = path_list
        self.center_list = center_list
        self.ground_truth = ground_truth if ground_truth is not None else center_list
        self.correction_list = correction_list
        
        assert (len(path_list) == len(center_list))
        assert (len(path_list) == len(correction_list))

        self.q_maxerror = q_maxerror
        self.window_factor = window_factor
        self.window_outsize = window_outsize


        f_steps = [ToTensor()] + \
                  ([ColorJitter(brightness=jitter_brightness, contrast=jitter_contrast, saturation=jitter_saturation,
                                hue=jitter_hue
                                )] if do_jitter else []) + \
                  [Normalize(**imagenet_norm)]

        # f_steps = ([
        #                ColorJitter(brightness=jitter_brightness, contrast=jitter_contrast,
        #                            saturation=jitter_saturation, )]
        #            if do_jitter
        #            else []) + \
        #           [Normalize(**imagenet_norm), ToTensor()]
        # f_steps = [ToTensor(), Normalize(**imagenet_norm)]

        self.f = Compose(f_steps)

        self.do_cache = do_cache
        self.cache = {}

        self.do_augmentation = do_augmentation
        self.max_angle = max_angle
        self.max_translation = max_translation * window_outsize
        self.random_flip = random_flip

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        if idx not in self.cache:
            img_filename = self.path_list[idx]
            img = read_image(img_filename)

            if self.do_cache:
                self.cache[idx] = img
        else:
            img = self.cache[idx]

        # [size]
        correction = self.correction_list[idx]
        # [size, 30, 2]
        pred = self.center_list[idx]
        # [size]
        xyoffset = (correction * self.q_maxerror * self.window_factor)
	
        s = (1 / xyoffset) * self.window_outsize / 2
        t = -pred * s[:, None] + self.window_outsize / 2

        M = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]] * 30, dtype=np.float32)
        M[:, :2] *= s[:, None, None]
        M[:, :-1, -1] = t


        if self.do_augmentation:
            half_window = self.window_outsize / 2
            a = np.random.rand() * self.max_angle * 2 - self.max_angle
            # tx, ty = (np.random.rand(2) * self.max_translation * 2 - self.max_translation)
            tx, ty = self.max_translation * (np.random.rand(2) * 2 - 1)

            # image rotation matrix
            rot = cv2.getRotationMatrix2D((half_window, half_window), a, 1)
            rot[:2, -1] += [tx, ty]

            # ground-truth rotation matrix
            rot_ref = cv2.getRotationMatrix2D((0, 0), a, 1)
            rot_ref[:2, -1] += [tx / half_window, ty / half_window]

            M = np.matmul(rot, M)

        #print(M)

        tout = torch.stack(
            [self.f(cv2.warpAffine(img, m, dsize=(self.window_outsize, self.window_outsize))) for m in M[:, :2]])


        gt = (self.ground_truth[idx] - pred) / xyoffset[:, None]


        if self.do_augmentation:
            gt = np.matmul(rot_ref, np.concatenate([gt, np.ones((30, 1))], 1).T).T

        if self.random_flip and np.random.rand() > 0.5:
            # Flip images
            tout = torch.flip(tout, [3])
            tout = tout[LM_FLIP_IDX]
            # Flip x coordinate
            gt[:, 0] *= -1

        result = tout, torch.tensor(gt).float()
        return result

    def inverse_lms(self, lms):
        assert (lms.shape == (len(self.center_list), len(self.q_maxerror), 2))
        assert (not self.do_augmentation)
        assert (not self.random_flip)

        correction = self.correction_list
        # [size, 30, 2]
        pred = self.center_list
        # [size, 30]
        xyoffset = self.q_maxerror[None, :] * correction[:, None] * self.window_factor

        # gt = ((self.ground_truth[idx]-pred)/xyoffset)[self.lm_idx]
        inv = (lms * xyoffset[:, :, None]) + pred
        return inv

#

def flatten_and_index(*args, labels=30, filtern_nans=False):
    ret = tuple(e.view(-1, *e.shape[2:]) for e in args)
    idxs = torch.arange(labels).repeat(int(ret[0].size(0) / labels))
    if filtern_nans:
        ret_msk = (~ret[1].isnan().any(1))
        ohe = torch.nn.functional.one_hot(idxs[ret_msk], num_classes=labels).float()
        return (*(e[ret_msk] for e in ret), ohe)
    else:
        ohe = torch.nn.functional.one_hot(idxs, num_classes=labels).float()
        return (*ret, ohe)
        
        
        
# EMPIEZA CÓDIGO DE CARLOS LARA CASANOVA

def formatea_y_codifica(*args, labels=30):
    ret = tuple(e.view(-1, *e.shape[2:]) for e in args)
    idxs = torch.arange(labels).repeat(int(ret[0].size(0) / labels))
    
    ohe = torch.nn.functional.one_hot(idxs, num_classes=labels).float()
    return (*ret,ohe)
    
def formatea_y_codifica_normales(*args, labels=30):
    ret = tuple(e.view(-1, *e.shape[2:]) for e in args)
    idxs = torch.arange(labels).repeat(int(ret[0].size(0) / labels))
    
    ohe = torch.nn.functional.one_hot(idxs, num_classes=labels).float()
    return (*ret,ohe)

class AllLandmarkDatasetCV2Augmented_clasificacion(Dataset):
    def __init__(self, path_list, center_list, visible_list, correction_list, q_maxerror, window_factor,
                 window_outsize, do_cache=False, do_augmentation=False, max_angle=25,
                 max_translation=0.06, jitter_brightness=None, jitter_contrast=None, jitter_saturation=None,
                 jitter_hue=None, do_jitter=False, random_flip=False, **kwargs):

        self.path_list = path_list
        self.center_list = center_list
        self.visible_list = visible_list
        self.correction_list = correction_list

        
        assert (len(path_list) == len(visible_list))
        assert (len(path_list) == len(correction_list))

        self.q_maxerror = q_maxerror
        self.window_factor = window_factor
        self.window_outsize = window_outsize


        # FUNCIÓN QUE APLICA EL DATA AUGMENTATION Y LA NORMALIZACIÓN DE LA ENTRADA
        f_steps = [ToTensor()] + \
                  ([ColorJitter(brightness=jitter_brightness, contrast=jitter_contrast, saturation=jitter_saturation,
                                hue=jitter_hue
                                )] if do_jitter else []) + \
                  [Normalize(**imagenet_norm)]

        self.f = Compose(f_steps)



        # UTILIZACIÓN DE LA CACHE 
        self.do_cache = do_cache
        self.cache = {}
        
        # PARÁMETROS DEL AUMENTO DE DATOS
        self.do_augmentation = do_augmentation
        self.max_angle = max_angle
        self.max_translation = max_translation * window_outsize
        self.random_flip = random_flip

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        if idx not in self.cache:
            img_filename = self.path_list[idx]
            img = read_image(img_filename)

            if self.do_cache:
                self.cache[idx] = img
        else:
            img = self.cache[idx]


        # [size]
        correction = self.correction_list[idx]
        # [size, 30, 2]
        pred = self.center_list[idx]
        # [size]
        xyoffset = (correction * self.q_maxerror * self.window_factor)
        # [size, 30, 1]
        valor_visibilidad = self.visible_list[idx]	


        # CÁLCULO DE LAS TRANSFORMACIONES DE LA IMAGEN A LA VENTANA QUE CONTIENE EL LANDMARK	
        s = (1 / xyoffset) * self.window_outsize / 2
        t = -pred * s[:, None] + self.window_outsize / 2


        M = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]] * 30, dtype=np.float32)
        M[:, :2] *= s[:, None, None]
        M[:, :-1, -1] = t



        # SI SE HACE AUMENTO DE DATOS SE MODIFICA LA MATRIZ DE TRANSFORMACIÓN 
        # PARA QUE HAGA AUMENTO DE DATOS A CADA VENTANA DE CADA LANDMARK
        if self.do_augmentation:
            half_window = self.window_outsize / 2
            a = np.random.rand() * self.max_angle * 2 - self.max_angle
            # tx, ty = (np.random.rand(2) * self.max_translation * 2 - self.max_translation)
            tx, ty = self.max_translation * (np.random.rand(2) * 2 - 1)

            # image rotation matrix
            rot = cv2.getRotationMatrix2D((half_window, half_window), a, 1)
            rot[:2, -1] += [tx, ty]


            M = np.matmul(rot, M)



        tout = torch.stack(
            [self.f(cv2.warpAffine(img, m, dsize=(self.window_outsize, self.window_outsize))) for m in M[:, :2]])




        if self.random_flip and np.random.rand() > 0.5:
            # Flip images
            tout = torch.flip(tout, [3])
            tout = tout[LM_FLIP_IDX]


        valor_visibilidad = [int(x) for x in valor_visibilidad]

        result = tout, torch.tensor(valor_visibilidad).long()
        return result


class AllLandmarkDatasetCV2Augmented_clasificacion_caracompleta(Dataset):
    def __init__(self, path_list, visible_list, box_list, window_outsize=224, do_cache=False, do_augmentation=False, max_angle=25,
                 max_translation=0.06, jitter_brightness=None, jitter_contrast=None, jitter_saturation=None,
                 jitter_hue=None, do_jitter=False, random_flip=False, **kwargs):

        self.path_list = path_list
        self.visible_list = visible_list
        self.window_outsize=window_outsize
        self.box_list = box_list

        assert (len(path_list) == len(visible_list))
        assert (len(path_list) == len(box_list))


        # FUNCIÓN QUE APLICA EL DATA AUGMENTATION Y LA NORMALIZACIÓN DE LA ENTRADA
        f_steps = [ToTensor()] + \
                  ([ColorJitter(brightness=jitter_brightness, contrast=jitter_contrast, saturation=jitter_saturation,
                                hue=jitter_hue
                                )] if do_jitter else []) + \
                  [Normalize(**imagenet_norm)]

        self.f = Compose(f_steps)



        # UTILIZACIÓN DE LA CACHE 
        self.do_cache = do_cache
        self.cache = {}
        
        # PARÁMETROS DEL AUMENTO DE DATOS
        self.do_augmentation = do_augmentation
        self.max_angle = max_angle
        self.max_translation = max_translation * window_outsize
        self.random_flip = random_flip

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        if idx not in self.cache:
            img_filename = self.path_list[idx]
            img = read_image(img_filename)

            if self.do_cache:
                self.cache[idx] = img
        else:
            img = self.cache[idx]



        # [size, 30, 1]
        valor_visibilidad = self.visible_list[idx]	

        extra_espacio = 70/2

        box = self.box_list[idx]
        box = [box[0]-extra_espacio,box[1]-extra_espacio,box[2]+extra_espacio,box[3]+extra_espacio]
        yoffset = box[3]-box[1]
        xoffset = box[2]-box[0]
        centro = [box[0]+xoffset/2, box[1]+yoffset/2]
        # CÁLCULO DE LAS TRANSFORMACIONES DE LA IMAGEN A LA VENTANA QUE CONTIENE EL LANDMARK	
        s_x = (1 / xoffset) * self.window_outsize
        s_y = (1 / yoffset) * self.window_outsize
        t_x = -box[0] * s_x
        t_y = -box[1] * s_y


        M = np.array([[[s_x, 0, t_x], [0, s_y, t_y], [0, 0, 1]]] * 30, dtype=np.float32)


        if self.do_augmentation:
            half_window = self.window_outsize / 2
            a = np.random.rand() * self.max_angle * 2 - self.max_angle
            # tx, ty = (np.random.rand(2) * self.max_translation * 2 - self.max_translation)
            tx, ty = 0 * (np.random.rand(2) * 2 - 1)

            # image rotation matrix
            rot = cv2.getRotationMatrix2D((half_window, half_window), a, 1)
            rot[:2, -1] += [tx, ty]


            M = np.matmul(rot, M)


        

        tout = torch.stack(
            [self.f(cv2.warpAffine(img, m, dsize=(self.window_outsize, self.window_outsize))) for m in M[:, :2]])
 
        

        if self.random_flip and np.random.rand() > 0.5:
            # Flip images
            tout = torch.flip(tout, [3])
            tout = tout[LM_FLIP_IDX]


        valor_visibilidad = [int(x) for x in valor_visibilidad]

        result = tout, torch.tensor(valor_visibilidad).long()
        return result


class AllLandmarkDatasetCV2Augmented_hibrido(Dataset):
    def __init__(self, path_list, center_list, correction_list, q_maxerror, window_factor,
                 window_outsize, ground_truth=None, do_cache=False, do_augmentation=False, max_angle=25,
                 max_translation=0.06, jitter_brightness=None, jitter_contrast=None, jitter_saturation=None,
                 jitter_hue=None, do_jitter=False, random_flip=False, **kwargs):

        self.path_list = path_list
        self.center_list = center_list
        self.correction_list = correction_list
        self.ground_truth = ground_truth if ground_truth is not None else center_list
        
        assert (len(path_list) == len(center_list))
        assert (len(path_list) == len(correction_list))

        self.q_maxerror = q_maxerror
        self.window_factor = window_factor
        self.window_outsize = window_outsize


        # FUNCIÓN QUE APLICA EL DATA AUGMENTATION Y LA NORMALIZACIÓN DE LA ENTRADA
        f_steps = [ToTensor()] + \
                  ([ColorJitter(brightness=jitter_brightness, contrast=jitter_contrast, saturation=jitter_saturation,
                                hue=jitter_hue
                                )] if do_jitter else []) + \
                  [Normalize(**imagenet_norm)]

        self.f = Compose(f_steps)



        # UTILIZACIÓN DE LA CACHE 
        self.do_cache = do_cache
        self.cache = {}
        
        # PARÁMETROS DEL AUMENTO DE DATOS
        self.do_augmentation = do_augmentation
        self.max_angle = max_angle
        self.max_translation = max_translation * window_outsize
        self.random_flip = random_flip

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        if idx not in self.cache:
            img_filename = self.path_list[idx]
            img = read_image(img_filename)

            if self.do_cache:
                self.cache[idx] = img
        else:
            img = self.cache[idx]


        # [size]
        correction = self.correction_list[idx]
        # [size, 30, 2]
        pred = self.center_list[idx]
        # [size]
        xyoffset = (correction * self.q_maxerror * self.window_factor)



        # CÁLCULO DE LAS TRANSFORMACIONES DE LA IMAGEN A LA VENTANA QUE CONTIENE EL LANDMARK	
        s = (1 / xyoffset) * self.window_outsize / 2
        t = -pred * s[:, None] + self.window_outsize / 2


        M = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]] * 30, dtype=np.float32)
        M[:, :2] *= s[:, None, None]
        M[:, :-1, -1] = t



        # SI SE HACE AUMENTO DE DATOS SE MODIFICA LA MATRIZ DE TRANSFORMACIÓN 
        # PARA QUE HAGA AUMENTO DE DATOS A CADA VENTANA DE CADA LANDMARK
        if self.do_augmentation:
            half_window = self.window_outsize / 2
            a = np.random.rand() * self.max_angle * 2 - self.max_angle
            # tx, ty = (np.random.rand(2) * self.max_translation * 2 - self.max_translation)
            tx, ty = self.max_translation * (np.random.rand(2) * 2 - 1)

            # image rotation matrix
            rot = cv2.getRotationMatrix2D((half_window, half_window), a, 1)
            rot[:2, -1] += [tx, ty]
            
            # ground-truth rotation matrix
            rot_ref = cv2.getRotationMatrix2D((0, 0), a, 1)
            rot_ref[:2, -1] += [tx / half_window, ty / half_window]

            M = np.matmul(rot, M)



        tout = torch.stack(
            [self.f(cv2.warpAffine(img, m, dsize=(self.window_outsize, self.window_outsize))) for m in M[:, :2]])


        gt = (self.ground_truth[idx] - pred) / xyoffset[:, None]

        if self.do_augmentation:
            gt = np.matmul(rot_ref, np.concatenate([gt, np.ones((30, 1))], 1).T).T

        if self.random_flip and np.random.rand() > 0.5:
            # Flip images
            tout = torch.flip(tout, [3])
            tout = tout[LM_FLIP_IDX]
            # Flip x coordinate
            gt[:, 0] *= -1


        salida = [[value[0],value[1],int(not np.isnan(value[0]))] for value in gt]

        result = tout, torch.tensor(salida).float()
        return result
        
    def inverse_lms(self, lms):
        assert (lms.shape == (len(self.center_list), len(self.q_maxerror), 2))
        assert (not self.do_augmentation)
        assert (not self.random_flip)

        correction = self.correction_list
        # [size, 30, 2]
        pred = self.center_list
        # [size, 30]
        xyoffset = self.q_maxerror[None, :] * correction[:, None] * self.window_factor

        # gt = ((self.ground_truth[idx]-pred)/xyoffset)[self.lm_idx]
        inv = (lms * xyoffset[:, :, None]) + pred
        return inv


class AllLandmarkDatasetCV2Augmented_clasificacion_caracompleta_con_normal(Dataset):
    def __init__(self, path_list, visible_list, box_list, normal_list=None, window_outsize=224, do_cache=False, do_augmentation=False, max_angle=25,
                 max_translation=0.06, jitter_brightness=None, jitter_contrast=None, jitter_saturation=None,
                 jitter_hue=None, do_jitter=False, random_flip=False, **kwargs):

        self.path_list = path_list
        self.visible_list = visible_list
        self.window_outsize=window_outsize
        self.box_list = box_list
        self.normal_list = normal_list

        assert (len(path_list) == len(visible_list))
        assert (len(path_list) == len(box_list))


        # FUNCIÓN QUE APLICA EL DATA AUGMENTATION Y LA NORMALIZACIÓN DE LA ENTRADA
        f_steps = [ToTensor()] + \
                  ([ColorJitter(brightness=jitter_brightness, contrast=jitter_contrast, saturation=jitter_saturation,
                                hue=jitter_hue
                                )] if do_jitter else []) + \
                  [Normalize(**imagenet_norm)]

        self.f = Compose(f_steps)



        # UTILIZACIÓN DE LA CACHE 
        self.do_cache = do_cache
        self.cache = {}
        
        # PARÁMETROS DEL AUMENTO DE DATOS
        self.do_augmentation = do_augmentation
        self.max_angle = max_angle
        self.max_translation = max_translation * window_outsize
        self.random_flip = random_flip

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        if idx not in self.cache:
            img_filename = self.path_list[idx]
            img = read_image(img_filename)

            if self.do_cache:
                self.cache[idx] = img
        else:
            img = self.cache[idx]



        # [size, 30, 1]
        valor_visibilidad = self.visible_list[idx]	

        extra_espacio = 70/2

        box = self.box_list[idx]
        box = [box[0]-extra_espacio,box[1]-extra_espacio,box[2]+extra_espacio,box[3]+extra_espacio]
        yoffset = box[3]-box[1]
        xoffset = box[2]-box[0]
        centro = [box[0]+xoffset/2, box[1]+yoffset/2]
        normals = torch.tensor(self.normal_list[idx])
        # CÁLCULO DE LAS TRANSFORMACIONES DE LA IMAGEN A LA VENTANA QUE CONTIENE EL LANDMARK	
        s_x = (1 / xoffset) * self.window_outsize
        s_y = (1 / yoffset) * self.window_outsize
        t_x = -box[0] * s_x
        t_y = -box[1] * s_y


        M = np.array([[[s_x, 0, t_x], [0, s_y, t_y], [0, 0, 1]]] * 30, dtype=np.float32)

        matriz_rotacion = None
        if self.do_augmentation:
            half_window = self.window_outsize / 2
            a = np.random.rand() * self.max_angle * 2 - self.max_angle
            # tx, ty = (np.random.rand(2) * self.max_translation * 2 - self.max_translation)
            tx, ty = 0 * (np.random.rand(2) * 2 - 1)

            # image rotation matrix
            rot = cv2.getRotationMatrix2D((half_window, half_window), a, 1)
            rot[:2, -1] += [tx, ty]


            M = np.matmul(rot, M)
            matriz_rotacion = torch.eye(3,3)
            a = math.radians(a)
            matriz_rotacion[0][0] = math.cos(a)
            matriz_rotacion[0][1] = -math.sin(a)
            matriz_rotacion[1][1] = math.cos(a)
            matriz_rotacion[1][0] = math.sin(a)
            normals = torch.matmul(matriz_rotacion, normals)


        

        tout = torch.stack(
            [self.f(cv2.warpAffine(img, m, dsize=(self.window_outsize, self.window_outsize))) for m in M[:, :2]])
 
        

        if self.random_flip and np.random.rand() > 0.5:
            # Flip images
            tout = torch.flip(tout, [3])
            tout = tout[LM_FLIP_IDX]
            matriz_flip = torch.eye(3,3)
            matriz_flip[0][0] = -1
            normals = torch.matmul(matriz_flip,normals)


        valor_visibilidad = [int(x) for x in valor_visibilidad]

        normals = torch.transpose(normals,0,1)
        norms = torch.norm(normals, dim=1, keepdim=True)
        normals = normals/norms

        

        result = tout, normals, torch.tensor(valor_visibilidad).long()
        return result

# TERMINA CÓDIGO DE CARLOS LARA CASANOVA
