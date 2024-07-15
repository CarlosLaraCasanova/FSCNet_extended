import os
import json
from copy import deepcopy
from typing import Dict, Tuple

import copy
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from tqdm.auto import tqdm, trange
import torch

from dltrain.arguments import parse_arguments, MODEL_CHOICES, DATASET_CHOICES, read_visiblity_lst, read_boxes, \
    read_ver_and_dist, read_kfold, read_pts_lst
from dltrain.models import custom_resnet_layers_to_freeze
from dltrain.utils import load_M_json, MSELossNaN, mse_corrected, plot_history, LM_NAMES, save_M_json, plot_accuracy, LM_NAMES
from dltrain.data import flatten_and_index, formatea_y_codifica
from dltrain.federated import aggregate_weights_mean, clip_nan_grad

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt


def run():



    args = parse_arguments()


    alfa = args.alfa
    beta = args.beta
    
    out_path = os.path.join(args.tosave, args.label)
    os.makedirs(out_path, exist_ok=True)

    device = torch.device('cuda', args.gpu_id)

    # Datapath
    FORENSIC_DS = "./FORENSE/"
    
    # Comput folds
    kfold_idxs = read_kfold(args)
    

    # # Load data
    # df1 = pd.read_csv(args.gt)
    # filenames = df1.iloc[:, 0]
    # np_dataset_pts_lst = df1.iloc[:, 1:].to_numpy().reshape((-1, 30, 2))
    # filenames: LISTA CON LOS NOMBRES DE LOS ARCHIVOS
    # np_dataset_pts_lst: LISTA CON LOS (X,Y) DE LOS LANDMARKS DE CADA ARCHIVO
    filenames, np_dataset_pts_lst = read_pts_lst(args)
    
    # VARIABLE AUXILIAR PARA PRUEBAS
    predicciones_pos = np.zeros((len(np_dataset_pts_lst), 30, 2)) + np.nan
    
    # np_dataset_boxes = LAS FACEBOXES DE CADA CARA EN CADA IMAGEN DE TRAIN
    # dataset_correction = LAS CORRECCIÓNES DE CADA FACEBOX EN CADA IMAGEN DE TRAIN
    np_dataset_boxes, dataset_correction = read_boxes(args)

    # Load pre-computed predictions
    # np_dataset_dist_lst_norm = np.array(dataset_dist_lst) / dataset_correction[:, None, None]
    # np_dataset_ver_lst: LISTA DE (X,Y,Z) DE LOS VÉRTICES DE LAS MÁSCARAS PARA CADA IMAGEN DE  ENTRENAMIENTO
    # np_dataset_dist_list_norm: LISTA DE MATRICES DE DISTANCIA NORMALIZADAS POR LA
    # 									 CORRECCIÓN. (DISTANCIA ENTRE VÉRTICES MÁSCARA Y LANDMRKS)
    np_dataset_ver_lst, np_dataset_dist_lst_norm = read_ver_and_dist(args)



    # Load transformation matrices
    # v_list: LISTA DE 4 VÉRTICES PARA CALCULAR LA TRANSOFRMACIÓN DE LA MÁSCARA
    #  			A UNA POSICIÓN COMÚN
    # v_ref: MEDIA DE LAS POSICIONES DE DICHOS 4 VÉRTICES EN LAS MÁSCARAS DE LOS
    # 		  DATOS DE ENTRENAMIENTO 
    v_list = [31517, 31106, 31980, 31509]
    v_ref = np_dataset_ver_lst[:, :, v_list].mean(0)


    # CALCULA LAS MATRICES DE ROTACIÓN PARA CADA IMAGEN DE ENTRADA
    # Y LAS GUARDA EN UN .JSON
    assert args.opt_mat is not None
    if not os.path.isfile(args.opt_mat):
        from dltrain.matrix_optim import optimize_matrix
        optimize_matrix(args,v_list,v_ref)


    # CARGA EL JSON CON LAS MATRICES DE ROTACIÓN PARA VERTEX Y TRICHION PARA CADA IMAGEN 
    # DIVIDO POR FOLDS!!
    M_sol_dict = load_M_json(args.opt_mat)

  
    # DICCIONARIO CON LOS VALORES DE LOS Nº VÉRTICES DE LAS MÁSCARAS Y SUS 
    # POSICIONES EN LA MÁSCARA NORMALIZADA
    pre_dict = dict(v_list=v_list, v_ref=v_ref)

    # EXPERIMENT_KEY = '3d_best_idx_corrected_01_RESNET18'
    # NOMBRE DEL EXPERIMENTO
    EXPERIMENT_KEY = args.label
    

    # Deeplearning config
    # batch_size = 64
    batch_size = args.batch
    # dataloader_num_workers = 0
    dataloader_num_workers = args.workers
    # epochs = 200
    epochs = args.epochs
    lr = args.lr
    
    
    
    # ROI config
    # window_factor = 2
    # MULTIPLICADOR DEL TAMAÑO DE LAS ROI
    window_factor = args.w_factor
    # window_outsize = 32
    # RESOLUCIÓN DE LAS ROI
    window_outsize = args.w_out
    # error_quantile = 0.9
    error_quantile = args.e_quantile


    # Load base model
    model = MODEL_CHOICES[args.model](dropout = args.dropout)
    
    model.to(device)
    base_sd = copy.deepcopy(model.state_dict())



    def train_joint_model(model, tr_dl, ts_dl, epochs, label='dl training', lr=lr):
    
    
        optim = torch.optim.Adam(model.parameters(), lr=lr)
        if args.reduce_times > 0:
            scheduler_every_n = int(epochs / (args.reduce_times+1))
            scheduler_gamma = args.reduce_factor
            scheduler = torch.optim.lr_scheduler.StepLR(optim, scheduler_every_n, scheduler_gamma)
        else:
            scheduler = None
        loss_reg_f = torch.nn.MSELoss(reduction='mean')
        loss_clas_f = torch.nn.CrossEntropyLoss(reduction='mean')


        TRAIN_KEY = 'train'
        TEST_KEY = 'test'
        keys = [TRAIN_KEY, TEST_KEY] if ts_dl is not None else [TRAIN_KEY]
        # {"train":tr_dl, "test":ts_dl}
        dl_dict = {k:dl for k,dl in zip(keys, [tr_dl, ts_dl])}

        loss_history = {k: [] for k in keys}
        clas_loss_history = {k: [] for k in keys}
        reg_loss_history = {k: [] for k in keys}
        acc_history = {k: [] for k in keys}
        
        pbar = trange(epochs, desc=label, leave=False)
        for e in pbar:

            for k in keys:
                is_train = k == TRAIN_KEY
                model.train(is_train)
                dl = dl_dict[k]

                for minibatch in dl:


                    mini_im, mini_lm, mini_ohe = flatten_and_index(*minibatch,filtern_nans=False)

                    model.zero_grad()	

                    mini_im = mini_im.to(device)
                    mini_lm_reg = mini_lm[:,:2].to(device)
                    mini_lm_clas = mini_lm[:,-1].type(torch.LongTensor).to(device)
                    mini_ohe = mini_ohe.to(device)
                    idx_visibles = (~mini_lm.isnan().any(1))

                    with torch.set_grad_enabled(is_train):
                        mini_pred = model(mini_im, mini_ohe)
                        mini_pred_reg = mini_pred[:,:2]
                        mini_pred_clas = mini_pred[:,-2:]
                        loss_class = loss_clas_f(mini_pred_clas, mini_lm_clas)
                        loss_reg = loss_reg_f(mini_pred_reg[idx_visibles], mini_lm_reg[idx_visibles])
                        loss = alfa*loss_reg + beta*loss_class
                        loss_history[k].append(loss.item())
                        reg_loss_history[k].append(loss_reg.item())
                        clas_loss_history[k].append(loss_class.item())
                            
                        '''
                        mini_pred_clase = torch.argmax(mini_pred, dim=1)
                        correctos = (mini_pred_clase == mini_lm).sum().item()
                        total = mini_lm.size(0)
                        accuracy = correctos/total
                        acc_history[k].append(accuracy)
                        '''

                    
                    if is_train:
                        loss.backward()
                        optim.step()
                    


            if scheduler is not None:
                scheduler.step()

            # Update pbar
            losses_pbar = {
                k: np.mean(loss_history[k][-len(dl_dict[k]):])
                for k
                in keys
            }
            pbar.set_postfix(**{k:losses_pbar[k] for k in keys})
   


        if TEST_KEY in keys:
            with torch.no_grad():
                mini_im, mini_lm, mini_ohe = formatea_y_codifica(*next(iter(dl_dict[TEST_KEY])))
                mini_im = mini_im.to(device)
                mini_ohe = mini_ohe.to(device)
                last_pred = model(mini_im, mini_ohe).detach().cpu().view((-1, len(LM_NAMES), 4))
        else:
            last_pred = None

        return loss_history, last_pred, acc_history, reg_loss_history, clas_loss_history


 
    # def get_maxerror(xy_relative_error_tr,error_quantile):
    #     q_maxerror = np.nanquantile(np.abs(xy_relative_error_tr), [error_quantile], 0).max((0, -1))
    #     if args.single_max:
    #         q_maxerror[:] = q_maxerror.max()
    #     return q_maxerror
    def get_q_maxerror(yp, yp_gt, correction):
        xy_relative_error = (yp - yp_gt) / correction[:, None, None]
        q_maxerror = np.nanquantile(np.abs(xy_relative_error), [error_quantile], 0).max((0, -1))
        if args.single_max:
            q_maxerror[:] = q_maxerror.max()
        return q_maxerror

    def register_config_buffers(q_maxerror):
        model.register_buffer('q_maxerror', torch.tensor(q_maxerror))
        model.register_buffer('window_factor', torch.tensor(window_factor))
        model.register_buffer('window_outsize', torch.tensor(window_outsize))

    def reset_model(q_maxerror):
        model.load_state_dict(base_sd,strict=False)
        register_config_buffers(q_maxerror)

    def joint_training(yp_all, tr_idx, ts_idx, q_maxerror, cv_idx, **kwargs):
        yp_tr, yp_ts = (yp_all[idx] for idx in (tr_idx, ts_idx))
        yp = (np.zeros((len(ts_idx), len(LM_NAMES), 4)) + np.nan) if ts_idx else None

        # Reset model
        reset_model(q_maxerror)

        # Set dataloader
        dataset_class = DATASET_CHOICES[args.dataset]
        do_jitter = 'jitter' in args.dataset

        tr_ds = dataset_class(path_list=[os.path.join(FORENSIC_DS, x) for x in filenames[tr_idx]],
                              center_list=yp_tr, ground_truth=np_dataset_pts_lst[tr_idx],
                              correction_list=dataset_correction[tr_idx],
                              q_maxerror=q_maxerror, window_factor=window_factor, window_outsize=window_outsize,
                              do_cache=True, do_augmentation=True, do_jitter=do_jitter,
                              max_angle=args.max_angle, max_translation=args.max_translation,
                              jitter_brightness=args.jitter_brightness, jitter_contrast=args.jitter_contrast,
                              jitter_saturation=args.jitter_saturation, jitter_hue=args.jitter_hue,
                              random_flip=args.random_flip, box_list = np_dataset_boxes[tr_idx]
                              )
        
        
        tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=dataloader_num_workers,
                           persistent_workers=True and dataloader_num_workers > 0, pin_memory=True)

        
        if ts_idx is not None:
            ts_ds = dataset_class(path_list=[os.path.join(FORENSIC_DS, x) for x in filenames[ts_idx]],
                                  center_list=yp_ts, ground_truth=np_dataset_pts_lst[ts_idx],
                                  correction_list=dataset_correction[ts_idx],
                                  q_maxerror=q_maxerror, window_factor=window_factor, window_outsize=window_outsize,
                                  do_cache=True, do_augmentation=False, box_list = np_dataset_boxes[ts_idx])
            ts_dl = DataLoader(ts_ds, batch_size=len(ts_ds), shuffle=False, drop_last=False,
                               persistent_workers=True and dataloader_num_workers > 0,
                               num_workers=dataloader_num_workers, pin_memory=True)
        else:
            ts_dl = None

        train_history, lm_yp_ts, acc_history, reg_loss_history, clas_loss_history = train_joint_model(model, tr_dl, ts_dl, epochs,
                                                    label=f'{cv_idx}: JOINT training')

        # Store cv prediction
        if ts_dl is not None:
            yp[:] = lm_yp_ts
            predicciones_pos[ts_idx] = ts_ds.inverse_lms(lm_yp_ts[:,:,0:2])
            
        # Save model & history
        model_key = f'{EXPERIMENT_KEY}_cv{cv_idx}'
        torch.save(model.state_dict(), os.path.join(out_path, model_key + '.pth'))
        if 'extra_info' in kwargs:
            save_M_json(kwargs['extra_info'], os.path.join(out_path, model_key + '_extra.json'))

        plot_history(train_history, epochs, title=model_key).savefig(os.path.join(out_path, model_key + '.jpg'))
        plot_history(reg_loss_history, epochs, title=model_key+" reg loss").savefig(os.path.join(out_path, model_key + '_reg.jpg'))
        plot_history(clas_loss_history, epochs, title=model_key+" class loss").savefig(os.path.join(out_path, model_key + '_class.jpg'))
        #plot_accuracy(acc_history, epochs, title=model_key).savefig(os.path.join(out_path, model_key + "_acc.jpg"))
        plt.close()
        return yp, train_history["test"][-1]

    def get_3ddfa_prediction(fold_idx, tr_idx) -> Tuple[np.ndarray, Dict]:
        if fold_idx < 0: # all data
            assert fold_idx in M_sol_dict
        # Compute best vertices idx
        mean_dist = np.nanmean(np_dataset_dist_lst_norm[tr_idx], axis=0)
        best_idx = mean_dist.argmin(axis=0)

        # Transform all
        yp_all = np_dataset_ver_lst[:, :2, best_idx].transpose((0, 2, 1))
        # POR CADA LANDMARK
        for vidx in range(30):
            # SI ES TRICHION O VERTEX
            if vidx in M_sol_dict.get(fold_idx, {}):
                # LA MATRIZ DE ROTACIÓN DE LOS EJEMPLOS DEL FOLD DEL LANDMARK VIDX
                M_sol = M_sol_dict[fold_idx][vidx][0]
                
                # COJEMOS LAS POSICIONES DE LOS VERTICES DEL MEJOR VÉRTICE PARA ESTE LANDMARK
                target_list = np_dataset_ver_lst[:, :, [best_idx[vidx]]]
                # SE AÑADE UN 1 A CADA COORD  (X,Y,Z) ---> (X,Y,Z,1)
                target_list_fill = np.concatenate((target_list, np.ones((target_list.shape[0], 1, 1))),
                                                  axis=1).transpose(
                    (0, 2, 1))
                    
                # LAS NUEVAS ESTIMACIONES PARA ESTOS LANDMARKS Y SE ACTUALIZAN
                new_yp_all = np.matmul(target_list_fill, M_sol).squeeze()[:, :2]
                yp_all[:, vidx] = new_yp_all

        return yp_all, {'best_idx': best_idx}



    # Initialize prediction
    yp = np.zeros((len(np_dataset_pts_lst), 30, 4)) + np.nan

    errores_folds = []


    for cv_idx, (tr_idx, ts_idx) in enumerate(tqdm(kfold_idxs)):
        # PREDICCIONES BASADAS EN 3DDFA
        yp_all, extra_info = get_3ddfa_prediction(cv_idx, tr_idx)
        extra_info.update(pre_dict)
        if M_sol_dict is not None:
            extra_info.update(dict(rot={k: v[1]['x'] for (k, v) in M_sol_dict[cv_idx].items()}))

        # xy_relative_error_tr = (yp_all[tr_idx] - np_dataset_pts_lst[tr_idx]) / dataset_correction[tr_idx, None, None]
        # q_maxerror = get_maxerror(xy_relative_error_tr, error_quantile)
        q_maxerror = get_q_maxerror(
            yp_all[tr_idx], np_dataset_pts_lst[tr_idx], dataset_correction[tr_idx])



        yp[ts_idx], error_idx = joint_training(yp_all,tr_idx, ts_idx, q_maxerror, cv_idx)
        errores_folds.append(error_idx)
   
 

    #CÓDIGO DE CARLOS LARA
    salida = ""
    for e in errores_folds:
        salida = salida + str(e) + " "
    salida = salida + "\n"
    
    ts_idx=kfold_idxs[0][1]
    true_positives = np.zeros(30)
    true_negatives = np.zeros(30)
    false_positives = np.zeros(30)
    false_negatives = np.zeros(30)
    filenames, np_dataset_visibility_list = read_visiblity_lst(args)
    for idx_imagen in range(len(ts_idx)):
        for landmark in range(30):
                     
             img = ts_idx[idx_imagen]
             pred = np.argmax(yp[img][landmark][2:])
             verdad = np_dataset_visibility_list[img][landmark]

             if (pred == verdad):
                 if (verdad):
                     true_positives[landmark] += 1
                 else:
                     true_negatives[landmark] += 1
             else:
                 if (verdad):
                     false_negatives[landmark] += 1
                 else:
                     false_positives[landmark] += 1                
   
    for landmark in range(30):
        salida = salida + LM_NAMES[landmark] + " " + str(int(true_positives[landmark])) + " " + str(int(true_negatives[landmark])) + " " + str(int(false_positives[landmark])) + " " + str(int(false_negatives[landmark])) + "\n"

    with open(os.path.join(out_path, 'resultados.txt'),'a') as f:
        f.write(salida)

    # CÓDIGO DE CARLOS LARA

    '''
    for i in range(len(ts_idx)):
        from dltrain.utils import read_image
        
        img = read_image("./FORENSE/" + filenames[ts_idx[i]])
        plt.Figure(figsize=(10, 10))
        # Plot image
        plt.imshow(img)
        for landmark in range(30):
            # Plot landmarks
            pred = predicciones_pos[ts_idx[i]][landmark]
            plt.scatter(pred[0], pred[1], s=1, c='red')
            
        # Save
        plt.savefig("./pruebas/"+ filenames[i])
        plt.close()            
    '''
    '''
    if args.train_all:
        all_idx = np.arange(len(np_dataset_pts_lst))
        cv_idx = -1
        yp_all, extra_info = get_3ddfa_prediction(cv_idx, all_idx)
        extra_info.update(pre_dict)
        if M_sol_dict is not None:
            extra_info.update(dict(rot={k: v[1]['x'] for (k, v) in M_sol_dict[cv_idx].items()}))

        q_maxerror = get_q_maxerror(yp_all, np_dataset_pts_lst, dataset_correction)
        # np.save(os.path.join(out_path, f'{EXPERIMENT_KEY}_all_q_maxerror.npy'), q_maxerror)
        joint_training(yp_all, all_idx, None, q_maxerror, 'ALL')
    '''
    # if args.only_maxerror:
    #     return

    # Save results
    np.save(os.path.join(out_path, f'{EXPERIMENT_KEY}'), yp)
    # Print error
    error_dict = {}
    #for i, n in enumerate(LM_NAMES):
    #    error_dict[n] = mse_corrected(np_dataset_pts_lst[:, i], yp[:, i], correction=dataset_correction)
    #print(pd.Series(error_dict))

    # Save config
    with open(os.path.join(out_path, 'config.json'), 'w') as ofile:
        json.dump(vars(args), ofile)


if __name__ == '__main__':
    run()
