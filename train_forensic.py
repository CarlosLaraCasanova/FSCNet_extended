import os
import json
from copy import deepcopy
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from tqdm import tqdm, trange
import torch

from dltrain.arguments import parse_arguments, MODEL_CHOICES, DATASET_CHOICES, read_pts_lst, \
    read_ver_lst, read_kfold
from dltrain.models import custom_resnet_layers_to_freeze
from dltrain.utils import load_M_json, MSELossNaN, mse_corrected, plot_history, LM_NAMES, save_M_json, \
    get_filelist_boxes, read_boxes, get_correction, get_filelist_vertices, get_ver_dist_list,\
    save_M_json
from dltrain.data import flatten_and_index
from dltrain.federated import aggregate_weights_mean, clip_nan_grad
from dltrain.matrix_optim import optimize_matrix

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from sklearn.model_selection import KFold

def run():
    args = parse_arguments()

    train_method = args.train_method

    out_path = os.path.join(args.tosave, args.label)
    os.makedirs(out_path, exist_ok=True)

    device = torch.device('cuda', args.gpu_id)

    # Datapath
    FORENSIC_DS = args.data

    # Comput folds
    kfold_idxs = read_kfold(args)

    # # Load data
    # df1 = pd.read_csv(args.gt)
    # filenames = df1.iloc[:, 0]
    # np_dataset_pts_lst = df1.iloc[:, 1:].to_numpy().reshape((-1, 30, 2))
    filenames, np_dataset_pts_lst = read_pts_lst(args)


    # Normalize distances
    # np_dataset_boxes = np.load(args.boxes)
    # dataset_correction = (np_dataset_boxes[:, [2, 3]] - np_dataset_boxes[:, [0, 1]]).max(1)
    # if args.boxes:
    #     np_dataset_boxes= read_boxes(args.boxes)

    np_dataset_boxes = np.array(get_filelist_boxes(filenames, args.data))


    # Remove files without boxes

    # np_dataset_boxes = np.array([e for b,e in zip(boxes_mask,dataset_boxes) if b])
    # filenames = np.array([e for b,e in zip(boxes_mask,filenames) if b])
    # np_dataset_pts_lst = np_dataset_pts_lst[boxes_mask]

    assert not np.isnan(np_dataset_boxes).any()
    # boxes_mask = np.isnan(np_dataset_boxes).any(1)
    # drop_idxs = np.arange(len(boxes_mask))[boxes_mask]
    # kfold_idxs = [[[v for v in c if v not in drop_idxs] for c in b] for b in kfold_idxs]


    # mask out empty boxes
    dataset_correction = get_correction(np_dataset_boxes)

    # Load pre-computed predictions
    M_sol_dict = None
    if args.pre_mode == 'hrnet':
        assert args.hrnet_pred is not None
        np_dataset_hrnet_lst = np.load(args.hrnet_pred)
        pre_dict = {}

    elif args.pre_mode == '3ddfa':
        # assert args.ver is not None
        # assert args.opt_mat is not None
        #
        # np_dataset_ver_lst = np.load(args.ver)
        # # Compute every 3DFFA vertex to Ground-truth distance
        # dataset_dist_lst = []
        # for ver_lst, pts_lst in zip(np_dataset_ver_lst, np_dataset_pts_lst):
        #     dist_matrix = cdist(ver_lst[:2].T, pts_lst)
        #     dataset_dist_lst.append(dist_matrix)
        #
        # np_dataset_dist_lst_norm = np.array(dataset_dist_lst) / dataset_correction[:, None, None]
        if args.ver is not None:
            np_dataset_ver_lst = read_ver_lst(args)
        else:
            np_dataset_ver_lst = get_filelist_vertices(np_dataset_boxes, filenames, args.data,
                                                       gpu_mode = torch.cuda.is_available(), gpu_id = args.gpu_id)
        np_dataset_dist_lst_norm = get_ver_dist_list(np_dataset_pts_lst, np_dataset_ver_lst, dataset_correction)

        # Load transformation matrices
        v_list = [31517, 31106, 31980, 31509]
        v_ref = np_dataset_ver_lst[:, :, v_list].mean(0)

        # assert args.opt_mat is not None
        opt_mat_path = args.opt_mat if args.opt_mat is not None else os.path.join(args.data, 'opt_mat.json')
        if not os.path.isfile(opt_mat_path):
            M_sol_dict, _ = optimize_matrix(v_list,v_ref,np_dataset_pts_lst, dataset_correction,np_dataset_ver_lst, np_dataset_dist_lst_norm,
                    kfold_idxs, args.opt_maxiter)
            save_M_json(M_sol_dict, opt_mat_path)
        M_sol_dict = load_M_json(opt_mat_path)

        pre_dict = dict(v_list=v_list, v_ref=v_ref)


    # EXPERIMENT_KEY = '3d_best_idx_corrected_01_RESNET18'
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
    window_factor = args.w_factor
    # window_outsize = 32
    window_outsize = args.w_out
    # error_quantile = 0.9
    error_quantile = args.e_quantile

    # Load base model
    model = MODEL_CHOICES[args.model](cmap=None if args.cmap < 0 else args.cmap, pretrained=args.pretrained,
                                      dropout=args.dropout)
    model.to(device)
    base_sd = model.state_dict()

    # freeze layers
    if args.freeze_layers > 0:
        layers_to_freeze = custom_resnet_layers_to_freeze[args.freeze_layers]
        for l in layers_to_freeze:
            getattr(model, l).requires_grad = False

    # Aux training functions
    def filter_nan_idxs(lm_idx, filename_idxs):
        idxs_mask = ~np.isnan(np_dataset_pts_lst[filename_idxs][:, lm_idx]).any(-1)
        return np.array(filename_idxs)[idxs_mask], idxs_mask

    def train_split_model(model, tr_dl, ts_dl, epochs, label='dl training', lr=lr):
        optim = torch.optim.Adam(model.parameters(), lr=lr)
        loss_f = MSELossNaN()

        TRAIN_KEY = 'train'
        TEST_KEY = 'test'
        keys = [TRAIN_KEY, TEST_KEY] if ts_dl is not None else [TRAIN_KEY]
        dl_dict = {k: dl for k, dl in zip(keys, [tr_dl, ts_dl])}

        loss_history = {k: [] for k in keys}

        for e in tqdm(range(epochs), desc=label, leave=False):
            for k in keys:
                is_train = k == TRAIN_KEY
                dl = dl_dict[k]
                for mini_im, mini_lm in dl:
                    model.zero_grad()
                    mini_im = mini_im.to(device)
                    mini_lm = mini_lm.to(device)

                    with torch.set_grad_enabled(is_train):
                        mini_pred = model(mini_im)
                        loss = loss_f(mini_pred, mini_lm)

                        loss_history[k].append(loss.item())

                    if is_train:
                        loss.backward()
                        optim.step()

                    elif e == epochs - 1:
                        last_pred = mini_pred.detach().cpu()

        return loss_history, last_pred

    def train_joint_model(model, tr_dl, ts_dl, epochs, label='dl training', lr=lr):
        optim = torch.optim.Adam(model.parameters(), lr=lr)
        if args.reduce_times > 0:
            scheduler_every_n = int(epochs / (args.reduce_times+1))
            scheduler_gamma = args.reduce_factor
            scheduler = torch.optim.lr_scheduler.StepLR(optim, scheduler_every_n, scheduler_gamma)
        else:
            scheduler = None
        loss_f = MSELossNaN()

        TRAIN_KEY = 'train'
        TEST_KEY = 'test'
        keys = [TRAIN_KEY, TEST_KEY] if ts_dl is not None else [TRAIN_KEY]
        dl_dict = {k:dl for k,dl in zip(keys, [tr_dl, ts_dl])}

        loss_history = {k: [] for k in keys}

        pbar = trange(epochs, desc=label, leave=False)
        for e in pbar:
            for k in keys:
                is_train = k == TRAIN_KEY
                model.train(is_train)
                dl = dl_dict[k]

                for minibatch in dl:
                    mini_im, mini_lm, mini_ohe = flatten_and_index(*minibatch, filtern_nans=True)
                    model.zero_grad()

                    mini_im = mini_im.to(device)
                    mini_lm = mini_lm.to(device)
                    mini_ohe = mini_ohe.to(device)

                    with torch.set_grad_enabled(is_train):
                        mini_pred = model(mini_im, mini_ohe)
                        loss = loss_f(mini_pred, mini_lm)

                        loss_history[k].append(loss.item())

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
                mini_im, mini_lm, mini_ohe = flatten_and_index(*next(iter(dl_dict[TEST_KEY])),
                                                               filtern_nans=False)
                mini_im = mini_im.to(device)
                mini_ohe = mini_ohe.to(device)
                last_pred = model(mini_im, mini_ohe).detach().cpu().view((-1, len(LM_NAMES), 2))
        else:
            last_pred = None

        return loss_history, last_pred

    def split_training(yp_all, tr_idx, ts_idx, q_maxerror, cv_idx, **kwargs):
        yp_tr, yp_ts = (yp_all[idx] for idx in (tr_idx, ts_idx))
        yp = np.zeros((len(ts_idx), len(LM_NAMES))) + np.nan

        for vidx in range(len(LM_NAMES)):
            # Reset model
            reset_model(q_maxerror)

            # Set dataloader
            filtered_tr_idx, filter_mask = filter_nan_idxs(vidx, tr_idx)

            dataset_class = DATASET_CHOICES[args.dataset]
            do_jitter = 'jitter' in args.dataset
            do_augmentation = args.dataset != "all_noaug"

            tr_ds = dataset_class(landmark_idx=vidx,
                                  path_list=[os.path.join(FORENSIC_DS, x) for x in filenames[filtered_tr_idx]],
                                  center_list=yp_tr[filter_mask], ground_truth=np_dataset_pts_lst[filtered_tr_idx],
                                  correction_list=dataset_correction[filtered_tr_idx],
                                  q_maxerror=q_maxerror, window_factor=window_factor, window_outsize=window_outsize,
                                  do_cache=True, do_augmentation=do_augmentation, do_jitter=do_jitter,
                                  max_angle=args.max_angle, max_translation=args.max_translation,
                                  random_flip=args.random_flip
                                  )
            tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=dataloader_num_workers,
                               persistent_workers=True and dataloader_num_workers > 0, pin_memory=True)

            if len(ts_idx) > 0:
                ts_ds = dataset_class(landmark_idx=vidx,
                                      path_list=[os.path.join(FORENSIC_DS, x) for x in filenames[ts_idx]],
                                      center_list=yp_ts, ground_truth=np_dataset_pts_lst[ts_idx],
                                      correction_list=dataset_correction[ts_idx],
                                      q_maxerror=q_maxerror, window_factor=window_factor, window_outsize=window_outsize,
                                      do_cache=True, do_augmentation=False)
                ts_dl = DataLoader(ts_ds, batch_size=len(ts_ds), shuffle=False, drop_last=False,
                                   persistent_workers=True and dataloader_num_workers > 0,
                                   num_workers=dataloader_num_workers, pin_memory=True)
            else:
                ts_dl = None

            train_history, lm_yp_ts = train_split_model(model, tr_dl, ts_dl, epochs,
                                                        label=f'{cv_idx}: {LM_NAMES[vidx]} training')

            # Store cv prediction
            if ts_dl is not None:
                yp[:, vidx] = ts_ds.inverse_lms(lm_yp_ts)

            # Save model & history
            model_key = f'{EXPERIMENT_KEY}_lm{vidx:02d}_cv{cv_idx}'
            torch.save(model.state_dict(), os.path.join(out_path, model_key + '.pth'))
            if 'extra_info' in kwargs:
                save_M_json(kwargs['extra_info'], os.path.join(out_path, model_key + '_extra.json'))

            plot_history(train_history, epochs, title=model_key).savefig(os.path.join(out_path, model_key + '.jpg'))
            plt.close()

        return yp

    def federated_training(yp_all, tr_idx, ts_idx, q_maxerror, cv_idx, update_e, except_layers, **kwargs):
        yp_tr, yp_ts = (yp_all[idx] for idx in (tr_idx, ts_idx))
        yp = np.zeros((len(ts_idx), len(LM_NAMES), 2)) + np.nan

        except_keys = [k for k in base_sd.keys() if k.split('.')[0] in except_layers]

        reset_model(q_maxerror)
        base_sd_with_buffers = model.state_dict()
        m_weights = [deepcopy(base_sd_with_buffers) for _ in range(len(LM_NAMES))]

        dataset_class = DATASET_CHOICES[args.dataset]
        do_jitter = args.dataset == 'jitter'

        tr_ds = dataset_class(path_list=[os.path.join(FORENSIC_DS, x) for x in filenames[tr_idx]],
                              center_list=yp_tr, ground_truth=np_dataset_pts_lst[tr_idx],
                              correction_list=dataset_correction[tr_idx],
                              q_maxerror=q_maxerror, window_factor=window_factor, window_outsize=window_outsize,
                              do_cache=True, do_augmentation=True, do_jitter=do_jitter)
        tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=dataloader_num_workers,
                           persistent_workers=True and dataloader_num_workers > 0, pin_memory=True)

        if len(ts_idx) > 0:
            ts_ds = dataset_class(path_list=[os.path.join(FORENSIC_DS, x) for x in filenames[ts_idx]],
                                  center_list=yp_ts, ground_truth=np_dataset_pts_lst[ts_idx],
                                  correction_list=dataset_correction[ts_idx],
                                  q_maxerror=q_maxerror, window_factor=window_factor, window_outsize=window_outsize,
                                  do_cache=True, do_augmentation=False)
            ts_dl = DataLoader(ts_ds, batch_size=len(ts_ds), shuffle=False, drop_last=False,
                               persistent_workers=True and dataloader_num_workers > 0,
                               num_workers=dataloader_num_workers, pin_memory=True)
        else:
            ts_dl = None

        loss_f = MSELossNaN()
        TRAIN_KEY = 'train'
        TEST_KEY = 'test'
        keys = [TRAIN_KEY, TEST_KEY] if ts_dl is not None else [TRAIN_KEY]
        dl_dict = {k: dl for k, dl in zip(keys, [tr_dl, ts_dl])}

        loss_history = {i : {k: [] for k in keys} for i in range(len(LM_NAMES))}

        if args.weighted:
            count = (~np.isnan(tr_dl.dataset.ground_truth).any(-1)).sum(0)
            fed_weights = torch.tensor(count/count.sum()).float().to(device)
        else:
            fed_weights = torch.ones(len(LM_NAMES),device=device)/len(LM_NAMES)

        optims = [torch.optim.Adam(model.parameters(), lr=lr) for _ in range(len(LM_NAMES))]
        pbar = trange(epochs,desc='Federated learning', leave=False)
        for e in pbar:
            # Aggregate weights
            if e % update_e == 0:
                m_weights = aggregate_weights_mean(m_weights,except_keys,weights=fed_weights)

            # Split data per client
            dl_load = {k: [x for x in dl] for k,dl in dl_dict.items()}

            # Train each client / landmark
            for lm_idx in range(len(LM_NAMES)):
                model.load_state_dict(m_weights[lm_idx])
                # optim = torch.optim.SGD(model.parameters(), lr=lr)
                optim = optims[lm_idx]

                for k in keys:
                    dl = dl_load[k]
                    is_train = k == TRAIN_KEY
                    for mini_im, mini_lm in dl:
                        msk = (~mini_lm[:,lm_idx].isnan().any(1))
                        if msk.sum().item() == 0:
                            continue
                        mini_lm = mini_lm[msk, lm_idx].to(device)
                        mini_im = mini_im[msk, lm_idx].to(device)


                        with torch.set_grad_enabled(is_train):
                            mini_pred = model(mini_im)
                            loss = loss_f(mini_pred, mini_lm)
                            loss_history[lm_idx][k].append(loss.item())

                        if is_train:
                            loss.backward()
                            if args.clip_nan:
                                clip_nan_grad(model.parameters())
                            optim.step()

                            parameters_nan = torch.tensor([
                                e.isnan().any()
                                for e
                                in model.parameters()]
                            ).any().item()
                            output_nan = mini_pred.isnan().any().item()

                            if parameters_nan or output_nan:
                                print(parameters_nan,output_nan,e,lm_idx)
                                model_key = f'{EXPERIMENT_KEY}_lm{lm_idx:02d}_cv{cv_idx}_e{e}_NANed'
                                torch.save(model.state_dict(), os.path.join(out_path, model_key + '.pth'))
                                torch.save(dict(mini_im=mini_im, mini_lm=mini_lm),
                                           os.path.join(out_path,model_key + 'INPUT.pth'))
                                raise Exception(f'NAN during training {parameters_nan} {output_nan} {e} {lm_idx}')


                if e == (epochs-1) and TEST_KEY in keys:
                    with torch.no_grad():
                        assert(len(dl_load[TEST_KEY]) == 1)
                        mini_im = dl_load[TEST_KEY][0][0][:,lm_idx].to(device)
                        yp[:, lm_idx] = model(mini_im).detach().cpu().numpy()

                # Send weights
                m_weights[lm_idx] = deepcopy(model.state_dict())

            # Update pbar
            losses_pbar = {
                k: np.mean([e[k][-len(dl_dict[k]):] for e in loss_history.values()])
                for k
                in keys
            }
            pbar.set_postfix(**{k:losses_pbar[k] for k in keys})


        # Store cv prediction
        if TEST_KEY in keys:
            yp[:] = ts_ds.inverse_lms(yp)

        # Save model & history
        for lm_idx, sd in enumerate(m_weights):
            model_key = f'{EXPERIMENT_KEY}_lm{lm_idx:02d}_cv{cv_idx}'
            torch.save(sd, os.path.join(out_path, model_key + '.pth'))
            if 'extra_info' in kwargs:
                save_M_json(kwargs['extra_info'], os.path.join(out_path, model_key + '_extra.json'))

            plot_history(loss_history[lm_idx], epochs, title=model_key).savefig(os.path.join(out_path, model_key + '.jpg'))
            plt.close()

        return yp

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
        yp = (np.zeros((len(ts_idx), len(LM_NAMES), 2)) + np.nan) if ts_idx else None

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
                              random_flip=args.random_flip
                              )
        tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=dataloader_num_workers,
                           persistent_workers=True and dataloader_num_workers > 0, pin_memory=True)

        if ts_idx is not None:
            ts_ds = dataset_class(path_list=[os.path.join(FORENSIC_DS, x) for x in filenames[ts_idx]],
                                  center_list=yp_ts, ground_truth=np_dataset_pts_lst[ts_idx],
                                  correction_list=dataset_correction[ts_idx],
                                  q_maxerror=q_maxerror, window_factor=window_factor, window_outsize=window_outsize,
                                  do_cache=True, do_augmentation=False)
            ts_dl = DataLoader(ts_ds, batch_size=len(ts_ds), shuffle=False, drop_last=False,
                               persistent_workers=True and dataloader_num_workers > 0,
                               num_workers=dataloader_num_workers, pin_memory=True)
        else:
            ts_dl = None

        train_history, lm_yp_ts = train_joint_model(model, tr_dl, ts_dl, epochs,
                                                    label=f'{cv_idx}: JOINT training')

        # Store cv prediction
        if ts_dl is not None:
            yp[:] = ts_ds.inverse_lms(lm_yp_ts)

        # Save model & history
        model_key = f'{EXPERIMENT_KEY}_cv{cv_idx}'
        torch.save(model.state_dict(), os.path.join(out_path, model_key + '.pth'))
        if 'extra_info' in kwargs:
            save_M_json(kwargs['extra_info'], os.path.join(out_path, model_key + '_extra.json'))

        plot_history(train_history, epochs, title=model_key).savefig(os.path.join(out_path, model_key + '.jpg'))
        plt.close()
        return yp

    def get_3ddfa_prediction(fold_idx, tr_idx) -> Tuple[np.ndarray, Dict]:
        if fold_idx < 0: # all data
            assert fold_idx in M_sol_dict
        # Compute best vertices idx
        mean_dist = np.nanmean(np_dataset_dist_lst_norm[tr_idx], axis=0)
        best_idx = mean_dist.argmin(axis=0)

        # Transform all
        yp_all = np_dataset_ver_lst[:, :2, best_idx].transpose((0, 2, 1))
        for vidx in range(30):
            if vidx in M_sol_dict.get(fold_idx, {}):
                M_sol = M_sol_dict[fold_idx][vidx][0]

                target_list = np_dataset_ver_lst[:, :, [best_idx[vidx]]]
                target_list_fill = np.concatenate((target_list, np.ones((target_list.shape[0], 1, 1))),
                                                  axis=1).transpose(
                    (0, 2, 1))
                new_yp_all = np.matmul(target_list_fill, M_sol).squeeze()[:, :2]
                yp_all[:, vidx] = new_yp_all

        return yp_all, {'best_idx': best_idx}

    def get_hrnet_prediction(fold_idx, *kwargs) -> Tuple[np.ndarray, Dict]:
        if fold_idx < 0:  # all data
            assert np_dataset_hrnet_lst.shape[0] == 6
        return np_dataset_hrnet_lst[fold_idx], {}

    # Define functions
    try:
        train_func = {
            'joint': joint_training,
            'split': split_training,
            'federated': federated_training,
        }[train_method]
    except:
        raise NotImplementedError(f'Unknown training method: {train_method}')

    try:
        pred_func = {
            '3ddfa': get_3ddfa_prediction,
            'hrnet': get_hrnet_prediction,
        }[args.pre_mode]
    except:
        raise NotImplementedError(f'Unknown prediction mode: {args.pre_mode}')

    # Initialize prediction
    yp = np.zeros((len(np_dataset_pts_lst), 30, 2)) + np.nan

    for cv_idx, (tr_idx, ts_idx) in enumerate(tqdm(kfold_idxs)):
        yp_all, extra_info = pred_func(cv_idx, tr_idx)
        extra_info.update(pre_dict)
        if M_sol_dict is not None:
            extra_info.update(dict(rot={k: v[1]['x'] for (k, v) in M_sol_dict[cv_idx].items()}))

        # xy_relative_error_tr = (yp_all[tr_idx] - np_dataset_pts_lst[tr_idx]) / dataset_correction[tr_idx, None, None]
        # q_maxerror = get_maxerror(xy_relative_error_tr, error_quantile)
        q_maxerror = get_q_maxerror(
            yp_all[tr_idx], np_dataset_pts_lst[tr_idx], dataset_correction[tr_idx])

        # # Save q_maxerror
        # np.save(os.path.join(out_path, f'{EXPERIMENT_KEY}_cv{cv_idx}_q_maxerror.npy'), q_maxerror)
        # if args.only_maxerror:
        #     continue

        yp[ts_idx] = train_func(yp_all, tr_idx, ts_idx, q_maxerror, cv_idx,
                                            update_e=args.merge_every, except_layers=args.custom_layers,
                                            extra_info=extra_info)

        # if train_method == 'split':
        #     yp[ts_idx] = split_training(yp_all, tr_idx, ts_idx, q_maxerror, cv_idx)
        # elif train_method == 'joint':
        #     yp[ts_idx] = joint_training(yp_all, tr_idx, ts_idx, q_maxerror, cv_idx)
        # elif train_method == 'federated':
        #     yp[ts_idx] = federated_training(yp_all, tr_idx, ts_idx, q_maxerror, cv_idx,
        #                                     update_e=args.merge_every, except_layers=args.custom_layers)
        # else:
        #     raise NotImplementedError

    if args.train_all:
        all_idx = np.arange(len(np_dataset_pts_lst))
        cv_idx = -1
        yp_all, extra_info = pred_func(cv_idx, all_idx)
        extra_info.update(pre_dict)
        if M_sol_dict is not None:
            extra_info.update(dict(rot={k: v[1]['x'] for (k, v) in M_sol_dict[cv_idx].items()}))

        q_maxerror = get_q_maxerror(yp_all, np_dataset_pts_lst, dataset_correction)
        # np.save(os.path.join(out_path, f'{EXPERIMENT_KEY}_all_q_maxerror.npy'), q_maxerror)
        train_func(yp_all, all_idx, None, q_maxerror, 'ALL',
               update_e=args.merge_every, except_layers=args.custom_layers,
                                            extra_info=extra_info)

    # if args.only_maxerror:
    #     return

    # Save results
    np.save(os.path.join(out_path, f'{EXPERIMENT_KEY}'), yp)
    # Print error
    error_dict = {}
    for i, n in enumerate(LM_NAMES):
        error_dict[n] = mse_corrected(np_dataset_pts_lst[:, i], yp[:, i], correction=dataset_correction)
    print(pd.Series(error_dict))

    # Save config
    with open(os.path.join(out_path, 'config.json'), 'w') as ofile:
        json.dump(vars(args), ofile, indent=4)


if __name__ == '__main__':
    run()
