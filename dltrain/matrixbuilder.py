import numpy as np
from copy import deepcopy
from scipy import optimize
from tqdm.auto import tqdm, trange
from dltrain.utils import mse_corrected, filter_nan, t2_test, LM_NAMES, save_M_json
from dltrain.matrixbuilder import MatrixBuilder
from dltrain.arguments import parse_arguments, read_pts_lst, read_boxes, read_ver_and_dist, read_kfold


class MatrixBuilder:
    # ver_lst: [batch_size, 3:(x,y,z), 4: ref_points]
    # v_ref: [3: (x,y,z), 4: ref_points]
    def __init__(self, ver_lst, v_ref):
        batch_size = ver_lst.shape[0]
        a = np.concatenate((ver_lst, np.ones((batch_size, 1, 4))), axis=1).transpose((0, 2, 1))
        b = v_ref.T
        x = np.linalg.solve(a, b[None, :])
        fill = np.repeat(np.array([0, 0, 0, 1])[None, :, None], batch_size, axis=0)
        x_fill = np.concatenate((x, fill), axis=-1)
        x_inv = np.linalg.inv(x_fill)[:, :, :3]

        self.x_fill = x_fill
        self.x_inv = x_inv

    # v: [12,]
    def __call__(self, v):
        M = np.concatenate((v, [0, 0, 0, 1])).reshape(4, 4)
        M_norm = np.matmul(
            np.matmul(
                self.x_fill,
                M),
            self.x_inv)
        return M_norm

def eucl_dist(y, yp, correction=None):
    correction = np.ones(y.shape[1]) if correction is None else correction
    return np.nanmean(np.sqrt((((y - yp)) ** 2).sum(1)) / correction)

# Fitness function
class CostFunction:
    def __init__(self, build_matrix, np_dataset_pts_lst, dataset_correction, target_list_fill, vidx, tr_idx):
        self.build_matrix = build_matrix
        self.np_dataset_pts_lst = np_dataset_pts_lst[tr_idx][:,vidx]
        self.dataset_correction = dataset_correction[tr_idx]
        self.target_list_fill = target_list_fill[tr_idx]
        self.tr_idx = tr_idx

    def __call__(self, v):
        # Build matrix
        M = self.build_matrix(v)[self.tr_idx]
        yp = np.matmul(self.target_list_fill, M).squeeze()[:, :2]
        e = mse_corrected(yp, self.np_dataset_pts_lst, correction=self.dataset_correction)
        return e

def optimize_matrix(args, v_list=None):
    if v_list is None:
        v_list = [31517, 31106, 31980, 31509]

    _ , np_dataset_pts_lst = read_pts_lst(args)
    np_dataset_boxes, dataset_correction = read_boxes(args)
    np_dataset_ver_lst, np_dataset_dist_lst_norm = read_ver_and_dist(args)
    kfold_idxs = read_kfold(args)

    v_ref = np_dataset_ver_lst[:, :, v_list].mean(0)
    build_matrix = MatrixBuilder(np_dataset_ver_lst[:, :, v_list], v_ref)

    def optimize_fold(tr_idx, ts_idx):
        d = {}

        mean_dist = np.nanmean(np_dataset_dist_lst_norm[tr_idx], axis=0)
        best_idx = mean_dist.argmin(axis=0)

        yp_tr = deepcopy(np_dataset_ver_lst[tr_idx][:, :2, best_idx].transpose((0, 2, 1)))
        yp_ts = deepcopy(np_dataset_ver_lst[ts_idx][:, :2, best_idx].transpose((0, 2, 1)))

        # Replace misaligned
        # xy_relative_error = (np_dataset_ver_lst[:, :2, best_idx].transpose((0, 2, 1))
        #                      - np_dataset_pts_lst) / dataset_correction[:, None, None]
        for vidx in trange(30, desc='Optimization', leave=False):
            # Xpop = filter_nan(xy_relative_error[tr_idx, vidx], 1)
            # p_value = t2_test(Xpop)

            # !!! Precomputed: only vertex and trichion
            # if p_value < ALPHA:
            if vidx in np.isin(LM_NAMES, ['Vertex', 'Trichion']).nonzero()[0]:
                # print(LM_NAMES[vidx], p_value)
                # Precompute target
                target_list = np_dataset_ver_lst[:, :, [best_idx[vidx]]]
                target_list_fill = np.concatenate((target_list,
                                                   np.ones((target_list.shape[0], 1, 1))), axis=1).transpose((0, 2, 1))
                foo = CostFunction(build_matrix, np_dataset_pts_lst, dataset_correction, target_list_fill, vidx, tr_idx)

                # Transformation matrix
                sol = optimize.differential_evolution(
                    foo, [(-5, +5)] * 12,
                    workers=24, maxiter=args.opt_maxiter, updating='deferred')
                print(sol.message)

                M_sol = build_matrix(sol.x)
                # if lower train error
                old_error_norm = eucl_dist(yp_tr[:, vidx], np_dataset_pts_lst[tr_idx][:, vidx],
                                           correction=dataset_correction[tr_idx])
                new_yp_tr = np.matmul(target_list_fill[tr_idx], M_sol[tr_idx]).squeeze()[:, :2]
                new_error_norm = eucl_dist(new_yp_tr, np_dataset_pts_lst[tr_idx][:, vidx],
                                           correction=dataset_correction[tr_idx])
                if new_error_norm < old_error_norm:
                    print('better', new_error_norm, old_error_norm)
                    d[vidx] = (M_sol, sol)

                    if len(ts_idx) > 0:
                        # predict test again
                        new_yp_ts = np.matmul(target_list_fill[ts_idx], M_sol[ts_idx]).squeeze()[:, :2]
                        yp_ts[:, vidx] = new_yp_ts

        return d, yp_ts

    M_sol_dict = {}
    yp = np.zeros((165, 30, 2))
    for cv_idx, (tr_idx, ts_idx) in enumerate(tqdm(kfold_idxs)):
        M_sol_dict[cv_idx], yp[ts_idx] = optimize_fold(tr_idx, ts_idx)
    M_sol_dict[-1], _ = optimize_fold(np.arange(np_dataset_pts_lst.shape[0]), [])

    # Save & return
    save_M_json(M_sol_dict, args.opt_mat)
    return yp

if __name__ == '__main__':
    args = parse_arguments()
    optimize_matrix(args)