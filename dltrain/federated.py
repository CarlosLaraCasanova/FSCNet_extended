import torch
from copy import deepcopy

def aggregate_weights_mean(federated_weights, except_keys=[], weights=None):
    with torch.no_grad():
        # if there are nan we stop the training
        are_nan_in_fed = torch.stack([v.isnan().any() for e in federated_weights for v in e.values()]).any()
        if are_nan_in_fed:
            torch.save(federated_weights, 'nan_weights.pth')
            raise ValueError("Nan in federated weights")

        if weights is None:
            device = next(iter(federated_weights[0].values())).device
            weights = torch.ones(
                len(federated_weights),
                device=device) / len(federated_weights)

        assert (weights.size(0) == len(federated_weights))



        mean_weights = [
            torch.tensordot(weights, torch.stack(v), 1) if v[0].dtype.is_floating_point else v[0]
            for v
            in zip(*map(lambda x: x.values(), federated_weights))]
        mean_keys = [v[0] for v in zip(*map(lambda x: x.keys(), federated_weights))]

        # if there are nan we stop the training
        are_nan_in_mean = torch.stack([v.isnan().any() for v in mean_weights]).any()
        if are_nan_in_mean:
            torch.save(mean_weights, 'nan_weights.pth')
            raise ValueError("Nan in federated weights")

        mean_dict = dict(zip(mean_keys, mean_weights))
        for i, w in enumerate(federated_weights):
            for k, m in w.items():
                if k not in except_keys:
                    federated_weights[i][k] = mean_dict[k]

    return deepcopy(federated_weights)

def clip_nan_grad(parameters):
    for param in parameters:
        if param.grad is not None:
            torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)