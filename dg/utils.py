# %%
import os
import json
import csv
import numpy as np
import torch
from easydict import EasyDict

# %%
def make_file_dir(filename):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        

def load_args(filename):
    with open(filename, "r") as f:
        args = json.load(f)
    args = EasyDict(args)
    return args

def save_json(args, filename):
    with open(filename, "w") as f:
        json.dump(args, f, indent=4)

def merge_parameter(base_params, override_params):
    """
    Update the parameters in ``base_params`` with ``override_params``.
    Can be useful to override parsed command line arguments.

    Parameters
    ----------
    base_params : namespace or dict
        Base parameters. A key-value mapping.
    override_params : dict or None
        Parameters to override. Usually the parameters got from ``get_next_parameters()``.
        When it is none, nothing will happen.

    Returns
    -------
    namespace or dict
        The updated ``base_params``. Note that ``base_params`` will be updated inplace. The return value is
        only for convenience.
    """
    if override_params is None:
        return base_params
    for k, v in override_params.items():
        if k not in base_params:
            base_params[k] = v
        elif type(base_params[k]) != type(v) and base_params[k] is not None:
            raise TypeError('Expected \'%s\' in override parameters to have type \'%s\', but found \'%s\'.' %
                            (k, type(base_params[k]), type(v)))
        else:
            base_params[k] = v
    return base_params


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def MMD_2dis_loss(x, y):
    '''
    Calculate the Maximum Mean Discrepency of two data distribution x and y
    k<x,x> = e^(-0.5/sigma^2 * (xixiT - 2xixj + xjxjT))
    MMD(x, y) = mean(k<x,x> - 2k<x,y> + k<y,y>)
    '''
    device = x.device
    xx = torch.matmul(x, x.T)
    yy = torch.matmul(y, y.T)
    xy = torch.matmul(x, y.T)
    n = x.shape[0]
    a0 = 1. / (n * (n - 1))
    a1 = -2. / (n * n)

    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)

    kxx = rx.T - 2*xx + rx
    kxy = rx.T - 2*xy + ry
    kyy = ry.T - 2*yy + ry

    XX = torch.zeros(xx.shape).to(device)
    XY = torch.zeros(xx.shape).to(device)
    YY = torch.zeros(xx.shape).to(device)

    bandwidth = [1, 5, 10]
    for b in bandwidth:
        XX += torch.exp(-0.5/b* kxx)
        XY += torch.exp(-0.5/b* kxy)
        YY += torch.exp(-0.5/b* kyy)
    
    mmd = a0 * (torch.sum(XX) - torch.trace(XX) + torch.sum(YY) - torch.trace(XX)) + a1 * torch.sum(XY)
    return mmd

class MMD_multidis_loss():
    def __init__(self, n_sources) -> None:
        self.n_sources = n_sources
    
    def __call__(self, data, domain_label):
        total = 0.
        for i in range(self.n_sources):
            domain_i = data[domain_label == i]
            for j in range(i + 1, self.n_sources):
                domain_j = data[domain_label == j]
                mmd_ij = MMD_2dis_loss(domain_i, domain_j)
                total += mmd_ij
        return total
        