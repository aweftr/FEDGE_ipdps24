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