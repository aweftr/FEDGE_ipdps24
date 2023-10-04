# %%
import os
import nni
import csv
import random
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from easydict import EasyDict
from model import LinearRegression, LRG_random_dataset, CART_random_dataset
from utils import merge_parameter, load_args, make_file_dir, storFile
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import tree
from sklearn import metrics
# %%
# setting up logger
logfile = "./log/CART.log"
make_file_dir(logfile)
logger = logging.getLogger("CART logger")
logger.setLevel(logging.DEBUG)
ch = logging.FileHandler(logfile)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s:%(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)
# %%

DIR = "/home/yyx/interference_prediction/dacbip/0_data.csv"
raw_data = pd.read_csv(DIR)
raw_data = np.array(raw_data.values, dtype = 'float32')

CART_raw_data1 = raw_data[:,0:219]
CART_raw_data2 = raw_data[:,439:468] 
x = np.concatenate((CART_raw_data1,CART_raw_data2), 1)     
label = raw_data[:,-1] + 1
train_dataset, test_dataset, train_label, test_label = train_test_split(x,label,test_size=0.2,random_state=1)
clf = tree.DecisionTreeRegressor()
clf = clf.fit(train_dataset, train_label)
predict = clf.predict(test_dataset)
print(metrics.mean_squared_error(predict, test_label))


# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic =True



# %%
# Get and update args
# setup_seed(20)
# cmd_args = EasyDict(vars(get_parameters()))
# args = load_args(cmd_args.config)
# tuner_args = nni.get_next_parameter()
# args = merge_parameter(args, cmd_args)
# args = merge_parameter(args, tuner_args)

# logger.info(args)