# %%
import os
import argparse
import logging
from utils import make_file_dir
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from qos_datasets import *
import torch
from torch.utils.data import DataLoader
import pickle
# %%
# setting up logger
logfile = "./log/LR.log"
make_file_dir(logfile)
logger = logging.getLogger("LR logger")
logger.setLevel(logging.DEBUG)
ch = logging.FileHandler(logfile)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s:%(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)
# %%
def get_parameters():
    parser = argparse.ArgumentParser(description="Linear Regression for prediction")
    parser.add_argument('--target', type=int, default=7, metavar='N',
                        help='target domain number. 0 to 7.')
    parser.add_argument("--output", type=str,
                        default='output/LR_output.csv', help="output file")
    parser.add_argument("--feature", type=str, default="all", help="feature file")
    parser.add_argument('--save_finalresult', type=int, default=1, metavar='N',
                        help='save final result or not')
    # args = parser.parse_args([])
    args = parser.parse_args()
    return args

def save_finalresult(args, test_loss):
    make_file_dir(args.output)
    with open(args.output, mode="a+") as f:
        f.write(str(args.target) + ", ")
        f.write(str(test_loss))
        f.write("\n")

args = get_parameters()
logger.info(args)
target_app = args.target
features = args.feature
data_dir = "data/dg-data"
app_list = ["cassandra", "etcd", "hbase", "kafka", "milc", "mongoDB", "rabbitmq", "redis"]

# %%
train_datas = []
train_labels = []
val_datas = []
val_labels = []
for idx, app in enumerate(app_list):
    with open(os.path.join(data_dir, "{}.pickle".format(app)), "rb") as f:
        full, train, val = pickle.load(f)
    if idx == target_app:
        x_test, y_test = sklearn_data(full)
        continue
    x_train, y_train = sklearn_data(train)
    x_val, y_val = sklearn_data(val)
    train_datas.append(x_train)
    train_labels.append(y_train)
    val_datas.append(x_val)
    val_labels.append(y_val)
    logger.info("app: {}, train shape: {}".format(app, x_train.shape))

x_train = np.concatenate(train_datas, axis=0)
y_train = np.concatenate(train_labels, axis=0)
x_val = np.concatenate(val_datas, axis=0)
y_val = np.concatenate(val_labels, axis=0)

logger.info("Total train shape: {}".format(x_train.shape))
logger.info("Total val shape: {}".format(x_val.shape))
logger.info("Total test shape: {}".format(x_test.shape))
# %%
model = LinearRegression()
reg = model.fit(x_train, y_train)
y_train_pred = model.predict(x_train)
y_val_pred = model.predict(x_val)
y_test_pred = model.predict(x_test)

train_loss_mse = mean_squared_error(y_train, y_train_pred)
val_loss_mse = mean_squared_error(y_val, y_val_pred)
test_loss_mse = mean_squared_error(y_test, y_test_pred)
train_loss_mae = mean_absolute_error(y_train, y_train_pred)
val_loss_mae = mean_absolute_error(y_val, y_val_pred)
test_loss_mae = mean_absolute_error(y_test, y_test_pred)
logger.info("Train mse loss: {:.4e}, Val mse loss: {:.4e}, Test mse loss: {:.4e}".format(train_loss_mse, val_loss_mse, test_loss_mse))
logger.info("Train mae loss: {:.4e}, Val mae loss: {:.4e}, Test mae loss: {:.4e}".format(train_loss_mae, val_loss_mae, test_loss_mae))
if args.save_finalresult:
    save_finalresult(args, test_loss_mae)