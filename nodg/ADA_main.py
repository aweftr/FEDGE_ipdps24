# %%
import os
import argparse
import logging
from easydict import EasyDict
from model import LR_dataset
from utils import make_file_dir
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
# %%
# setting up logger
logfile = "./log/ADA.log"
make_file_dir(logfile)
logger = logging.getLogger("ADA logger")
logger.setLevel(logging.DEBUG)
ch = logging.FileHandler(logfile)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s:%(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)
# %%
def get_parameters():
    parser = argparse.ArgumentParser(description="Adaboost-MLP for prediction")
    parser.add_argument("--data", type=str,
                        default='data/output/pred_TPS/cassandra-TPS.csv', help="data file")
    parser.add_argument("--output", type=str,
                        default='output/LR_output.csv', help="output file")
    parser.add_argument("--feature", type=str, help="feature file")
    parser.add_argument('--save_finalresult', type=int, default=1, metavar='N',
                        help='save final result or not')
    # args = parser.parse_args(args=["--log_interval", "5", "--save_model", "1"])
    args = parser.parse_args()
    return args

def save_finalresult(args, test_loss):
    app_qos = args.data.split("/")[-1].split(".")[0]
    make_file_dir(args.output)
    with open(args.output, mode="a+") as f:
        f.write(app_qos + ", ")
        f.write(str(test_loss))
        f.write("\n")
# %%
args = EasyDict(vars(get_parameters()))
logger.info(args)
lr_dataset = LR_dataset(args.data, args.feature)
x, y = lr_dataset.get_data()
logger.info("x shape: {}, y shape: {}".format(x.shape, y.shape))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

regr = AdaBoostRegressor(MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True, activation='logistic'), n_estimators=300)
# regr = AdaBoostRegressor(n_estimators=300)
# regr = XGBRegressor(tree_method='gpu_hist')
regr.fit(x_train, y_train)

y_train_pred = regr.predict(x_train)
y_test_pred = regr.predict(x_test)

train_loss = mean_squared_error(y_train_pred, y_train)
test_loss = mean_squared_error(y_test_pred, y_test)
logger.info("Train loss: {:.4e}, Test loss: {:.4e}".format(train_loss, test_loss))