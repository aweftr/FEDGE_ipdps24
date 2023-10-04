# %%
import os
import argparse
import logging
import pandas as pd
from easydict import EasyDict
from model import LR_dataset
from utils import make_file_dir
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
# %%
# setting up logger
logfile = "./log/XGB.log"
make_file_dir(logfile)
logger = logging.getLogger("XGB logger")
logger.setLevel(logging.DEBUG)
ch = logging.FileHandler(logfile)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s:%(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)
# %%
def get_parameters():
    parser = argparse.ArgumentParser(description="XGBoost for prediction")
    parser.add_argument("--data", type=str,
                        default='data/output/pred_TPS/cassandra-TPS.csv', help="data directory")
    parser.add_argument("--output", type=str,
                        default='output/XGB_output.csv', help="output file")
    parser.add_argument("--feature_output", type=str,
                        default='output/XGB_features.csv', help="feature output file")                 
    parser.add_argument("--feature", type=str, help="feature file")
    parser.add_argument('--save_finalresult', type=int, default=1, metavar='N',
                        help='save final result or not')
    # args = parser.parse_args(args=[])
    args = parser.parse_args()
    return args

def save_finalresult(args, test_loss):
    app_qos = args.data.split("/")[-1].split(".")[0]
    make_file_dir(args.output)
    with open(args.output, mode="a+") as f:
        f.write(app_qos + ", ")
        f.write(str(test_loss))
        f.write("\n")

def save_selected_features(args, selected_features):
    app_qos = args.data.split("/")[-1].split(".")[0]
    make_file_dir(args.feature_output)
    if os.path.exists(args.feature_output):
        df = pd.read_csv(args.feature_output)
        df[app_qos] = selected_features
    else:
        df = pd.DataFrame({app_qos: selected_features})
    df.to_csv(args.feature_output, index=False)

# %%
args = EasyDict(vars(get_parameters()))
logger.info(args)
model = XGBRegressor(tree_method='gpu_hist')

lr_dataset = LR_dataset(args.data, args.feature)
x, y = lr_dataset.get_data()
logger.info("x shape: {}, y shape: {}".format(x.shape, y.shape))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

reg = model.fit(x_train, y_train)
# %%
# feature importance
sorted_idx = model.feature_importances_.argsort()
feature_names = lr_dataset.feature_names
selected_features = feature_names[sorted_idx][:20]
# print(selected_features)
# %%
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

train_loss = mean_squared_error(y_train_pred, y_train)
test_loss = mean_squared_error(y_test_pred, y_test)
logger.info("Train loss: {:.4e}, Test loss: {:.4e}".format(train_loss, test_loss))
if args.save_finalresult:
    save_finalresult(args, test_loss)
    save_selected_features(args, selected_features)
