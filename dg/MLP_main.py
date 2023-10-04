# %%
import os
import argparse
import logging
import pickle
import torch
# set torch random seed so that the result is reproducible
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from mymodel import MLP
from utils import *
from qos_datasets import *
# %%
# setting up logger
logfile = "./log/MLP.log"
make_file_dir(logfile)
logger = logging.getLogger("MLP logger")
logger.setLevel(logging.DEBUG)
ch = logging.FileHandler(logfile)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s:%(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)
# %%
def get_parameters():
    parser = argparse.ArgumentParser(description="Multilayer perceptron for prediction")
    parser.add_argument('--target', type=int, default=7, metavar='N',
                        help='target domain number. 0 to 7.')
    parser.add_argument("--output", type=str,
                        default='output/MLP_output.csv', help="output file")
    parser.add_argument("--feature", type=str, default="all", help="feature file")
    parser.add_argument('--save_finalresult', type=int, default=1, metavar='N',
                        help='save final result or not')
    # args = parser.parse_args([])
    args = parser.parse_args()
    return args


args = get_parameters()
logger.info(args)
target_app = args.target
features = args.feature
batch_size = 100
test_batch_size = 4196
lr = 0.001
iterations = 2000
data_dir = "data/dg-data"
app_list = ["cassandra", "etcd", "hbase", "kafka", "milc", "mongoDB", "rabbitmq", "redis"]
model_save_path = "model/MLP"
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
# generate remote DG dataset for torch
train_loaders = []
val_loaders = []
for idx, app in enumerate(app_list):
    with open(os.path.join(data_dir, "{}.pickle".format(app)), "rb") as f:
        full, train, val = pickle.load(f)
    if idx == target_app:
        test_loader = DataLoader(full, batch_size=test_batch_size)
        continue
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=test_batch_size)
    train_loaders.append(train_loader)
    val_loaders.append(val_loader)

# %%
def train(train_loaders, model, criterion, optimizer, iteration):
    model.train()
    datas = []
    labels = []
    for i in range(len(train_loaders)):
        # Shuffle is True, so the data of different epoch is random generated. 
        app, nostress, noapp, label = next(iter(train_loaders[i]))
        data = torch.cat([app, noapp], dim=1)
        datas.append(data)
        labels.append(label)
    datas = torch.cat(datas, dim=0).to(device)
    labels = torch.cat(labels, dim=0).unsqueeze(1).to(device)
    outputs = model(datas)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    logger.info("Iteration {}, train MSE loss {:.6f}".format(iteration, loss.item()))
    return loss.item()

def val(val_loaders, model, criterion):
    model.eval()
    total_count = 0
    total_loss = 0.
    for i in range(len(val_loaders)):
        for app, nostress, noapp, label in val_loaders[i]:
            data = torch.cat([app, noapp], dim=1).to(device)
            label = label.unsqueeze(1).to(device)
            outputs = model(data)
            loss = criterion(outputs, label)
            total_loss += loss.item() * data.shape[0]
            total_count += data.shape[0]
    total_loss /= total_count
    return total_loss

def test(test_loader, model, criterion):
    model.eval()
    total_count = 0
    total_loss = 0.
    for app, nostress, noapp, label in test_loader:
        data = torch.cat([app, noapp], dim=1).to(device)
        label = label.unsqueeze(1).to(device)
        outputs = model(data)
        loss = criterion(outputs, label)
        total_loss += loss.item() * data.shape[0]
        total_count += data.shape[0]
    total_loss /= total_count
    return total_loss

def save_finalresult(args, test_loss):
    make_file_dir(args.output)
    with open(args.output, mode="a+") as f:
        f.write(str(args.target) + ", ")
        f.write(str(test_loss))
        f.write("\n")

# %%
app_dim = next(iter(test_loader))[0].shape[1]
noapp_dim = next(iter(test_loader))[2].shape[1]

model = MLP(app_dim + noapp_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
es = EarlyStopping(patience=20, path=os.path.join(model_save_path, "{}.pt".format(target_app)), trace_func=logger.info, delta=-0.001)

for itr in range(iterations):
    train(train_loaders, model, criterion, optimizer, itr)
    if (itr + 1) % 10 == 0:
        val_loss = val(val_loaders, model, F.l1_loss)
        es(val_loss, model)
        if es.early_stop:
            logger.info("Early stop! Val MAE loss: {}".format(es.val_loss_min))
            break
        else:
            logger.info("\tVal MAE loss: {}".format(es.val_loss_min))
    # logger.info("\tVal MAE loss: {}".format(val_loss))

model.load_state_dict(torch.load(os.path.join(model_save_path, "{}.pt".format(target_app))))
test_loss = test(test_loader, model, F.l1_loss)
logger.info("Target app: {}, MAE loss: {}".format(target_app, test_loss))
if args.save_finalresult:
    save_finalresult(args, test_loss)