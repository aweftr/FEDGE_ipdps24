# %%
from qos_datasets import *
import pickle
import os
import torch
torch.manual_seed(0)
from torch.utils.data import random_split
# %%
data_dir = "data/app-data"
outputdir = "data/dg-data"
# outputdir = "data/dg-data-feature"
# outputdir = "data/dg-data-online"
if not os.path.exists(outputdir):
    os.makedirs(outputdir)
app_list = ["cassandra", "etcd", "hbase", "kafka", "milc", "mongoDB", "rabbitmq", "redis"]
train_ratio = 0.8
sfeatures = "all"
# features = "data/selected_features.json"
# %%
for app in app_list:
    print("Dealing with {}".format(app))
    testdata = QosData(data_dir, app, features=sfeatures)
    noappdata = NoappData(data_dir, features=sfeatures)
    remotedataset = RemoteDataset(testdata.data_for_remote_pred, noappdata.data)
    print("Generating remote dataset with length {}".format(len(remotedataset)))
    stress = []
    nostress = []
    noapp = []
    qos = []

    for i in range(len(remotedataset)):
        stress.append(remotedataset[i][0].unsqueeze(0))
        nostress.append(remotedataset[i][1].unsqueeze(0))
        noapp.append(remotedataset[i][2].unsqueeze(0))
        qos.append(remotedataset[i][3].unsqueeze(0))
    stress = torch.concat(stress, dim=0)
    nostress = torch.concat(nostress, dim=0)
    noapp = torch.concat(noapp, dim=0)
    qos = torch.concat(qos, dim=0)
    print("stress shape: {}, noapp shape: {}".format(stress.shape, noapp.shape))

    total_len = len(remotedataset)
    train_len = int(total_len * train_ratio)
    test_len = total_len - train_len

    app_full = FullDataset(stress, nostress, noapp, qos)
    app_train, app_test = random_split(app_full, [train_len, test_len])
    print("Train len: {}, test length: {}".format(len(app_train), len(app_test)))

    outputname = os.path.join(outputdir, "{}.pickle".format(app))
    with open(outputname, "wb") as f:
        pickle.dump([app_full, app_train, app_test], f)