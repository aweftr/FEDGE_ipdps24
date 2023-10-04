# FEDGE without adversarial learning and STG. 
# %%
import os
import argparse
import logging
import pickle
from mymodel import *
from utils import *
from qos_datasets import *
import numpy as np
np.random.seed(0)
import torch
# set torch random seed so that the result is reproducible
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
# %%
# setting up logger
logfile = "./log/FEDGE-noD.log"
make_file_dir(logfile)
logger = logging.getLogger("FEDGE logger")
logger.setLevel(logging.DEBUG)
ch = logging.FileHandler(logfile)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s:%(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)
# %%
def get_parameters():
    parser = argparse.ArgumentParser(description="FEDGE for prediction")
    parser.add_argument('--target', type=int, default=7, metavar='N',
                        help='target domain number. 0 to 7.')
    parser.add_argument("--output", type=str,
                        default='output/FEDGE-noD_output.csv', help="output file")
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
batch_size = 100
test_batch_size = 4096
ae_lr = 0.001
dis_lr = 0.001
iterations = 2000
data_dir = "data/dg-data"
app_list = ["cassandra", "etcd", "hbase", "kafka", "milc", "mongoDB", "rabbitmq", "redis"]
model_save_path = "model/FEDGE-noD"
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
writer = SummaryWriter("runs/fedge-nod_{}".format(target_app))

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
lambda_0 = 0.5
lambda_1 = 1
lambda_2 = 0.1
app_dim = next(iter(test_loader))[0].shape[1]
noapp_dim = next(iter(test_loader))[2].shape[1]

fedge = FEDGE(app_dim, noapp_dim).to(device)

Q_optimizer = optim.Adam(fedge.Q.parameters(), lr=ae_lr)
P_optimizer = optim.Adam(fedge.P.parameters(), lr=ae_lr)
R_optimizer = optim.Adam(fedge.R.parameters(), lr=ae_lr)

# D_optimizer = optim.Adam(fedge.D.parameters(), lr=dis_lr)

MMD_loss = MMD_multidis_loss(7)

# %%
def train(model, itr):
    # def train(Q, P, R, D, train_loaders, MMD_loss, Q_optimizer, P_optimizer, R_optimizer, D_optimizer):
    # train
    model.train()
    app_data = []
    nostress_data = []
    noapp_data = []
    labels = []
    domain_labels = []
    for i in range(len(train_loaders)):
        app, nostress, noapp, label = next(iter(train_loaders[i]))
        app_data.append(app)
        nostress_data.append(nostress)
        noapp_data.append(noapp)
        labels.append(label)
        domain_label = torch.ones(label.shape[0]) * i
        domain_labels.append(domain_label)
    app_data = torch.cat(app_data, dim=0).to(device)
    nostress_data = torch.cat(nostress_data, dim=0).to(device)
    noapp_data = torch.cat(noapp_data, dim=0).to(device)
    labels = torch.cat(labels, dim=0).unsqueeze(1).to(device)
    domain_labels = torch.cat(domain_labels, dim=0).to(device)


    # reconstruction and adversarial minimization
    code = model.Q(app_data)
    re_data = model.P(code)
    # D_fake = model.D(code)

    R_input = torch.cat([code, noapp_data], dim=1)
    output = model.R(R_input)

    L_MMD = MMD_loss(code, domain_labels)
    # L_re = F.mse_loss(app_data, re_data)
    L_re = F.mse_loss(nostress_data, re_data)
    L_qos = F.mse_loss(output, labels)

    # fake_ones = torch.ones(D_fake.shape).to(device)
    # adversarial loss
    # make the generated code close to 1 to cheat discriminator
    # L_adv_min = F.mse_loss(D_fake, fake_ones)

    # total_loss = L_qos + lambda_0 * L_re + lambda_1 * L_MMD + lambda_2 * L_adv_min
    total_loss = L_qos + lambda_0 * L_re + lambda_1 * L_MMD
    # logger.info("Iteration {}, L_qos: {:.3f}, L_re: {:.3f}, L_MMD: {:.3f}, L_adv_min: {:.3f}".format(itr, L_qos.item(), L_re.item(), L_MMD.item(), L_adv_min.item()))
    logger.info("Iteration {}, L_qos: {:.3f}, L_re: {:.3f}, L_MMD: {:.3f}".format(itr, L_qos.item(), L_re.item(), L_MMD.item()))
    model.Q.zero_grad()
    model.P.zero_grad()
    model.R.zero_grad()
    total_loss.backward()
    R_optimizer.step()
    P_optimizer.step()
    Q_optimizer.step()

    # adversarial
    # code = code.detach()
    # real_prior = torch.tensor(np.random.laplace(0, np.sqrt(2)/2, code.shape)).float().to(device)
    # real_labels = torch.ones(real_prior.shape[0])
    # fake_labels = torch.zeros(code.shape[0])
    # datas = torch.cat([code, real_prior], dim=0).to(device)
    # labels = torch.cat([real_labels, fake_labels], dim=0).unsqueeze(1).to(device)

    # model.D.zero_grad()
    # D_output = model.D(datas)
    # # In the equation we want to maximize the GAN loss, 
    # # so we can multiply a -1 to minimize the loss instead. 
    # # The minimum of the loss is -1
    # L_adv_max = -F.mse_loss(D_output, labels)
    # logger.info("\tL_adv: {}".format(L_adv_max.item()))
    # writer.add_scalars("Loss", {"L_qos": L_qos.item(), "L_re": L_re.item(), "L_MMD": L_MMD.item(), "L_adv_min": L_adv_min.item(), "L_adv_max": L_adv_max.item()}, itr)
    # # writer.add_scalars("Loss", {"L_qos": L_qos.item(), "L_re": L_re.item(), "L_adv_min": L_adv_min.item(), "L_adv_max": L_adv_max.item()}, itr)
    # L_adv_max.backward()
    # D_optimizer.step()


def val(model, criterion):
    model.eval()
    total_loss = 0.
    total_count = 0
    for dataloader in val_loaders:
        for app, nostress, noapp, label in dataloader:
            app = app.to(device)
            noapp = noapp.to(device)
            labels = label.unsqueeze(1).to(device)

            code = model.Q(app)
            R_input = torch.cat([code, noapp], dim=1)
            outputs = model.R(R_input)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_count += app.shape[0]
    total_loss /= total_count
    return total_loss

def test(model, criterion):
    model.eval()
    total_loss = 0.
    total_count = 0
    for app, nostress, noapp, label in test_loader:
        app = app.to(device)
        noapp = noapp.to(device)
        labels = label.unsqueeze(1).to(device)

        code = model.Q(app)
        R_input = torch.cat([code, noapp], dim=1)
        outputs = model.R(R_input)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        total_count += app.shape[0]
    total_loss /= total_count
    return total_loss

# %%
val_criterion = nn.L1Loss(reduction='sum')
es = EarlyStopping(patience=20, path=os.path.join(model_save_path, "{}.pt".format(target_app)), trace_func=logger.info, delta=-0.001)

for itr in range(iterations):
    train(fedge, itr)
    if (itr + 1) % 10 == 0:
        val_loss = val(fedge, val_criterion)
        writer.add_scalar("val Loss", val_loss, itr)
        es(val_loss, fedge)
        if es.early_stop:
            logger.info("Early stop! Val MAE loss: {}".format(es.val_loss_min))
            break
        else:
            logger.info("\tVal MAE loss: {}".format(val_loss))

writer.close()
fedge.load_state_dict(torch.load(os.path.join(model_save_path, "{}.pt".format(target_app))))
test_loss = test(fedge, val_criterion)
logger.info("Target app: {}, MAE loss: {}".format(target_app, test_loss))
if args.save_finalresult:
    save_finalresult(args, test_loss)
