# %%
import os
# import nni
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
from model import DAE_MLP, Pred_dataset
from utils import merge_parameter, load_args, make_file_dir, EarlyStopping
from sklearn.model_selection import train_test_split
# %%
# setting up logger
logfile = "./log/DAE_MLP.log"
make_file_dir(logfile)
logger = logging.getLogger("DAE_MLP logger")
logger.setLevel(logging.DEBUG)
ch = logging.FileHandler(logfile)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s:%(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)
# %%
def get_parameters():
    parser = argparse.ArgumentParser(description="DAE_MLP")
    parser.add_argument("--data_dir", type=str,
                        default='data/output/pred_TPS/cassandra-TPS.csv', help="data directory")
    parser.add_argument("--config", type=str,
                        default="config/DAE_MLP_config.json", help="config filename")
    parser.add_argument('--save_model', type=int, default=0, metavar='N',
                        help='save model or not')
    parser.add_argument('--save_finalresult', type=int, default=0, metavar='N',
                        help='save final result or not')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train (default: 500)')
    parser.add_argument('--cuda', type=int, default=1, metavar='N',
                        help='use CUDA training')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    # args = parser.parse_args(args=["--log_interval", "5", "--save_model", "1"])
    args = parser.parse_args()
    return args

class DAE_MLP_trainer():
    def __init__(self, args):
        self.args = args
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.app = self.args.data_dir.split("/")[-1].split("-")[0]
        self.app_qos = self.args.data_dir.split("/")[-1].split(".")[0]
        self.DAE_model_path = args.DAE_model_path + self.app + "_DAE.pt"
        self.model = DAE_MLP(args).to(self.device)
        self.model.load_model(self.DAE_model_path, args.AE_model_path)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.save_model = args.save_model
        self.save_finalresult = self.args.save_finalresult

        self.dataset = Pred_dataset(args.data_dir)
        self.train_dataset, self.test_dataset = train_test_split(self.dataset,test_size=0.2,random_state=0)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset)


    def _train(self, epoch):
        self.model.train()
        train_loss = 0.
        for idx, (VM, SoI, labels) in enumerate(self.train_dataloader):
            VM, SoI, labels = VM.to(self.device), SoI.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(VM, SoI)
            output = torch.flatten(output)
            loss = self.criterion(output, labels)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            if idx % self.args.log_interval == 0:
                logger.info("Train Epoch {}, [{} / {}], Loss: {:.6f}".format(
                    epoch, idx * len(VM), len(self.train_dataset), loss.item()
                ))

        train_loss /= idx + 1
        logger.info("Train Avg loss: {:.4f}".format(train_loss))
        return train_loss

    def _test(self, epoch):
        self.model.eval()
        test_loss = 0.
        with torch.no_grad():
            for VM, SoI, labels in self.test_dataloader:
                VM, SoI, labels = VM.to(self.device), SoI.to(self.device), labels.to(self.device)
                output = self.model(VM, SoI)
                output = torch.flatten(output)
                loss = self.criterion(output, labels)
                test_loss += loss.item()

        test_loss /= len(self.test_dataset)
        logger.info("Test Avg loss: {:.4f}".format(test_loss))
        return test_loss

    def _save_model_state_dict(self):
        save_filename = "model/DAE_MLP/" + self.app_qos + "_DAE_MLP.pt"
        make_file_dir(save_filename)
        model_st = self.model.state_dict()
        torch.save(model_st, save_filename)

    def _save_finalresult(self, test_loss):
        savefile = "output/DAE_MLP_output.csv"
        make_file_dir(savefile)
        with open(savefile, mode="a+") as f:
            f.write(self.app_qos + ", ")
            f.write(str(test_loss))
            f.write("\n")

    def trainer(self):
        es_file = "tmp/daemlpcheckpoint.pt"
        make_file_dir(es_file)
        earlystopping = EarlyStopping(path=es_file)
        for epoch in range(self.args.epochs):
            train_loss = self._train(epoch)
            test_loss = self._test(epoch)
            # nni.report_intermediate_result(test_loss)
            earlystopping(test_loss, self.model)
            if earlystopping.early_stop:
                logger.info("Early stop! test loss: {}".format(earlystopping.val_loss_min))
                break
        self.model.load_state_dict(torch.load(es_file))

        if self.save_model:
            self._save_model_state_dict()
        if self.save_finalresult:
            self._save_finalresult(earlystopping.val_loss_min)
        # nni.report_final_result(test_loss)

# %%
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic =True

def main(args):
    logger.debug("Create DAE_MLP trainer")
    dae_mlp_t = DAE_MLP_trainer(args)
    dae_mlp_t.trainer()

# %%
# Get and update args
setup_seed(20)
cmd_args = EasyDict(vars(get_parameters()))
args = load_args(cmd_args.config)
# tuner_args = nni.get_next_parameter()
args = merge_parameter(args, cmd_args)
# args = merge_parameter(args, tuner_args)
logger.info(args)
main(args)