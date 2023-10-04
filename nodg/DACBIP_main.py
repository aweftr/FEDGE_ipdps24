# The former FEDGE - DACBIP(do not consider domain generalization)
# %%
import os
import random
import nni
import csv
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from easydict import EasyDict
from model import DACBIP, DACBIP_random_dataset
from utils import merge_parameter, load_args, make_file_dir, storFile
from sklearn.model_selection import train_test_split
# %%
# setting up logger
logfile = "./log/DACBIP.log"
make_file_dir(logfile)
logger = logging.getLogger("DACBIP logger")
logger.setLevel(logging.DEBUG)
ch = logging.FileHandler(logfile)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s:%(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)
# %%
def get_parameters():
    parser = argparse.ArgumentParser(description="Denoising Autoencoder")
    parser.add_argument("--data_dir", type=str,
                        default='/home/yyx/interference_prediction/dacbip/0_data.csv', help="data directory")
    # parser.add_argument("--data_dir_label", type=str,
    #                     default='/home/yyx/interference_prediction/dacbip/label.csv', help="data d2irectory")
    parser.add_argument("--config", type=str,
                        default="/home/yyx/interference_prediction/dacbip/config/DACBIP_config.json", help="config filename")
    parser.add_argument('--save_model', type=int, default=0, metavar='N',
                        help='save model or not')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--cuda', type=int, default=1, metavar='N',
                        help='use CUDA training')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args(args=["--log_interval", "5", "--save_model", "1"])
    # args - parser.parse_args()
    return args

class DACBIP_trainer():
    def __init__(self, args):
        self.args = args
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.model = DACBIP(args).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.save_model = args.save_model

        # self.train_dataset = DACBIP_random_dataset(1000, args)
        # self.train_dataloader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True)
        # self.test_dataset = DACBIP_random_dataset(500, args)
        # self.test_dataloader = DataLoader(self.test_dataset)

        self.dataset = DACBIP_random_dataset(args.data_dir)
        self.train_dataset, self.test_dataset = train_test_split(self.dataset,test_size=0.2,random_state=0)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True)
        # self.test_dataset = DACBIP_random_dataset(args.data_dir)
        self.test_dataloader = DataLoader(self.test_dataset)

    def _train(self, epoch):
        self.model.train()
        for idx, (VM, SoI, labels) in enumerate(self.train_dataloader):
            VM, SoI, labels = VM.to(self.device), SoI.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(VM, SoI)
            # print(VM)
            # print(SoI)
            # print(output)
            output = torch.flatten(output)
            # for name, parameters in self.model.named_parameters():
            #     print(name,':',parameters)
            # print(output)
            loss = self.criterion(output, labels)
            # print("loss:", loss)
            loss.backward()
            # for name, parameters in self.model:#.named_parameters():
            #     print(name,'::::',parameters.weight.grad)            
            self.optimizer.step()
            if idx % self.args.log_interval == 0:
                logger.info("Train Epoch {}, [{} / {}], Loss: {:.6f}".format(
                    epoch, idx * len(VM), len(self.train_dataset), loss.item()
                ))

    def _test(self, epoch):
        LOSS = []
        Prediction = []
        Turevalue = []
        self.model.eval()
        test_loss = 0.
        with torch.no_grad():
            for VM, SoI, labels in self.test_dataloader:
                VM, SoI, labels = VM.to(self.device), SoI.to(self.device), labels.to(self.device)
                output = self.model(VM, SoI)
                output = torch.flatten(output)
                Prediction.append(output)
                Turevalue.append(labels)
                loss = self.criterion(output, labels)
                test_loss += loss.item()
                LOSS.append(loss.item())
        test_loss /= len(self.test_dataset)

        logger.info("Test Avg loss: {:.4f}".format(test_loss))
        return test_loss, Prediction, Turevalue

    def _save_model_state_dict(self):
        save_filename = "/home/yyx/interference_prediction/dacbip/model/DACBIP.pt"
        make_file_dir(save_filename)
        model_st = self.model.state_dict()
        k = list(model_st.keys())
        for i in k:
            if "CNN" not in i and "FC" not in i:
                del model_st[i]
        torch.save(model_st, save_filename)
    
    def trainer(self):
        for epoch in range(self.args.epochs):
            self._train(epoch)
            test_loss, Prediction, Turevalue = self._test(epoch)
            nni.report_intermediate_result(test_loss)
        if self.save_model:
            self._save_model_state_dict()
        nni.report_final_result(test_loss)
        storFile(Turevalue,'/home/yyx/DAE_DACBIP1_True.csv')
        storFile(Prediction,'/home/yyx/DAE_DACBIP1_Pred.csv')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic =True

def main(args):
    logger.debug("Create DACBIP trainer")
    dacbip_t = DACBIP_trainer(args)
    dacbip_t.trainer()

# %%
# Get and update args
setup_seed(20)
cmd_args = EasyDict(vars(get_parameters()))
args = load_args(cmd_args.config)
tuner_args = nni.get_next_parameter()
args = merge_parameter(args, cmd_args)
args = merge_parameter(args, tuner_args)
logger.info(args)

# logger.debug("Create DACBIP trainer")
# dacbip_t = DACBIP_trainer(args)
# dacbip_t.trainer()
main(args)
# %%
# Test Code
