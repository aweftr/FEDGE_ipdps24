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
from model import MLP, MLP_dataset
from utils import merge_parameter, load_args, make_file_dir, EarlyStopping
from sklearn.model_selection import train_test_split
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
    parser.add_argument("--data", type=str,
                        default='data/output/pred_TPS/cassandra-TPS.csv', help="data directory")
    parser.add_argument("--output", type=str,
                        default='output/MLP_output.csv', help="output file")
    parser.add_argument("--feature", type=str, help="feature file")
    parser.add_argument("--config", type=str,
                        default="config/MLP_config.json", help="config filename")
    parser.add_argument('--save_model', type=int, default=0, metavar='N',
                        help='save model or not')
    parser.add_argument("--model_savepath", type=str,
                        default="model/MLP/", help="model save path")
    parser.add_argument('--save_finalresult', type=int, default=0, metavar='N',
                        help='save final result or not')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--cuda', type=int, default=1, metavar='N',
                        help='use CUDA training')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    # args = parser.parse_args(args=["--log_interval", "5", "--save_model", "1"])
    args = parser.parse_args()
    return args


class MLP_trainer():
    def __init__(self, args):
        self.args = args
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.dataset = MLP_dataset(args.data, args.feature)
        self.train_dataset, self.test_dataset = train_test_split(self.dataset,test_size=0.2,random_state=0)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset)
        self.args.input_size = self.dataset.x.shape[1]
        logger.info("input size: {}".format(self.args.input_size))

        self.model = MLP(args).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.save_model = args.save_model
        self.save_finalresult = self.args.save_finalresult


    def _train(self, epoch):
        self.model.train()
        train_loss = 0.
        for idx, (data ,label) in enumerate(self.train_dataloader):
            data = data.to(self.device)
            label = label.to(self.device).unsqueeze(1)
            self.optimizer.zero_grad()
            output = self.model(data)

            loss = self.criterion(output, label)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            if idx % self.args.log_interval == 0:
                logger.info("Train Epoch {}, [{} / {}], Loss: {:.6f}".format(
                    epoch, idx * len(data), len(self.train_dataset), loss.item()
                ))
        train_loss /= idx + 1
        logger.info("Train Avg loss: {:.4e}".format(train_loss))
        return train_loss
    
    def _test(self, epoch):
        self.model.eval()
        test_loss = 0.
        with torch.no_grad():
            for data, label in self.test_dataloader:
                data = data.to(self.device)
                label = label.to(self.device).unsqueeze(1)
                output = self.model(data)
                loss = self.criterion(output, label)
                test_loss += loss.item()
        test_loss /= len(self.test_dataset)
        logger.info("Test Avg loss: {:.4e}".format(test_loss))
        return test_loss

    def _save_model_state_dict(self):
        app_qos = self.args.data.split("/")[-1].split(".")[0]
        save_filename = self.args.model_savepath + app_qos + "_MLP.pt"
        make_file_dir(save_filename)
        torch.save(self.model.state_dict(), save_filename)

    def _save_finalresult(self, test_loss):
        app_qos = self.args.data.split("/")[-1].split(".")[0]
        make_file_dir(args.output)
        with open(args.output, mode="a+") as f:
            f.write(app_qos + ", ")
            f.write(str(test_loss))
            f.write("\n")

    def trainer(self):
        es_file = "tmp/mlpcheckpoint.pt"
        make_file_dir(es_file)
        earlystopping = EarlyStopping(patience=5, path=es_file)
        for epoch in range(self.args.epochs):
            self._train(epoch)
            test_loss= self._test(epoch)
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
    logger.debug("Create MLP trainer")
    mlp = MLP_trainer(args)
    mlp.trainer()

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