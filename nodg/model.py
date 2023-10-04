# %%
import os
import torch
import torch.nn as nn
import torch.optim as optim
from easydict import EasyDict
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import load_args, save_json

# %%
class AE(nn.Module):
    def __init__(self, args, ):
        super(AE, self).__init__()
        self.encoder = nn.ModuleList()
        self.encoder.append(nn.Linear(args.input_size, args.encoder_sizes[0]))
        # self.encoder.append(nn.ReLU())
        self.encoder.append(nn.Sigmoid())
        for i in range(len(args.encoder_sizes) - 1):
            self.encoder.append(nn.Linear(args.encoder_sizes[i], args.encoder_sizes[i + 1]))
            # self.encoder.append(nn.ReLU())
            self.encoder.append(nn.Sigmoid())

        self.decoder = nn.ModuleList()
        for i in range(len(args.encoder_sizes) - 1, 0, - 1):
            self.decoder.append(nn.Linear(args.encoder_sizes[i], args.encoder_sizes[i - 1]))
            # self.decoder.append(nn.ReLU())
            self.decoder.append(nn.Sigmoid())
        self.decoder.append(nn.Linear(args.encoder_sizes[0], args.output_size))
        # self.decoder.append(nn.ReLU())
    
    def forward(self, x):
        for idx, layer in enumerate(self.encoder):
            x = layer(x)
        for idx, layer in enumerate(self.decoder):
            x = layer(x)
        return x

class pretrained_encoder(nn.Module):
    def __init__(self, args, encoder_type):
        super(pretrained_encoder, self).__init__()
        self.encoder = nn.ModuleList()
        self.encoder.append(nn.Linear(args[encoder_type + "input_size"], args[encoder_type + "encoder_sizes"][0]))
        self.encoder.append(nn.Sigmoid())
        # self.encoder.append(nn.ReLU())
        for i in range(len(args[encoder_type + "encoder_sizes"]) - 1):
            self.encoder.append(nn.Linear(args[encoder_type + "encoder_sizes"][i], args[encoder_type + "encoder_sizes"][i + 1]))
            # self.encoder.append(nn.ReLU())
            self.encoder.append(nn.Sigmoid())
    
    def forward(self, x):
        for idx, layer in enumerate(self.encoder):
            x = layer(x)
        return x

def load_encoder_state_dict(pretrained_encoder, filename):
    pretrained_encoder.load_state_dict(torch.load(filename), strict=False)
    
# class DACBIP(nn.Module):
#     def __init__(self, args):
#         super(DACBIP, self).__init__()
#         self.VM_encoder = pretrained_encoder(args, "VM_")
#         # load_encoder_state_dict(self.VM_encoder, args.VM_encoder_model)
#         self.SoI_encoder = pretrained_encoder(args, "SoI_")
#         # load_encoder_state_dict(self.SoI_encoder, args.SoI_encoder_model)

#         self.CNN = nn.ModuleList()
#         self.CNN.append(nn.Conv2d(1, args.CNN_channels[0], kernel_size=args.CNN_kernel_size, padding=1))
#         self.CNN.append(nn.ReLU())
#         for i in range(len(args.CNN_channels) - 1):
#             if args.CNN_channels[i + 1] == 0:
#                 self.CNN.append(nn.MaxPool2d(2, 2))
#             elif args.CNN_channels[i] == 0:
#                 self.CNN.append(nn.Conv2d(args.CNN_channels[i - 1], args.CNN_channels[i + 1], kernel_size=args.CNN_kernel_size, padding=1))
#                 self.CNN.append(nn.ReLU())
#             else:
#                 self.CNN.append(nn.Conv2d(args.CNN_channels[i], args.CNN_channels[i + 1], kernel_size=args.CNN_kernel_size, padding=1))
#                 self.CNN.append(nn.ReLU())
        
#         self.FC = nn.ModuleList()
#         self.FC.append(nn.Linear(16, 8))##!
#         self.FC.append(nn.ReLU())
#         self.FC.append(nn.Linear(8, args.output_size))
#         self.FC.append(nn.ReLU()
    
#     def forward(self, VM, SoI):
#         VM_features = self.VM_encoder(VM)
#         SoI_features = self.SoI_encoder(SoI)
#         # print("VM",VM_features)
#         # print("SOI",SoI_features)        
#         # VM_features = torch.unsqueeze(VM_features, 2)
#         # SoI_features = torch.unsqueeze(SoI_features, 1)
#         # print(VM_features.shape)
#         # print(SoI_features.shape)
#         x = torch.cat((VM_features, SoI_features),1)
#         # x = torch.mul(VM_features, SoI_features)
#         # print("x:", x)
#         x = torch.unsqueeze(x, 1)
#         # print("x:", x)
#         # for idx, layer in enumerate(self.CNN):
#         #     x = layer(x)
#             # print(x.shape)
#             # print(layer)
#         x = torch.flatten(x, start_dim=1)
#         # print("x:", x)
#         # print(x.shape)
#         for idx, layer in enumerate(self.FC):
#             x = layer(x)
#         # print("X:", x)
#         return x

class DAE_MLP(nn.Module):
    def __init__(self, args):
        super(DAE_MLP, self).__init__()
        self.args = args
        self.DAE_encoder = pretrained_encoder(args, "DAE_")
        self.AE_encoder = pretrained_encoder(args, "AE_")

        self.fc_input_size = args.DAE_encoder_sizes[-1] + args.AE_encoder_sizes[-1]
        # print("self.fc_input_size", self.fc_input_size)
        self.FC = nn.ModuleList()
        self.FC.append(nn.Linear(self.fc_input_size, args.fc_sizes[0]))
        # self.FC.append(nn.ReLU())
        self.FC.append(nn.Sigmoid())
        for i in range(len(args.fc_sizes) - 1):
            self.FC.append(nn.Linear(args.fc_sizes[i], args.fc_sizes[i + 1]))
            # self.FC.append(nn.ReLU())
            self.FC.append(nn.Sigmoid())
        self.FC.append(nn.Linear(args.fc_sizes[-1], args.output_size))
    
    def load_model(self, DAE_model_path, AE_model_path):
        load_encoder_state_dict(self.DAE_encoder, DAE_model_path)
        load_encoder_state_dict(self.AE_encoder, AE_model_path)
        return
        
    def forward(self, DAE_data, AE_data):
        DAE_features = self.DAE_encoder(DAE_data)
        AE_features = self.AE_encoder(AE_data)
        # print(x.shape)
        x = torch.cat((DAE_features, AE_features), 1)
        for idx, layer in enumerate(self.FC):
            x = layer(x)
        return x

class MLP(nn.Module):
    def __init__(self, args):
        super(MLP,self).__init__()
        self.fc_input_size = args.input_size

        self.FC = nn.ModuleList()
        self.FC.append(nn.Linear(self.fc_input_size, args.fc_sizes[0]))
        # self.FC.append(nn.BatchNorm1d(args.fc_sizes[0]))
        # self.FC.append(nn.ReLU())
        self.FC.append(nn.Sigmoid())
        for i in range(len(args.fc_sizes) - 1):
            self.FC.append(nn.Linear(args.fc_sizes[i], args.fc_sizes[i + 1]))
            # self.FC.append(nn.BatchNorm1d(args.fc_sizes[i + 1]))
            # self.FC.append(nn.ReLU())
            self.FC.append(nn.Sigmoid())
        self.FC.append(nn.Linear(args.fc_sizes[-1], args.output_size))
        # self.FC.append(nn.ReLU())
        # self.FC.append(nn.Sigmoid())

    def forward(self, x):
        # x = torch.cat((VM_features, SoI_features), 1)
        for idx, layer in enumerate(self.FC):
            x = layer(x)
        return x

class LR_dataset():
    def __init__(self, csv_file, feature_file=None):
        self.raw_data = pd.read_csv(csv_file)
        if feature_file:
            with open(feature_file, "r") as f:
                a = f.readlines()
                self.selected_features = [i.strip() for i in a]
            self.x = self.raw_data[self.selected_features].values
            self.feature_names = self.raw_data[self.selected_features].columns
        else:
            self.x = self.raw_data.iloc[:, :-1].values
            self.feature_names = self.raw_data.columns[:-1]
        self.y = self.raw_data.iloc[:, -1].values

    def get_data(self):
        return self.x, self.y


class MLP_dataset(Dataset):
    def __init__(self, csv_file, feature_file=None):
        self.raw_data = pd.read_csv(csv_file)
        if feature_file:
            with open(feature_file, "r") as f:
                a = f.readlines()
                self.selected_features = [i.strip() for i in a]
            self.x = torch.from_numpy(self.raw_data[self.selected_features].values).float()
        else:
            self.x = torch.from_numpy(self.raw_data.iloc[:, :-1].values).float()
        self.y = torch.from_numpy(self.raw_data.iloc[:, -1].values).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class DAE_dataset(Dataset):
    def __init__(self, csv_file):
        self.raw_data = pd.read_csv(csv_file)
        self.stress_data = self.raw_data.iloc[:, 0: 218]
        self.nostress_data = self.raw_data.iloc[:, 218: ]
        self.stress_data = torch.tensor(self.stress_data.values).float()
        self.nostress_data = torch.tensor(self.nostress_data.values).float()

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        return self.stress_data[idx], self.nostress_data[idx]

class AE_dataset(Dataset):
    def __init__(self, csv_file):
        self.raw_data = pd.read_csv(csv_file)
        self.raw_data = torch.tensor(self.raw_data.values).float()

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        return self.raw_data[idx], self.raw_data[idx]

class Pred_dataset(Dataset):
    def __init__(self, csv_file):
        self.raw_data = pd.read_csv(csv_file)
        self.DAE_data = self.raw_data.iloc[:, 0: 218]
        self.AE_data = self.raw_data.iloc[:, 218: 246]
        self.label = self.raw_data.iloc[:,-1]
        self.DAE_data = torch.tensor(self.DAE_data.values).float()
        self.AE_data = torch.tensor(self.AE_data.values).float()
        self.label = torch.tensor(self.label.values).float()
    
    def __len__(self):
        return len(self.raw_data)
    
    def __getitem__(self, idx):
        return self.DAE_data[idx], self.AE_data[idx], self.label[idx]

# %%
# Test LR
# args = load_args("/home/public/data/config/LR_config.json")
# a = LinearRegression(args)
# print(a)
# data = LR_dataset(args.data_dir)
# %%
# Test MLP
# args = load_args("/home/public/data/config/MLP_config.json")
# a = MLP(args)
# print(a)
# data = LR_dataset(args.data_dir)
# %%
# Test DAE
# args = load_args("/home/public/data/config/AE_config.json")
# a = DAE(args)
# data = pd.read_csv('/home/public/data/interference_data/output/DAE/cassandra-stress_nostress_workload_merged.csv')
# data = torch.from_numpy(data.values).float()
# print(data[:,467])
# a = AE_dataset(args.data_dir)
#dae_dataset = DAE_random_dataset('/home/yyx/interference_prediction/dacbip/cassandra-all_merged.csv')
# dae_dataset = DAE_random_dataset(100,args)
#dae_dataloader = DataLoader(dae_dataset, batch_size=args.batch_size, shuffle=True)
#print(len(dae_dataset))
# for data in dae_dataloader:
#     print(data.shape)
#     break
# %%
# Test pretrained_encoder
# args = load_args("config/DAE_MLP_config.json")
# DAE_encoder = pretrained_encoder(args, "DAE_")
# load_encoder_state_dict(DAE_encoder, args.DAE_encoder_model)
# %%
# Test DACBIP
# args = load_args("config/DACBIP_config.json")
# model = DACBIP(args)
# print(model)
# dataset = DACBIP_random_dataset(1000, args)
# dataloader = DataLoader(dataset, args.batch_size, shuffle=True)
# for idx, (VM, SoI, labels) in enumerate(dataloader):
#     print(VM.shape, SoI.shape)
#     output = model(VM, SoI)
#     print(output.shape)
#     break
# s = model.state_dict()
# print(s.keys())
# k = list(s.keys())
# for i in k:
#     if "CNN" not in i and "FC" not in i:
#         del s[i]
# print(s.keys())

# %%
# Test DAE_MLP
# args = load_args("config/DAE_MLP_config.json")
# model = DAE_MLP(args)
# print(model)
# dataset = DACBIP_random_dataset(1000, args)
# dataloader = DataLoader(dataset, args.batch_size, shuffle=True)
# for idx, (VM, SoI, labels) in enumerate(dataloader):
#     print(VM.shape, SoI.shape)
#     output = model(VM, SoI)
#     print(output.shape)
#     break
# %%
# a = torch.rand(32, 16)
# b = torch.rand(32, 16)
# c = torch.cat((a, b), 1)
# print(c.shape)
# a = torch.unsqueeze(a, 2)
# b = torch.unsqueeze(b, 1)
# print(a.shape, b.shape)
# c = torch.matmul(a, b)
# print(c.shape)
