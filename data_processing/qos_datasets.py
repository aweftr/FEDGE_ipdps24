# data and dataset class for all you need: local or remote prediction
# select workload, features, qos, stress as you wish!
# %%
import os
import pandas as pd
import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split
# %%
NONE_SCALE_KEYS = ['timestamp',
                   'cpu_time',
                   'system_time',
                   'user_time',
                   'mem_util',
                   'user',
                   'nice',
                   'system',
                   'iowait',
                   'steal',
                   'idle',
                   'memused_percentage',
                   'commit_percentage']


class QosData():
    def __init__(self, data_dir, app, qos="latency", workload="all", stress_type="all", stress_intense="all", features="all") -> None:
        """The qos data with different filters

        Args:
            data_dir (str): directory of the data.
            app (str): app name.
            qos (str, optional): qos of the selected app, 'TPS' or 'latency'. Defaults to "latency".
                app-qos map: 
                {'cassandra': ['count', 'latency'],
                'etcd': ['latency', 'count'],
                'hbase': ['count', 'latency'],
                'kafka': ['count', 'latency'],
                'milc': ['speed'],
                'mongoDB': ['count', 'latency'],
                'rabbitmq': ['sent_speed', 'received_speed'],
                'redis': ['latency', 'count']}.
            workload (str, optional): workload of the selected app. Defaults to "all".
                app-workload map: 
                {'cassandra': ['wl1', 'wl10', 'wl100', 'wl150', 'wl50'], 
                'etcd': ['wl1', 'wl2', 'wl3', 'wl4', 'wl5'], 
                'hbase': ['wl1', 'wl10', 'wl100', 'wl20', 'wl50'], 
                'kafka': ['wl-1', 'wl25000', 'wl30000'], 
                'milc': ['wl3', 'wl4'], 
                'mongoDB': ['wl1', 'wl10', 'wl15', 'wl20', 'wl5'], 
                'rabbitmq': ['wl1', 'wl2', 'wl3', 'wl5'], 
                'redis': ['wl11', 'wl3', 'wl5', 'wl7', 'wl9']}.
            stress_type (str, optional): 'NET', 'L', 'MBW', 'FIO'. Defaults to "all".
            stress_intense (int, optional): stress intensity of the stress_type. Defaults to "all".
                stress type-intensity map: 
                {'NO_STRESS': [0], 
                'NET': [3, 6, 9, 12, 15], 
                'L': [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], 
                'MBW': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], 
                'FIO': [0, 1, 2, 3, 4, 5, 6, 7]}. 
            features (str, optional): 'all', 'online' or features json file location. Defaults to "all".
                'online' only contains libvirt and sar features. 
                Feature json file should contain all the selected features, must be json format. 
        """
        self.app = app
        self.qos = qos
        self.workload = workload
        self.stress_type = stress_type
        self.stress_intense = stress_intense
        self.features = features
        assert not (stress_intense != "all" and stress_type ==
                    "all"), "stress intensity depends on stress type!"
        total_df = pd.read_csv(os.path.join(data_dir, app + "-merged.csv"))

        if workload == "all":
            workload_filter = pd.Series(
                np.ones(total_df.shape[0])).astype(bool)
        else:
            workload_filter = (total_df["workload"] == self.workload)
        if stress_type == "all":
            stress_type_filter = pd.Series(
                np.ones(total_df.shape[0])).astype(bool)
        else:
            stress_type_filter = (total_df["stress_type"] == self.stress_type)
        if stress_intense == "all":
            stress_intense_filter = pd.Series(
                np.ones(total_df.shape[0])).astype(bool)
        else:
            assert type(stress_intense) == int
            stress_intense_filter = (
                total_df["stress_intensity"] == self.stress_intense)

        self.df = total_df[workload_filter &
                           stress_type_filter & stress_intense_filter]
        self._select_qos()
        self._load_feature_list()
        self.df.reset_index(drop=True, inplace=True)
        # self._check_data_imbalance()
        # if self.qos == "latency":
        # self._check_large_latency()
        self.data = self.df[self.feature_list]
        self.label = self.df["QoS"]
        if stress_type == "all":
            self.data_for_DAE = self.df[self.feature_list +
                                        ["stress_type", "workload"]]
            self.data_for_remote_pred = self.df[self.feature_list + [
                "stress_type", "stress_intensity", "workload", "QoS"]]

    def _load_feature_list(self):
        keys = list(self.df.columns)
        if self.features == "all":
            first_idx = keys.index("cpu_time")
            last_idx = keys.index("IPC")
            self.feature_list = keys[first_idx: last_idx + 1]
        elif self.features == "online":
            first_idx = keys.index("cpu_time")
            last_idx = keys.index("net_wr_packet")
            libvirt_features = keys[first_idx: last_idx + 1]
            first_idx = keys.index("tps")
            last_idx = keys.index("txkB_s")
            sar_features = keys[first_idx: last_idx + 1]
            self.feature_list = libvirt_features + sar_features
        else:
            with open(self.features, "r") as f:
                self.feature_list = json.load(f)
        assert "feature_list" in dir(self)

    def _check_data_imbalance(self):
        """Most app is balanced, so we do not need to balance the data
        """
        deg_ratio = (self.df["QoS"] < 0.95).value_counts(
        ).loc[True] / self.df.shape[0]
        if deg_ratio < 0.4:
            print("The degradation qos ratio of {} in qos: {}, workload: {}, stress_type: {}, stress_intensity: {} is too low: {:.1%}".format(
                self.app, self.qos, self.workload, self.stress_type, self.stress_intense, deg_ratio))
        #     print("You can decide whether to call imbalance_data() to balance the dataset!\n")
        # TODO imbalance_data()

    def _check_large_latency(self):
        ll_counts = (self.df["QoS"] < 0).value_counts()
        if True in ll_counts.index:
            ll_number = ll_counts.loc[True]
            print(ll_number, self.df.shape[0], self.app)
        else:
            print("No large latency", self.app)

    def _select_qos(self):
        """Select the qos according to app.
            If qos is TPS, then the smaller the qos the higher the degradation.  
            If qos is latency, then it is flipped by x = 2-x,
            which means 0.95 is the degradation threshold. 
            This makes Latency equals to TPS in terms of degradation. 
            If the degradation of latency is too high, which mean 2-x < 0, 
            we clip the negative value to 0. 
        """
        no_select = ['milc', 'rabbitmq']
        if self.qos == "TPS":
            if self.app not in no_select:
                TPS = self.df["count"]
            elif self.app == 'milc':
                TPS = self.df["speed"]
            else:
                TPS = (self.df["sent_speed"] + self.df["received_speed"]) / 2
            self.df.insert(self.df.shape[1], "QoS", TPS)
        else:
            if self.app not in no_select:
                LC = 2 - self.df["latency"]
            elif self.app == 'milc':
                LC = self.df["speed"].copy()
            else:
                LC = (self.df["sent_speed"] + self.df["received_speed"]) / 2
            LC[LC < 0] = 0
            self.df.insert(self.df.shape[1], "QoS", LC)

    def __len__(self):
        return self.df.shape[0]


class NoappData(Dataset):
    def __init__(self, data_dir, features="all") -> None:
        """Generate noapp data. 

        Args:
            data_dir (str): directory of the data.
            features (str, optional): 'all', 'online' or features json file location. Defaults to "all".
                Feature json file should be the same with QosData, must be json format. 
        """
        self.df = pd.read_csv(os.path.join(data_dir, "noapp-merged.csv"))
        self.features = features

        self._load_feature_list()
        self.data = self.df[self.feature_list +
                            ["stress_type", "stress_intensity"]]

    def _load_feature_list(self):
        keys = list(self.df.columns)
        if self.features == "all":
            first_idx = keys.index("tps")
            last_idx = keys.index("IPC")
            self.feature_list = keys[first_idx: last_idx + 1]
        elif self.features == "online":
            first_idx = keys.index("tps")
            last_idx = keys.index("txkB_s")
            self.feature_list = keys[first_idx: last_idx + 1]
        else:
            with open(self.features, "r") as f:
                self.feature_list = json.load(f)
            tmp_list = self.feature_list.copy()
            for i in tmp_list:
                if i not in keys:
                    self.feature_list.remove(i)
        assert "feature_list" in dir(self)

# class MinMaxDataset(Dataset):
#     def __init__(self, d_max, d_min) -> None:
#         """Base class for all high level dataset. Use _min_max_scale to min max scale your dataframe

#         Args:
#             d_max (pd.Series): the maximum of all your data.
#             d_min (pd.Series): the minimum of all your data.
#         """
#         self.d_max = d_max.copy()
#         self.d_min = d_min.copy()
#         for i in d_max.index:
#             if i in NONE_SCALE_KEYS:
#                 self.d_max.loc[i] = 1
#                 self.d_min.loc[i] = 0

#     def _min_max_scale(self, df):
#         df_keys = list(df.columns)
#         max_keys = list(self.d_max.index)
#         intersect = [i for i in max_keys if i in df_keys]
#         tmp_max = self.d_max.loc[intersect]
#         tmp_min = self.d_min.loc[intersect]
#         tmp_df = df[intersect].sub(tmp_min).divide(tmp_max - tmp_min)
#         divideBy0 = (tmp_max == tmp_min)
#         if divideBy0.any():
#             db0idx = divideBy0[divideBy0].index
#             tmp_df[db0idx] = 0
#         assert not tmp_df.isnull().any().any()
#         df[intersect] = tmp_df


class QosDataset(Dataset):
    def __init__(self, data, label) -> None:
        """qos dataset. 
            X is the features of current VM and PM. There is no remote PM. 
            Y is the degradation of the QoS.  

        Args:
            data (pd.DataFrame): generate by QosData class. QosData.data
            label (pd.Series): generate by QosData class. QosData.label
        """
        self.data = data.copy()
        self.label = label.copy()

    def __getitem__(self, index):
        data = torch.tensor(self.data.iloc[index]).float()
        label = torch.tensor(self.label.iloc[index]).float()
        return data, label

    def __len__(self):
        return self.data.shape[0]


class DAEDataset(Dataset):
    def __init__(self, df) -> None:
        """Dataset for DAE.
            X is the features of current VM and PM with stress. 
            Y is the features of current VM and PM without stress. 

        Args:
            df (pd.DataFrame): generate by QosData class. QosData.data_for_DAE
        """
        self.df = df.copy()
        self._generate_dataset()

    def _generate_dataset(self):
        workload = self.df["workload"].unique()
        stress_type = self.df["stress_type"].unique()
        keys = list(self.df.columns)
        end_idx = keys.index("stress_type")
        self.data = pd.DataFrame()
        self.label = pd.DataFrame()
        for i in workload:
            wl_data = self.df[self.df["workload"] == i]
            nostress_df = wl_data[wl_data["stress_type"]
                                  == "NO_STRESS"].iloc[:, : end_idx]
            stress_df = wl_data.iloc[:, : end_idx]
            nostress_df = pd.concat([nostress_df] * stress_df.shape[0], axis=0)
            self.data = pd.concat([self.data, stress_df], axis=0)
            self.label = pd.concat([self.label, nostress_df], axis=0)
        self.data.reset_index(drop=True, inplace=True)
        self.label.reset_index(drop=True, inplace=True)

    def __getitem__(self, index):
        data = torch.tensor(self.data.iloc[index]).float()
        label = torch.tensor(self.label.iloc[index]).float()
        return data, label

    def __len__(self):
        return self.data.shape[0]


class RemoteDataset(Dataset):
    def __init__(self, app_df, noapp_df):
        """Dataset for remote migration. 
            4 part is contained: stress, nostress, noapp, qos
            stress is the features of current VM and PM
            nostress is the nostress features of current VM whose workload is the same with stress
            noapp is the features of remote PM
            qos is the qos degradation of current VM when migrate it to remote PM

        Args:
            app_df (pd.DataFrame): dataframe of selected app. Generated by QosData. QosData.data_for_remote_pred
            noapp_df (pd.DataFrame): dataframe of noapp. Generated by NoappData. NoappData.data
        """
        self.app_df = app_df.copy()
        self.noapp_df = noapp_df.copy()
        self.noapp_len = self.noapp_df.shape[0]
        self._create_dataset()

    def _create_dataset(self):
        workload = self.app_df["workload"].unique()
        keys = list(self.app_df.columns)
        end_idx = keys.index("stress_type")
        self.nostress_df = pd.DataFrame()
        self.noapp_qos_df = pd.DataFrame()
        self.workload_len_list = []
        for i in workload:
            wl_df = self.app_df[self.app_df["workload"] == i]
            st_si_qos = wl_df[["stress_type", "stress_intensity", "QoS"]]
            st_si_mean_qos = st_si_qos.groupby(
                ["stress_type", "stress_intensity"]).mean()
            st_si_mean_qos.reset_index(inplace=True)
            noapp_qos_df = pd.merge(self.noapp_df, st_si_mean_qos, on=[
                                    "stress_type", "stress_intensity"])
            noapp_qos_df.drop(
                columns=["stress_type", "stress_intensity"], inplace=True)
            nostress_df = wl_df[wl_df["stress_type"]
                                == "NO_STRESS"].iloc[:, : end_idx]
            self.workload_len_list.append(wl_df.shape[0])
            self.nostress_df = pd.concat(
                [self.nostress_df, nostress_df], axis=0)
            self.noapp_qos_df = pd.concat(
                [self.noapp_qos_df, noapp_qos_df], axis=0)
        self.stress_df = self.app_df.iloc[:, : end_idx]

    def __getitem__(self, index):
        a = index
        base = 0
        for workload_idx, workload_len in enumerate(self.workload_len_list):
            b = a - workload_len * self.noapp_len
            if b < 0:
                stress_data_idx = a // self.noapp_len
                noapp_data_idx = a % self.noapp_len
                stress_data = self.stress_df.iloc[base + stress_data_idx]
                nostress_data = self.nostress_df.iloc[workload_idx]
                noapp_qos_data = self.noapp_qos_df.iloc[workload_idx *
                                                        self.noapp_len + noapp_data_idx]
                break
            else:
                a = b
                base += workload_len
                continue
        stress_data = torch.tensor(stress_data).float()
        nostress_data = torch.tensor(nostress_data).float()
        noapp_data = torch.tensor(noapp_qos_data.iloc[:-1]).float()
        qos = torch.tensor(noapp_qos_data.iloc[-1]).float()
        return stress_data, nostress_data, noapp_data, qos

    def __len__(self):
        return self.app_df.shape[0] * self.noapp_len


class FullDataset(Dataset):
    def __init__(self, stress, nostress, noapp, qos) -> None:
        self.stress = stress
        self.nostress = nostress
        self.noapp = noapp
        self.qos = qos

    def __getitem__(self, index):
        return self.stress[index], self.nostress[index], self.noapp[index], self.qos[index]

    def __len__(self):
        return self.stress.shape[0]


def local_train_test_sklearn(appdataset, ratio=0.8, seed=2):
    """split train and test data for local prediction using sklearn. 

    Args:
        appdataset (QosDataset): QosDataset
        ratio (float, optional): train ratio. Defaults to 0.7.
        seed (int, optional): random seed. Defaults to 2.

    Returns:
        x_train, y_train, x_test, y_test
    """
    train_len = int(len(appdataset) * ratio)
    test_len = len(appdataset) - train_len
    train_dataset, test_dataset = random_split(
        appdataset, [train_len, test_len], generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_dataset, batch_size=train_len)
    test_loader = DataLoader(test_dataset, batch_size=test_len)
    x_train, y_train = next(iter(train_loader))
    x_test, y_test = next(iter(test_loader))
    return x_train.numpy(), y_train.numpy(), x_test.numpy(), y_test.numpy()


def sklearn_data(dataset):
    """Generate data for sklearn

    Args:
        dataset (Dataset): train or test dataset

    Returns:
        x, y: the full matrix of the data and label
    """
    loader = DataLoader(dataset, batch_size=len(dataset))
    app_data, _, noapp_data, label = next(iter(loader))
    x = torch.cat([app_data, noapp_data], dim=1).numpy()
    y = label.numpy()
    return x, y

# %%
# # Test for QosData
# a = QosData("data/app-data", "cassandra", workload='wl1', stress_type='NET', stress_intense=3, features="test_featurelist.json")
# b = QosDataset(a.data, a.label)
# # %%
# # Test for QosData
# # Failure!
# a = QosData("data/app-data", "cassandra", workload='wl1', stress_intense=3, features="test_featurelist.json")
# # %%
# # Test for imbalance data and large latency
# app = ['cassandra',
# 'etcd',
# 'hbase',
# 'kafka',
# 'milc',
# 'mongoDB',
# 'rabbitmq',
# 'redis']
# for i in app:
#     a = QosData("data/app-data", i, qos="latency")
#     a = QosData("data/app-data", i, qos="TPS")
# # %%
# # Test for NoappData
# a = NoappData("data/app-data", features="test_featurelist.json")
# b = NoappData("data/app-data", features="all")
# # %%
# # Test for DAEDataset
# a = QosData("data/app-data", "cassandra", qos="latency")
# b = DAEDataset(a.data_for_DAE)
# # %%
# # Test for RemoteDataset
# a = QosData("data/app-data", "cassandra", qos="latency", features="test_featurelist.json")
# b = NoappData("data/app-data", features="test_featurelist.json")
# c = RemoteDataset(a.data_for_remote_pred, b.data)
# stress, nostress, noapp, qos = c[0]
# print((c[0][0] == c[1][0]).all()) # True
# print((c[0][0] == c[41][0]).all()) # False
# print((c[0][2] == c[41][2]).all()) # True
