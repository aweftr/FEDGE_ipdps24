# Drop not counted value, filter outliers, min max scale and merge data every 3 seconds. 
# %%
import os
import sys
import logging
import pandas as pd
import numpy as np
import json
# %%
DATADIR = r"data/mul"
os.chdir(DATADIR)

# %%
# setting up logger
logfile = "./get_max.log"
logger = logging.getLogger("get max logger")
logger.setLevel(logging.DEBUG)
ch = logging.FileHandler(logfile)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)
sh = logging.StreamHandler(sys.stdout)
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)
logger.addHandler(sh)

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

LOG_KEYS = ['net_rd_byte',
            'net_wr_byte',
            'net_rd_packet',
            'net_wr_packet',
            'rd_bytes',
            'wr_bytes',
            'rd_total_times',
            'wr_total_times',
            'flush_total_times']

# %%
def find_all_csv(relative_path, file_list, file_name="-1.csv"):
    for i in os.listdir(relative_path):
        file_path = os.path.join(relative_path, i)
        if os.path.isdir(file_path):
            find_all_csv(file_path, file_list, file_name)
        else:
            if i.endswith(file_name):
                file_list.append(file_path)


def filter_notcounted(df):
    value_keys = list(df.columns)
    if "stress_type" in value_keys:
        idx = value_keys.index("stress_type")
        value_keys = value_keys[: idx]
    if "timestamp" in value_keys:
        idx = value_keys.index("timestamp") + 1
        value_keys = value_keys[idx:]
    filter = (df != "<not")
    df_counted = df[filter]
    df_counted.dropna(axis=0, inplace=True)
    value_astype = {i: "float64" for i in value_keys}
    df_counted = df_counted.astype(value_astype)
    logger.info("After deleting not counted shape: {}".format(df_counted.shape))
    return df_counted


def add_columns(df_counted):
    # Add mem_util
    df_counted_add = df_counted.copy()
    logger.info("Add VM mem util, CPI, RCPI, MPKI")
    mem_util = (df_counted_add["available"] - df_counted_add["unused"]) / df_counted_add["available"]
    df_counted_add.drop(labels=["actual", "available", "unused"], axis=1, inplace=True)
    idx = df_counted_add.columns.get_loc("user_time") + 1
    df_counted_add.insert(idx, "mem_util", mem_util)
    # Add VM CPI, RCPI, MPKI
    VM_CPI = df_counted_add["UNHALTED_CORE_CYCLES"] / df_counted_add["INSTRUCTION_RETIRED"]
    VM_CPI[VM_CPI >= 30] = 30
    VM_RCPI = df_counted_add["UNHALTED_REFERENCE_CYCLES"] / df_counted_add["INSTRUCTION_RETIRED"]
    VM_RCPI[VM_RCPI >= 30] = 30
    VM_MPKI_LLC = df_counted_add["LLC_MISSES"] * 1000.0 / df_counted_add["INSTRUCTION_RETIRED"]
    VM_MPKI_LLC[VM_MPKI_LLC >= 30] = 30
    VM_MPKI_L2 = df_counted_add["L2_RQSTS:MISS"] * 1000.0 / df_counted_add["INSTRUCTION_RETIRED"]
    VM_MPKI_L2[VM_MPKI_L2 >= 300] = 300
    VM_MPKI_L1D = df_counted_add["MEM_LOAD_RETIRED:L1_MISS"] * 1000.0 / df_counted_add["INSTRUCTION_RETIRED"]
    VM_MPKI_L1D[VM_MPKI_L1D >= 300] = 300

    idx = df_counted_add.columns.get_loc("MEM_LOAD_RETIRED:L1_HIT")
    df_counted_add.insert(idx, "VM_CPI", VM_CPI)
    df_counted_add.insert(idx, "VM_RCPI", VM_RCPI)
    df_counted_add.insert(idx, "VM_MPKI_LLC", VM_MPKI_LLC)
    df_counted_add.insert(idx, "VM_MPKI_L2", VM_MPKI_L2)
    df_counted_add.insert(idx, "VM_MPKI_L1D", VM_MPKI_L1D)
    df_counted_add.dropna(axis=0, inplace=True)
    return df_counted_add

def get_max_without_outliers(df_counted_add):
    value_keys = list(df_counted_add.columns)
    max_total = {}
    for i in value_keys:
        if i in NONE_SCALE_KEYS:
            continue
        max_total[i] = {}
        max_total[i]["outliers"] = []
        col = df_counted_add[i].copy()
        while True:
            max = col.max()
            Q3 = col.quantile(0.75)
            if max <= Q3 * 1e5 or max <= 1e10:
                break
            else:
                max_total[i]["outliers"].append(max)
                col[col.idxmax()] = 0
        max_total[i]["max"] = col.max()
        max_total[i]["min"] = col.min()
        if max_total[i]["max"] > 1e13:
            logger.info("Warning!! {} have a very large value {:e}".format(i, max_total[i]["max"]))
    return max_total

def process_none_scale(df):
    df_proc = df.copy()
    for keys in NONE_SCALE_KEYS:
        if '_time' in keys:
            df_proc[keys] = df_proc[keys]/400.
    return df_proc

def get_stress_index(df, appname):
    stress_index = {}
    stress_type = list(df["stress_type"].drop_duplicates())
    for i in stress_type:
        stress_index[i] = {}
        stress_intensity = list(df[df["stress_type"] == i]["stress_intensity"].drop_duplicates())
        # print(stress_type, stress_intensity)
        for j in stress_intensity:
            stress_index[i][j] = df[(df["stress_type"] == i) & (df["stress_intensity"] == j)].index
    # Only consider bottom no_stress index
    idx = stress_index["NO_STRESS"][0]
    i = len(idx) - 1
    while True:
        if idx[i - 1] != idx[i] - 1:
            break
        else:
            i -= 1
    if appname in ['milc', 'noapp']:
        stress_index["NO_STRESS"][0] = idx[i:]
        return stress_index
    # Use quantile to filter irregular qos value
    # milc cannot be filtered because of its large variance
    idx = idx[i:]
    QoS_label = ["count", "latency", "tps.1", "sent_speed", "received_speed"]
    qos_filter = None
    for qos_label in QoS_label:
        if qos_label in df:
            qos = df.iloc[idx][qos_label]
            q1 = qos.quantile(0.2)
            q3 = qos.quantile(0.8)
            irq = (q3 - q1) * 2
            inf = q1 - irq
            sup = q3 + irq
            if qos_filter is None:
                qos_filter = (qos > inf) & (qos < sup)
            else:
                qos_filter &= ((qos > inf) & (qos < sup))
    
    assert qos_filter is not None
    stress_index["NO_STRESS"][0] = qos[qos_filter].index
    return stress_index


def milc_drop_qos(df):
    df_drop = df[df["speed"] < 1.5].copy()
    return df_drop

# %%
csv_file_list = []
find_all_csv("./", csv_file_list)

# %%
# Read all files
df = pd.DataFrame()
for i in csv_file_list:
    if "noapp" in i:
        continue
    print("Load {}".format(i))
    tmp = pd.read_csv(i)
    keys = list(tmp.columns)
    idx = keys.index("IPC") + 1
    tmp = tmp[keys[1 : idx]]
    df = pd.concat([df, tmp])
    # if (os.path.basename(i).split('-')[0]=='noapp'):
    #     print(df)
    #     break
print("Total df shape: {}".format(df.shape))

# %%
# Get the max of all files
df_counted = filter_notcounted(df)
df_counted_add = add_columns(df_counted)
# %%
max_total = get_max_without_outliers(df_counted_add)
with open("total_max.json", "w") as f:
    json.dump(max_total, f, indent=4)
# %%
drop_columns = []
for i in df_counted_add:
    if i in max_total:
        if (max_total[i]["max"] - max_total[i]["min"]) < 1e-8:
            drop_columns.append(i)
    if i in NONE_SCALE_KEYS:
        k_max = df_counted_add[i].max()
        k_min = df_counted_add[i].min()
        if (k_max - k_min) < 1e-8:
            drop_columns.append(i)
# %%
for f in csv_file_list:
    print("Dealing with {}".format(f))
    df = pd.read_csv(f)
    dirname = os.path.dirname(f)
    appname = os.path.basename(f).split('-')[0]
    outname = appname + "-outlier-3merge.csv"
    noapp = (os.path.basename(f).split('-')[0]=='noapp')
    #print(dirname, outname)
    df = filter_notcounted(df)
    
    if not noapp:
        df = add_columns(df)
        df = process_none_scale(df)
        df.drop(drop_columns, axis=1, inplace=True)
    else:
        for i in drop_columns:
            if i in df:
                df.drop(i, axis=1, inplace=True)
    if appname == "milc":
        df = milc_drop_qos(df)
            
    keys = list(df.columns)
    df_length = len(df)
    # filter max outliers 
    # set outliers to the resonable max value
    for i in keys:
        if i not in max_total:
            continue
        tmp = df[i].copy()
        # tmp[tmp > max_total[i]["max"]] = max_total[i]["max"]
        tmp = (df[i] - max_total[i]["min"]) / (max_total[i]["max"] - max_total[i]["min"])
        # tmp = df[i] / max_total[i]["max"] # min is assumed to be 0
        tmp[tmp >= 1] = 1.0
        df[i] = tmp
        # if i in LOG_KEYS:
            # df[i > 1] = np.log(df[df[i > 1]])
            # df[i] = np.log(tmp)
    
    df.reset_index(drop=True, inplace=True)
    keys = list(df.columns)
    df_out = pd.DataFrame()
    STEP = 3
    stress_index = get_stress_index(df, appname)
    if noapp:
        STEP=30
    else:
        start = keys.index("IPC") + 1
        end = keys.index("stress_type")
        QoS = keys[start: end]
        QoS_nostress = df.loc[stress_index["NO_STRESS"][0]][QoS].mean()
    
    # Merge data point every 3 seconds
    for si in stress_index:
        for j in stress_index[si]:
            k = 0
            while k < len(stress_index[si][j]):
                tmp = df.iloc[stress_index[si][j][k : k + STEP], 1: -2].mean()
                df_tmp = tmp.to_frame().T
                df_tmp.insert(0, "timestamp", df.loc[stress_index[si][j][k]]["timestamp"])
                df_tmp.insert(df_tmp.shape[1], "stress_type", df.loc[stress_index[si][j][k]]["stress_type"])
                df_tmp.insert(df_tmp.shape[1], "stress_intensity", df.loc[stress_index[si][j][k]]["stress_intensity"])
                if not noapp:
                    # Convert QoS to degration
                    df_tmp[QoS] = df_tmp[QoS] / QoS_nostress
                k += STEP
                df_out = pd.concat([df_out, df_tmp])
    df_out.to_csv(os.path.join(dirname, outname), index=False)
# %%
# get stress type - intensity map
# stress_type_int_dict = {}
# stress_type = df["stress_type"].unique()
# for i in stress_type:
#     stress_type_int_dict[i] =  list(df[df["stress_type"] == i]["stress_intensity"].unique())
