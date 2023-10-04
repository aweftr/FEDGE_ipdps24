# Select the important features according to the mu of STG. The features are saved in json format. 
# Run this file in the outer dg folder: python analyse/select_features.py
# %%
import numpy as np
import pickle
import os
import pandas as pd
import json
# %%
data_dir = "model/FEDGE-FS"
data = pd.read_csv("data/app-data/cassandra-merged.csv")
total_features = list(data.columns)
end_idx = total_features.index("IPC")
total_features = np.array(total_features[1: end_idx + 1])
# %%
flist = []
for i in range(8):
    with open(os.path.join(data_dir, "{}_mu.pt".format(i)), "rb") as f:
        mu = pickle.load(f)
    f_idx = np.argsort(mu)[::-1][:20]
    selected_features = total_features[f_idx]
    flist.append(set(selected_features))
    # flist.append(selected_features)
# %%
sfeatures = set.union(*flist)
with open("data/selected_features.json", "w") as f:
    json.dump(list(sfeatures), f, indent=True)