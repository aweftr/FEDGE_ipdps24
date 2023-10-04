# Merge the output of different methods into one csv file and calculate the average of the output. 
# Run this file in the outer dg folder: python analyse/analyse_output.py
# %%
import os
import pandas as pd

# %%
app_list = ["cassandra", "etcd", "hbase", "kafka", "milc", "mongoDB", "rabbitmq", "redis"]
app_series = pd.Series(app_list)
total_df = pd.DataFrame()
for i in os.listdir("output"):
    if i.endswith(".csv"):
        method = i.split('_')[0]
        df = pd.read_csv("output/{}".format(i), header=None)
        method_series = df.sort_values(0)[1]
        method_series.reset_index(drop=True, inplace=True)
        method_series.rename(method, inplace=True)
        total_df = pd.concat([total_df, method_series], axis=1)

# %%
total_df.set_index(app_series, inplace=True)
tmp = total_df.mean()
tmp = tmp.rename("Average")
total_df = pd.concat([total_df, tmp.to_frame().T], axis=0)
# total_df = total_df.append(tmp)

# %%
total_df.to_csv("output/all.csv")
# %%
# df = pd.read_csv("../output/{}".format("FEDGE_output.csv"), header=None)