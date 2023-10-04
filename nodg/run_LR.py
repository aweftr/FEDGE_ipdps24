# %%
import os
import subprocess

# %%


def find_all_csv(relative_path, file_list, file_name="-1.csv"):
    for i in os.listdir(relative_path):
        file_path = os.path.join(relative_path, i)
        if os.path.isdir(file_path):
            find_all_csv(file_path, file_list, file_name)
        else:
            if i.endswith(file_name):
                file_list.append(file_path)


# %%
# total = []
# LC = []
# TPS = []
# find_all_csv("data/output/pred_LC", LC, ".csv")
# find_all_csv("data/output/pred_TPS", TPS, ".csv")
# total.extend(LC)
# total.extend(TPS)
# %%
total = []
find_all_csv("data/output/pred_cluster", total, ".csv")
# %%
outfile = "output/feature_all/LR_cluster_output.csv"
if os.path.exists(outfile):
    os.remove(outfile)
# %%
for i in total:
    print(i, flush=True)
    subprocess.run(["python", "LR_main.py", "--data", i,
                   "--output", outfile, "--save_finalresult", "1"])
# %%
fs_outfile = "output/feature_R/LR_cluster_output.csv"
if os.path.exists(fs_outfile):
    os.remove(fs_outfile)
# %%
for i in total:
    print(i, flush=True)
    subprocess.run(["python", "LR_main.py", "--data", i, "--output", fs_outfile,
                   "--feature", "config/features_R", "--save_finalresult", "1"])
