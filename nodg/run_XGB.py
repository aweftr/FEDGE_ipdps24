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
outfile = "output/feature_all/XGB_cluster_output.csv"
featurefile = "output/feature_all/XGB_cluster_features.csv"
if os.path.exists(outfile):
    os.remove(outfile)
if os.path.exists(featurefile):
    os.remove(featurefile)
# %%
for i in total:
    print(i, flush=True)
    subprocess.run(["python", "XGB_main.py", "--data", i, "--output",
                   outfile, "--feature_output", featurefile, "--save_finalresult", "1"])
# %%
outfile = "output/feature_R/XGB_cluster_output.csv"
featurefile = "output/feature_R/XGB_cluster_features.csv"
if os.path.exists(outfile):
    os.remove(outfile)
if os.path.exists(featurefile):
    os.remove(featurefile)
# %%
for i in total:
    print(i, flush=True)
    subprocess.run(["python", "XGB_main.py", "--data", i, "--feature", "config/features_R",
                   "--output", outfile, "--feature_output", featurefile, "--save_finalresult", "1"])
