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
# # %%
# outfile = "output/feature_all/MLP_output.csv"
# if os.path.exists(outfile):
#     os.remove(outfile)
# model_savepath = "model/MLP/feature_all/"
# # %%
# for i in total:
#     print(i, flush=True)
#     subprocess.run(["python", "MLP_main.py", "--data", i, "--output", outfile,
#                    "--model_savepath", model_savepath, "--save_finalresult", "1", "--save_model", "1"])
# # %%
# outfile = "output/feature_R/MLP_output.csv"
# if os.path.exists(outfile):
#     os.remove(outfile)
# model_savepath = "model/MLP/feature_R/"
# # %%
# for i in total:
#     print(i, flush=True)
#     subprocess.run(["python", "MLP_main.py", "--data", i, "--output", outfile, "--model_savepath",
#                    model_savepath, "--feature", "config/features_R", "--save_finalresult", "1", "--save_model", "1"])
# %%
outfile = "output/feature_all/MLP_cluster_output.csv"
if os.path.exists(outfile):
    os.remove(outfile)
model_savepath = "model/MLP/feature_all/"
# %%
for i in total:
    print(i, flush=True)
    subprocess.run(["python", "MLP_main.py", "--data", i, "--output", outfile,
                   "--model_savepath", model_savepath, "--save_finalresult", "1", "--save_model", "1"])
# %%
outfile = "output/feature_R/MLP_cluster_output.csv"
if os.path.exists(outfile):
    os.remove(outfile)
model_savepath = "model/MLP/feature_R/"
# %%
for i in total:
    print(i, flush=True)
    subprocess.run(["python", "MLP_main.py", "--data", i, "--output", outfile, "--model_savepath",
                   model_savepath, "--feature", "config/features_R", "--save_finalresult", "1", "--save_model", "1"])