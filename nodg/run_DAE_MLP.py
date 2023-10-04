# %%
import os
import sys
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

def check_remove(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
# %%
# train file
total = []
LC = []
TPS = []
find_all_csv("data/output/pred_LC", LC, ".csv")
find_all_csv("data/output/pred_TPS", TPS, ".csv")
total.extend(LC)
total.extend(TPS)
# %%
# dae file
DAE = []
find_all_csv("data/output/DAE", DAE, ".csv")
# %%
app = []
for i in total:
    appname = i.split("/")[-1].split("-")[0]
    if appname not in app:
        app.append(appname)
# %%
check_remove("output/DAE_output.csv")
check_remove("output/AE_output.csv")
check_remove("output/DAE_MLP_output.csv")
# %%
try:
    for i in app:
        print(i, ':')
        # pretrain AE
        AE_path = "model/AE/AE.pt"
        if os.path.exists(AE_path):
            print("\tAE already pretrained. Use model in {}".format(AE_path))
        else:
            print("\tPretrain AE")
            subprocess.run(["python", "AE_main.py"])
        # pretrain DAE
        DAE_path = "model/DAE/{}_DAE.pt".format(i)
        if os.path.exists(DAE_path):
            print("\tDAE already pretrained. Use model in {}".format(DAE_path))
        else:
            print("\tPretrain DAE")
            for j in DAE:
                if i in j:
                    dae_file = j
            subprocess.run(["python", "DAE_main.py", "--data_dir", dae_file])
except:
    sys.exit(1)
# %%
for i in total:
    print(i, flush=True)
    subprocess.run(["python", "DAE_MLP_main.py", "--data_dir", i, "--save_finalresult", "1", "--save_model", "1"])