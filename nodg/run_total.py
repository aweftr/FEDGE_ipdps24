import os
import subprocess

total = ["data/output/pred_LC/total-LC.csv", "data/output/pred_TPS/total-TPS.csv"]

subprocess.run(["python", "DAE_main.py", "--data_dir", "data/output/DAE/total-stress_nostress_workload_merged.csv"])

for i in total:
    # subprocess.run(["python", "LR_main.py", "--data_dir", i, "--save_finalresult", "1"])
    # subprocess.run(["python", "XGB_main.py", "--data_dir", i, "--save_finalresult", "1"])
    # subprocess.run(["python", "MLP_main.py", "--data_dir", i, "--save_finalresult", "1", "--save_model", "1"])
    subprocess.run(["python", "DAE_MLP_main.py", "--data_dir", i, "--save_finalresult", "1", "--save_model", "1"])