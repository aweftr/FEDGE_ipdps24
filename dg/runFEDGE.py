import subprocess
import time
import numpy as np
import argparse
tl = []

def get_parameters():
    parser = argparse.ArgumentParser(description="run FEDGE and its variant")
    parser.add_argument("--type", type=str,
                        default='', help="FEDGE type (noDG, noD, noM, FS)")
    args = parser.parse_args()
    return args

args = get_parameters()
FEDGE_type = "FEDGE"
if args.type:
    FEDGE_type += "-{}".format(args.type)
print(FEDGE_type)
for i in range(8):
    print(i, flush=True)
    start_time = time.time()
    subprocess.run(["python", "{}_main.py".format(FEDGE_type), "--target", str(i)])
    end_time = time.time()
    print(end_time - start_time)
    tl.append(end_time - start_time)

tl = np.array(tl)
with open("output/{}_run_time".format(FEDGE_type), "w") as f:
    f.write(str(tl.mean()))
print(tl.mean())