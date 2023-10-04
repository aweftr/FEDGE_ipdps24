import subprocess
import time
import numpy as np
tl = []

for i in range(8):
    print(i, flush=True)
    start_time = time.time()
    subprocess.run(["python", "LR_main.py", "--target", str(i)])
    end_time = time.time()
    print(end_time - start_time)
    tl.append(end_time - start_time)

tl = np.array(tl)
with open("output/LR_run_time", "w") as f:
    f.write(str(tl.mean()))
print(tl.mean())