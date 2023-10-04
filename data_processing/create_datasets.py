# Merge different workload of one app into one file. Add app, workload information for creating dataset
# %%
import os
import pandas as pd
import shutil
# %%
DATADIR = r"data/mul"
os.chdir(DATADIR)
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
csv_file_list = []
find_all_csv("./", csv_file_list, "outlier-3merge.csv")
# %%
app_csvs = {}
for i in csv_file_list:
    print("Dealing with {}".format(i))
    dirname = os.path.dirname(i)
    appname = os.path.basename(i).split('-')[0]
    # if appname == 'noapp':
    #     continue
    if appname not in app_csvs:
        app_csvs[appname]=[]
    app_csvs[appname].append(i)
# %%
workload_dict = {}
for app in app_csvs:
    df_out = pd.DataFrame()
    outdir = "../app-data"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if app == "noapp":
        shutil.copy(app_csvs[app][0], os.path.join(outdir, "noapp-merged.csv"))
        continue
    workload_dict[app] = []
    for f in app_csvs[app]:
        dirname = os.path.dirname(f)
        workload = os.path.basename(dirname)
        workload_dict[app].append(workload)
        df = pd.read_csv(f)
        # only need leave one no stress point
        nostress = df[df['stress_type'] == "NO_STRESS"]
        nostress_idx = nostress.index
        keys = list(df.columns)
        start_idx = keys.index("cpu_time")
        end_idx = keys.index("stress_type")
        nostress_mean = nostress.iloc[:, start_idx: end_idx].mean()
        df.iloc[0, start_idx: end_idx] = nostress_mean
        df.drop(nostress_idx[1: len(nostress_idx)], axis=0, inplace=True)
        df.reset_index(drop=True, inplace=True)

        if app == "kafka":
            df.rename(columns={"tps.1": "count"}, inplace=True)
        app_df = [app] * df.shape[0]
        workload_df = [workload] * df.shape[0]
        df.insert(df.shape[1], "app", app_df)
        df.insert(df.shape[1], "workload", workload_df)
        df_out = pd.concat([df_out, df], axis=0)
    df_out.to_csv(os.path.join(outdir, app + "-merged.csv"), index=False)

# %%