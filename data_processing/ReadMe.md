## Data Processing

| File               | Description                                                                                                            | Output Files          |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------- | --------------------- |
| data_generate.py   | Drop not counted values, filter outliers, min max scale and merge data every 3 seconds                                 | *-outlier-3merge.csv  |
| create_datasets.py | Merge different workload of one app into one file. Add app, workload information for creating dataset                  | app-data/*-merged.csv |
| qos_datasets.py    | Import this file to select workload, features, QoS or stress as you wish! Please refer to the docstring of each class. | \                     |
| generate_dgdata.py | Generate the final dataset for code training. Select features using a json file.                                       | dg-data/{app}.pickle  |

### Usage
Please use the newest data. 

`data_generate.py` -> `create_datasets.py` -> `generate_dgdata.py` to generate the domain generazation dataset for training. 

Or

`data_generate.py` -> `create_datasets.py` -> import `qos_datasets` to use data or dataset classes. 