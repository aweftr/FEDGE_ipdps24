### Code for FEDGE

1. Please first follow the instruction in the data_processing folder to generate the dataset for domain generalization. After running the data processing scripts, you should get a `dg-data` folder or something with similar names. 

2. Put the `data` folder into this directory and run the code. `run*.py` trains the corresponding model using the data of 7 known applications and tests the model on the remaining unseen application. 8 different domain generalization tasks are evaluated sequentially. `runall.sh` trains and tests all available models on all available generalization tasks. The MAE results are saved to the `output` folder and the parameters are saved to the `model` folder. 

3. After running the code for all of the methods, run the scripts in the `analyse` folder to select features and analyze the output. The selected features are stored in `data/selected_features.json`, while the MAE results for all methods are stored in `output/all.csv`. 