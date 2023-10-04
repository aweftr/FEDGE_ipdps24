# DACBIP (former FEDGE) code

This is the archive of the DACBIP code. The latest FEDGE code is in the `dg` folder! 

`model.py` is the model implementation of all of the deep learning methods 
used in FABIP. 


## Baseline
### Linear Regression
- `LR_main.py`: Main Linear Regression predictor. 
Feeding with a input dataset, it will split this dataset into train and test dataset, 
learn a linear regression model and output the test error. 
- `run_LR.py`: Load dataset of different applications and train model to get the prediction error. 

### XGBoost
- `XGB_main.py`: Main MLP predictor. 
Feeding with a input dataset, it will split this dataset into train and test dataset, 
learn a model and output the test error. 
- `run_XGB.py`: Load dataset of different applications and train model to get the prediction error. 

### Multilayer Perceptron
- `model.py`: class MLP is the prediction model and class MLP_dataset deals with the dataset. 
- `MLP_main.py`: Main MLP predictor. 
Feeding with a input dataset, it will split this dataset into train and test dataset, 
learn a MLP model and output the test error. Run `python MLP_main.py --help` to get more information. 
- `config/MLP_config.json`: MLP parameters and hyperparameters config, overwritten by CMD parameters.
- `log/MLP.log`: MLP will log its intermidiate results to this file. 
- `run_MLP.py`: Load dataset of different applications and train model to get the prediction error. 



## FABIP (DACBIP v2)
- `model.py`: class AE, DAE, DAE_MLP
- `AE_main.py`: Train an AE model using PMr dataset, save the model to `model/AE/AE.pt`. 
- `DAE_main.py`: Train a DAE model using different application dataset or different cluster dataset, 
save the model to `model/DAE/{app}_DAE.pt` 
- `DAE_MLP.py`: Main DAE_MLP predictor. 
Feeding with a input dataset, it will split this dataset into train and test dataset, 
load the pretrained AE and DAE model, learn a DAE_MLP model and output the test error. 
Run `python DAE_MLP.py --help` to get more information. 
- `config/DAE_MLP_config.json`: DAE_MLP parameters and hyperparameters config, overwritten by CMD parameters. 
- `log/DAE_MLP.log, AE.log, DAE/log`: log of intermidiate results. 
- `run_DAE_MLP.py`: Load dataset, **pretrain AE and DAE** if no pretrained model is available, train model to get the prediction error. 


Training sequence: 
`AE` -> `DAE` -> `DAE_MLP`. 