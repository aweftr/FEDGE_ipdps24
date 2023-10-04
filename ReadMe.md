## Code and data for FEDGE

### Structure
| Folder          | Description                                                                                                       |
| --------------- | ----------------------------------------------------------------------------------------------------------------- |
| data_processing | Code for processing the dataset.                                                                                  |
| dg              | Code for FEDGE's domain generalization tasks.                                                                     |
| nodg            | Archive code of the DACBIP (former FEDGE). <br />The neural architecture and hyperparameters are searched by NNI. |

### Dataset
We use the open source interference-aware QoS degradation dataset [Alioth-dataset](https://github.com/StHowling/Alioth). The dataset contains two zip packages, where `data-mul.zip` is the data collected by stressing applications using stressors, and the other `data-app.zip` is the data from multiple applications mixed together to generate stress. 
