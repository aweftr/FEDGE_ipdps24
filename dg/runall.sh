#! /bin/bash
# source your conda environment

echo "running LR"
python runLR.py
echo "LR Done!"

echo "running MLP"
python runMLP.py
echo "MLP Done!"

echo "running XGB"
python runXGB.py
echo "XGB Done!"

echo "running FEDGE"
python runFEDGE.py
echo "FEDGE Done!"

echo "running FEDGE-noDG"
python runFEDGE.py --type "noDG"
echo "FEDGE-noDG Done!"

echo "running FEDGE-noD"
python runFEDGE.py --type "noD"
echo "FEDGE-noD Done!"

echo "running FEDGE-noM"
python runFEDGE.py --type "noM"
echo "FEDGE-noM Done!"

echo "running FEDGE-FS"
python runFEDGE.py --type "FS"
echo "FEDGE-FS Done!"

echo "Selecting features acoording to the result of FEDGE-FS"
python analyse/select_features.py
echo "Select features Done!"

echo "Merge the ouput of different methods"
python analyse/analyse_output.py
echo "All Done!"