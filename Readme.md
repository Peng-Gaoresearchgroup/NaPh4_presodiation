# ProjectName

### Introduction
A script for J. Zhao, et al. Enabling Sodium-Ion Batteries over 180 Wh/kg via Organic-Salt-Driven Sodium Replenishment.

### Contents
```./conf``` Configure file for hyperparameters.
```./data``` Input files for model train, molecule screening, etc.
```./model``` Models.
```./outputs``` Output files generated during run-time.
```./src``` main.py and utils.py

### System requirements
In order to run source code file in the Data folder, the following requirements need to be met:
- Windows, Mac, Linux
- Python and the required modules. See the [Instructions for use](#Instructions-for-use) for versions.

### Installation
```
git clone https://github.com/Peng-Gaoresearchgroup/NaPh4_presodiation.git
```

### Instructions for use
- Environment
```
# create environment, conda is recommended
conda create -n 3118rdkit -c conda-forge rdkit=2024.9.4 python=3.11.8

# install python modules
pip install -r ./requirments.txt

# switch to it
conda activate 3118rdkit
```
- Download pytorch

In this part, you need to install pytorch according to your situation, the basic flow is: install a driver according to your GPU, install a appropriate version of cuda according to the driver, and install a torch library according to the cuda version. You can test whether the torch is running properly by:
```
python -c "import torch;print(torch.cuda.is_available())"
```

- Reproduce the paper

```
python ./src/main.py
```

### Contributions
G. Wu, J. Zhao and Y. Gao designed a workflow. G.Wu completed programming.
### License
This project uses the [MIT LICENSE](LICENSE).

### Disclaimer
This code is intended for educational and research purposes only. Please ensure that you comply with relevant laws and regulations as well as the terms of service of the target website when using this code. The author is not responsible for any legal liabilities or other issues arising from the use of this code.

### Contact
If you have any questions, you can contact us at: yuegao@fudan.edu.cn