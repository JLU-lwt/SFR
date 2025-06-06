[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15607531.svg)](https://doi.org/10.5281/zenodo.15607531)

# SFR_Demo

## I. Introduction
**SFR** is a symbolic regression tool used for complex network symbolic regression. The **README.md** presents two parts: corpus creation and model application. In each part, we provide detailed usage instructions and test demos. If you are intersted, you can also create you own test cases for tesing.

## II. Corpus creation
### 1. How to use
a. In **run** file, use `python -u Create/create_demo.py "$@"`.

b. Click "Reproducible Run".

c. A corpus demo **100_1.csv** will be created and saved in the **results/**.

### 2. Customized cases
You can create corpus of other types or scales by setting various parameters.

```
#type
min_dimension: 1
max_dimension: 3
min_binary_ops_per_dim: 1
max_binary_ops_per_dim: 5
min_unary_ops_per_dim: 0
max_unary_ops_per_dim: 3

''''''
#scale
num_skeletons: 100

''''''
```
we suggest deploying code locally to achieve faster and more flexible corpus creation, and we also provide a multi-processing version **create_multi.py**.

## III. Model application
### 1. How to use (Taking USE_F as an example)
a. In **run** file, use `python -u Model/test_USE_F_demo.py "$@"`.

b. Click "Reproducible Run".

c. A result of demo **USE_F_demo_result.csv** will be created and saved in the **results/**.

d. A visualized result **Figure_0.png** will also be saved in the **results/** if you set `Draw=True`.
![](Figure_0.png)


### 2. Customized cases
You can create equations in the form of equations in the **data/USE_F_demo.csv** for regression testing.

we suggest deploying code locally to imporve efficiency, the model is saved in **code/Model/model_SFR.ckpt** and parameters are listed in **code/Model/config/USE_F.yaml**.

### 3. How to use (Taking USE as an example)

a. In **run** file, use `python -u Model/test_USE_demo.py "$@"`.

b. Click "Reproducible Run".

c. A result of demo **USE_F_demo_result.csv** will be created and saved in the **results/**.

d. A visualized result **Figure_0.png** will also be saved in the **results/** if you set `Draw=True`.
![](Figure_1.png)

### 4. Other information about demos 
- "USE_F_demo": The tesing demo of classical non-network symbolic regression on USE_F.
- "AI_Feynman_demo": The tesing demo of classical non-network symbolic regression on AI_Feynman.
- "USE_demo": The tesing demo of symbolic regression on compolex networks on USE.
- "Dynamics_demo": The tesing demo of inferring interpretabler network dynamics.


## IV. Reference
You can get more information about SFR from our paper https://arxiv.org/abs/2505.21879.

## V. Further reproduction
The complete code will be open sourced for use as soon as possible.





