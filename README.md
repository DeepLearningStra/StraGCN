# StraGCN

## Hardware
* CPU x86_64 with host memory >= 32GB. (Test on Intel Xeon Gold 5220 (16 cores) CPU with 128GB host memory)
* NVIDIA GPU (arch>=sm_60) with device memory >= 16GB. We mainly evaluate our design on RTX 4090.
## OS & Compiler
* `Ubuntu 22.04+`
* `GCC >= 9.0`
* `CUDA >= 11.0` and `nvcc >= 11.0`
## Import Files/Directories
* dgl/ : contains latest `DGL` implementation.
* pyg/ : contains latest `PyG` implementation.
* GNNA/ : contains `GNNA` implementation.
* StraGCN/ : contain the `StraGCN` implementation.
## Implementation
### Install environment
* Install `conda` on system
* Create conda environment
```
conda create -n env_name python=3.9
```
* Install pytorch
```
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 -c pytorch
```
* Install PyG
```
pip install torch-geometric
```
* Install DGL
```
conda install -c dglteam dgl-cuda11.0
pip install torch requests
```
* Install StraGCN
```
go to `StraGCN/` and run `python setup.py install` to install StraGCN modules
```
* Install GNNA
```
Go to `GNNAdvisor/GNNConv`, then `python setup.py install` to install the GNNAdvisor modules.
```
### Download the graph datasets
* Our preprocessed graph datasets in .npy format can be downloaded via `https://zenodo.org/records/15258956`
* Unzip the graph datasets `tar -zxvf StraGCN-graphs.tar.gz` at the project root directory

## Detailed Instructions
### Running DGL on GCN training
```
Go to `dgl/` directory and run "python gcn.py"
```
### Running PyG on GCN training
```
Go to `dgl/` directory and run "python pyg_main.py"
```
### Running GNNA on GCN training
```
Go to `GNNA/` directory and run "python GNNA_main.py"
```
### Running StraGCN on GCN training
```
Go to `StraGCN/` directory and run "python gcn.py"
```
