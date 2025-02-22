# Introduction

This is the implementation of the paper Federated Multimodal Learning with Dual Adapters and Selective Pruning for Communication and Computational Efficiency. 

## Operating System
Linux x86_64

## Environments
With the installed [conda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh), we can run this platform in a conda virtual environment called *dlp*. 
```
conda env create -f env.yaml # for Linux or Windows with CUDA >= 12.1
conda activate dlp
```

## Generating datasets

We provide **4** datasets: **Oxford-IIIT Pet (Pets)**,  **Describable Textures Dataset (DTD)**, **Oxford 102 Flower (Flowers)**, and **FGVC-Aircraft (Aircraft)**. They can be easy split into **IID** and **non-IID** version. For **non-IID**, we have practical setting (with hyperpameter for Dirichlet distribution $\beta$) and pathological setting (few-shot scenario). 

### Examples for **Pets**
- Total 10 clients, iid and balance scenario
    ```
    cd ./dataset
    python generate_pets.py 10 iid balance - - - - pfl
    ```

- Total 10 clients, practical noniid scenario with $\beta = 0.1$ 
    ```
    cd ./dataset
    python generate_pets.py 10 noniid - dir 0.1 - - pfl
    ```

- Total 10 clients, practical noniid scenario with $\beta = 0.01$ 
    ```
    cd ./dataset
    python generate_pets.py 10 noniid - dir 0.01 - - pfl



## Training and Evaluation

After generating and partitioning dataset for clients, we can run the training and evaluation. All codes corresponding to **FedDLP** and other baselines: **Local**, **FedCLIP**, **FLoRA**, and **FedDAT** are stored in `./scripts`. Different folder corresponds with that specific dataset. From the `FedDLP` folder, run the following, the results will be saved in the `logs` folder.

### Examples for **Local Training** on **DTD** 
```
bash ./scripts/dtd/localora_text.sh         # for text encoder
bash ./scripts/dtd/localora_image.sh        # for image encoder
```

### Examples for **FedCLIP** on **Pets** 
```
bash ./scripts/pets/fedclip_text.sh         # for text encoder
bash ./scripts/pets/fedclip_image.sh        # for image encoder
```

### Examples for **FLoRA** on **DTD** 
```
bash ./scripts/dtd/flora_text.sh            # for text encoder
bash ./scripts/dtd/flora_image.sh           # for image encoder
```

### Examples for **FedDAT** on **Flowers** 
```
bash ./scripts/flowers/feddat_text.sh       # for text encoder
bash ./scripts/flowers/feddat_image.sh      # for image encoder
```

### Examples for **FedDLP** on **Aircraft**
```
bash ./scripts/aircraft/feddlp_text.sh      # for text encoder
bash ./scripts/aircraft/feddlp_image.sh     # for image encoder
```

## Parameters

| Parameter | Description |
| --------- | ----------- |
|`data`     | Dataset to use. Options: `fmnist`, `cifar10`, `cifar100`, `pets`, `dtd`, `flowers`, `aircraft` (default: `pets`).|          
| `m`       | The base model. Options: `vit-b-32`, `vit-b-16`, `vit-l-14`, and `vit-l-14-336` (default: `vit-b-32`).|
| `algo`    | The training algorithm. Options: `fedloralocal (Local LoRA training)`, `fedclip (FedCLIP)`, `flora (FLoRA)`, `feddat (FedDAT)`, and `feddlp (FedDLP: out method)` (default: `feddlp`).|
| `gr`      | Number of communication rounds (default: `100`). |
| `jr`      | Ratio of participating clients per round (default: `1`). |
| `did`     | GPU device ID (default: `0`). |
| `nc`      | Number of clients (default: `10`). |
| `lbs`     | Training batch size (default: `32`). |
| `lr`      | Learning rate (default: `1e-5`). |
| `wd`      | Weight decay (default: `0`). |
| `pfl`     | For personalized federated learning if present, traditional federated learning otherwise. |
| `sd`      | The random seed (default: `0`). |


Feel free to change parameters to your desired experiments. If you use the same setting as our papers, you can simply adopt the hyperparameters reported in our paper.

# Acknowledgement

This code is heavily inspired from the popoular federated learning project [PFLlib](https://github.com/TsingZ0/PFLlib). Big shout out for their wonderful work!