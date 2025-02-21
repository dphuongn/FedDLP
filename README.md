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

We provide **7** popular datasets: **Fashion-MNIST (F-MNIST)**, **CIFAR-10**, **CIFAR-100**, **Oxford-IIIT Pet (OxfordPets)**, **Oxford 102 Flower (Flowers102)**, **FGVC-Aircraft (Aircraft)**, and **Describable Textures Dataset (DTD)**. They can be easy split into **IID** and **non-IID** version. For **non-IID**, we have practical setting (with hyperpameter for Dirichlet distribution $\beta$) and pathological setting (few-shot scenario). 

### Examples for **Oxford-IIIT Pet**
- Total 10 clients, iid and balance scenario
    ```
    cd ./dataset
    python generate_p37.py 10 iid balance - - - - 
    ```

- Total 10 clients, practical noniid scenario with $\beta = 0.1$ 
    ```
    cd ./dataset
    python generate_p37.py 10 noniid - dir 0.1 - - 
    ```


- Total 10 clients, practical noniid scenario with $\beta = 0.01$ 
    ```
    cd ./dataset
    python generate_p37.py 10 noniid - dir 0.01 - - 



## Training and Evaluation

After generating and partitioning dataset for clients, we can run the training and evaluation. All codes corresponding to **FLoRA** and other baselines: **FedFFT**, **FedLC**, **FedVM-LC**, and **FedAA** are stored in `./script`. Different folder corresponds with that specific dataset.

### Examples for **FedLC** on **OxfordPets** with **Practical non-IID** scenario
```
cd ./scripts/p37
bash fedlc.sh
```

### Examples for **FLoRA** on **Caltech101** with **IID few-shot** scenario
```
cd ./scripts/caltech101
bash flora.sh
```

## Parameters

| Parameter | Description |
| --------- | ----------- |
|`data`     | Dataset to use. Options: `(f10) fmnist`, `(c10) cifar10`, `(c100) cifar100`, `(p37) pets`, `f102 (flowers)`, `a100 (aircraf)`, `d47 (dtd)`.|          
| `m`       | The base model. Options: `vit-b-32`, `vit-b-16`, `vit-l-14`, and `vit-l-14-336` (default: `vit-b-32`).|
| `algo`     | The training algorithm. Options: `fedloralocal (Local training)`, `fedaa (FedCLIP)`, `flora (FLoRA)`, `feddat (FedDAT)`, and `feddlp (out method)`.|
| `gr`      | Number of communication rounds. |
| `jr`      | Ratio of participating clients per round (default: `1`). |
| `did`     | GPU device ID (default: `0`). |
| `nc`      | Number of clients. |
| `lbs`     | Batch size. |
| `lora_rank`               | The LoRA rank for **FLoRA**.|
| `lora_alpha`              | The LoRA scaling factor for **FLoRA**.|
| `lora_projection_text`    | LoRA apply to projection text for **FLoRA**.|
| `lora_projection_vision`  | LoRA apply to projection vision for **FLoRA**.|
| `sd`      | The random seed. |


Feel free to change parameters to your desired experiments. If you use the same setting as our papers, you can simply adopt the hyperparameters reported in our paper.

# Acknowledgement

This code is heavily inspired from the popoular federated learning project [PFLlib](https://github.com/TsingZ0/PFLlib). Big shout out for their wonderful work!