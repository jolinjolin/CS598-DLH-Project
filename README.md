
# CS598-DLH-Project

This is the CS598 DLH Project to replicate TIMER model (Token Imbalance Adaptation for Radiology Report Generation)

## Paper & Code
- TIMER
    - Paper: [Token Imbalance Adaptation for Radiology Report Generation](https://arxiv.org/abs/2304.09185).
    - Code: [TIMER](https://github.com/woqingdoua/TIMER).

TIMER's architecture and implementation build upon the following baseline models (as evaluated in the TIMER paper):

- BiLSTM 
    - Paper: [Show, Describe and Conclude: On Exploiting the Structure Information of Chest X-ray Reports](https://aclanthology.org/P19-1657/)
    - Code: no public codebase

- R2Gen
    - Paper: [Generating radiology reports via memory-driven transformer](https://arxiv.org/abs/2010.16056)
    - Code: [R2Gen](https://github.com/cuhksz-nlp/R2Gen)

- CMN
    - Paper: [Cross-modal Memory Networks for Radiology Report Generation](https://arxiv.org/abs/2204.13258)
    - Code: [CMN](https://github.com/zhjohnchan/R2GenCMN)

- CMM+RL
    - Paper: [Reinforced Cross-modal Alignment for Radiology Report Generation](https://aclanthology.org/2022.findings-acl.38/)
    - Code: [CMM+RL](https://github.com/synlp/R2GenRL)

## Denpendencies
### Install using conda
```
conda create --name TIMER --file env.yaml
```
### Install using pip
```
conda create --name TIMER python=3.8
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install -r requirements.txt
```

## Run
###
Run a task
```
bash <task>.sh
```

## Datasets
We use two datasets (IU X-Ray and MIMIC-CXR) in our paper.

For `IU X-Ray`, you can download the dataset from [here](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view?usp=sharing) and then put the files in `data/iu_xray`.

For `MIMIC-CXR`, you can download the dataset from [here](https://drive.google.com/file/d/1DS6NYirOXQf8qYieSVMvqNwuOlgAbM_E/view?usp=sharing) and then put the files in `data/mimic_cxr`.

NOTE: The `IU X-Ray` dataset is of small size, and thus the variance of the results is large.
There have been some works using `MIMIC-CXR` only and treating the whole `IU X-Ray` dataset as an extra test set.

## Train

Run `bash train_iu_xray.sh` to train a model on the IU X-Ray data.

Run `bash train_mimic_cxr.sh` to train a model on the MIMIC-CXR data.

## Test

Run `bash test_iu_xray.sh` to test a model on the IU X-Ray data.

Run `bash test_mimic_cxr.sh` to test a model on the MIMIC-CXR data.



