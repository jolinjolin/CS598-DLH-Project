
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
conda env create --file env.yaml
```
### Install using pip
```
conda create --name TIMER python=3.8
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install -r requirements.txt
```
Additionally, [scispacy](https://allenai.github.io/scispacy/) is needed for name entity recognition in the dataset and [Chexpert-labeler](https://github.com/stanfordmlgroup/chexpert-labeler) is needed for report labeling in order to calculation F1 scores.


## Run
###
Run a task
```
bash <task>.sh
```
For example, train TIMER
```
bash train_iu_xray_TIMER.sh
```
To test TIMER, uncomment `save_report` and update paths in the `test_iu_xray.sh` and then run
```
bash test_iu_xray.sh
```
Postprocessing, run
```
bash f1_scores.sh
```

## Datasets
We use [IU X-Ray](https://openi.nlm.nih.gov/) Chest X-RAY dataset from Indiana University for this study.

The processed data of `IU X-Ray` can be downloaded from [here](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view?usp=sharing).


