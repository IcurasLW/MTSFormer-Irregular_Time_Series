# Irregularity-Informed Time Series Analysis: Adaptive Modelling of Spatial and Temporal Dynamics

## Overview
This repository is  **Official** implementation of MTSFormer code for manuscripts *Irregularity-Informed Time Series Analysis: Adaptive Modelling of Spatial and Temporal Dynamics*, accepted by *CIKM 2024 Full Paper Track*. We proposed a multi-view transformer architecture with adaptive irregularity gate mechanism to unify irregularity learning in Naturally Irregular Time Series(NIRTS) and Accidentally Irregular Time Series(AIRTS). We evaluated our method in 5-fold cross validation under two experiments settings. Please see our paper for more details.


![Overall_Architecture](./figures/Overall_Architecture.png)


## Datasets
Followed by [Raindrop, ICLR'22](https://arxiv.org/pdf/2110.05357), we used the same data preprocessing. You can download the ready-to-use dataset in [Google Drive](https://drive.google.com/drive/folders/1VrB2mbiF58pS9UggxDecJu8T7qgY53TC?usp=sharing). There are three dataset from real-world application, P12 from ICU monitoring, P19 from sepsis monitoring, PAM from Physical Activity Monitoring. The description of dataset can be found in [Raindrop, ICLR'22](https://arxiv.org/pdf/2110.05357). We give the statistics of dataset:

![dataset_statistics](./figures//Dataset_statistics.png)


## Environment setup

```bash
conda create -n IRTS python==3.9
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Run Training
```bash
sh run_P12.sh
sh run_P19.sh
sh run_PAM.sh
```