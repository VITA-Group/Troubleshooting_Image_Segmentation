# Troubleshooting Image Segmentation Models withHuman-In-The-Loop

Haotao Wang, Tianlong Chen, Zhangyang Wang, Kede Ma

In Machine Learning Journal, 2021

## Overview
We propose a human-in-the-loop framework to efficiently troubleshoot state-of-the-art image segmentation models by iteratively spotting and fixing its errors.

![Proposed framework for troubleshooting segmentation models.](https://github.com/VITA-Group/Troubleshooting_Image_Segmentation/blob/main/framework.png)

## Crawl images from internet
Run `python images-web-crawler/sample_ours.py` under project root dir, or run `python sample_ours.py` under `images-web-crawler`.  

## Clean dataset
Run `python make_dataset.py` to copy all RGB images crawled by different keywords to a single folder.

## Pretrain DeepLabV3 on VOC dataset
Run `python train.py`.

## Troubleshoot pretrained DeepLabV3 by selecting MAD images
Run `python MAD_selection.py`. This saves two `.npy` files for each attacker-defender pair. For example `top_mIoUs_deeplabv3_resnet101_fcn_resnet101.npy` where deeplabv3_resnet101 is defender and fcn_resnet101 is attacker.

## Fix errors of pretrained DeepLabV3 by fine-tuning on the selected error-cases (MAD images)
Run `python finetune.py`.

