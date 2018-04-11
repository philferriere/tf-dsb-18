# 2018 Data Science Bowl
Dual U-Net Nuclei Segmentation solution.

## Data 

Download and unzip in a local folder (`dataset_root`):

https://raw.githubusercontent.com/AakashSudhakar/2018-data-science-bowl/master/compressed_files/stage1_test.zip
https://github.com/lopuhin/kaggle-dsbowl-2018-dataset-fixes/archive/master/stage1_train.zip

## Trained Models

Trained models are in [model_train_unet_256x256_nuclei_2px_contours_cyclic_lr.zip](https://drive.google.com/open?id=1ueFyYVR_-UbJC0U44iLvRpay_CqOxNji)

## Environment

Create a conda environment as defined in [dlwin36.yml](dlwin36.yml) or [requirements.txt](requirements.txt)

## Notebooks

## Data Preparation

Run the [dataset_prep.ipynb](dataset_prep.ipynb) jupyter notebook after having set `dataset_root` to the proper dataset folder.

## Model Training & Testing

Run the [model_train_unet_256x256_nuclei_2px_contours_cyclic_lr.ipynb](model_train_unet_256x256_nuclei_2px_contours_cyclic_lr.ipynb) jupyter notebook after having set `dataset_root` to the proper dataset folder.
