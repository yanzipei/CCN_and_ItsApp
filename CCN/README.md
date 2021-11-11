# Experiments on CCN

## Introduction
This repository contains the code for experiments on CCN.

## Usage
### Initialize
unzip `data/data.zip` to export the data.

### Train SVM
Use the default parameters in `train_svm.py`
```
python train_svm.py --data ./data/aug_data.npy --log_dir ./runs/train/svm
```

### Train MLP
Only supports single GPU, uses the default parameters.
```
python train_mlp.py --data ./data/aug_data.npy --log_dir ./runs/train/mlp
```

### Train ColorMLP
Only supports single GPU, uses the default parameters.
```
python train_color_mlp.py --data ./data/aug_data.npy --log_dir ./runs/train/color_mlp
```

## Structure
```text
.
├── README.md
├── data
│   ├── data.zip
├── ccn.py
├── config.py
├── generate_aug_data.py
├── generate_cleaned_data.py
├── get_rgb_mean_and_std.py
├── modules
│   ├── color_mlp.py
│   ├── gat_loss.py
│   └── mlp.py
├── predict_by_nearest_neighbor.py
├── rectify_data.py
├── rgb_channel_adj.py
├── train_color_mlp.py
├── train_mlp.py
├── train_svm.py
└── util.py
```
## Description
`data/data.zip`: zip file of cleaned data, augmented data and final rectified result.

`ccn.py`: classifier with reference from rectified results.

`config.py`: training configurations for classifiers: MLP, ColorMLP.

`generate_aug_data.py`: generate augmented data for training classifiers.

`generate_cleaned_data.py`: generate cleaned data for training classifiers.

`get_rgb_mean_and_std.py`: calculate mean and std.

`modules/color_mlp.py`: ColorMLP model.

`modules/gat_loss.py`: loss in Eq.(5)

`modules/mlp.py`: MLP model.

`predict_by_nearest_neighbor.py`: do prediction on RGB cube with reference to the nearest neighbor by using cleaned data.

`rectify_data.py`: rectify the classification results of ColorMLP.

`rgb_channel_adj.py`: generate adjacent matrix on RGB channel.

`train_color_mlp.py`: train ColorMLP.

`train_mlp.py`: train MLP.

`train_svm.py`: train SVM.

`util.py`: common tool functions.


