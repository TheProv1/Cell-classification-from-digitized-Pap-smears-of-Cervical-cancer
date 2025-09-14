# Cell classification from digitized Pap smears of Cervical cancer

## Tech Used

Language: Python

Packages: Tensorflow, Keras

## Dataset
### Dataset information

The dataset contains two directories

1. Training
2. Testing

The training dataset contains just a little over 85K images grouped into 4 classes, with a 70-30 split (training-validation).

The testing dataset contains about 18.1K images.

Size: 18G

Dataset link: https://www.kaggle.com/competitions/pap-smear-cell-classification-challenge

## Code Description

1. core_values.py: Contains the common values which are used in both the scripts.

2. project.ipynb: Main code consisting of the model definition, image augmentation, and training

3. preprocess_data.ipynb: Script for loading, pre-processesing, and normalizing the data. Finally, storing the normalized data into a new directory for quick read and access
