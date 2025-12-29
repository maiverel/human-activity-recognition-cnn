# Human Activity Recognition using CNNs

This project implements a convolutional neural network (CNN) for image classification as part of the **Yandex Academy ML Intensive (Spring 2025)**.

The goal is to classify images of people performing different activities using only the provided dataset, without external data or pretrained models.

## Problem Description

Given an image of a person, the model predicts one of **15 activity classes**, including:

- sports
- occupation
- conditioning exercise
- home activities
- lawn and garden
- home repair
- water activities
- winter activities
- fishing and hunting
- dancing
- walking
- music playing
- bicycling
- running
- inactivity (quiet/light)

## Dataset

- **Train set:** 12,000 images  
- **Test set:** 5,000 images  

Files provided:
- `img_train/` — training images  
- `img_test/` — test images  
- `train_answers.csv` — ground truth labels for training images  
- `activity_categories.csv` — mapping from class IDs to human-readable labels  

> The dataset is not included in this repository due to size and licensing constraints.

## Model Architecture

The model is a custom **convolutional neural network implemented in PyTorch**.

Key components:
- Stacked convolutional blocks (Conv2D → BatchNorm → ReLU → MaxPooling)
- Dropout regularization to prevent overfitting
- Fully connected classifier head
- Softmax output over 15 classes

No pretrained weights or external datasets were used.

---

## Training Details

- **Loss function:** Cross-Entropy Loss  
- **Optimizer:** Adam  
- **Evaluation metric:** F1-score (macro)  

The F1-score was chosen to balance precision and recall across classes, especially given potential class imbalance.

The public leaderboard uses only 50% of the test set, so care was taken to avoid overfitting.

## Evaluation Metric

The model is evaluated using the **F1-score**:

F1-score = 2 * (precision * recall) / (precision + recall)


## Submission Format

Predictions are saved in CSV format:

```bash
Id,target_feature
0,0
1,1
2,1
…
```

Where:
- `Id` — index of the image in the test dataset  
- `target_feature` — predicted class ID  

## Results

The trained model achieves a competitive F1-score on the validation split and demonstrates stable generalization without reliance on external data.

## How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2.	Train the model:
```bash
python src/train.py
```

3.	Generate predictions:
```bash
python src/inference.py
```

## Motivation

This project was created as part of my preparation in machine learning and computer vision. I focused on building models from scratch and understanding the full training pipeline.

