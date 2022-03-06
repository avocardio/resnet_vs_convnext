# Lightweight bird species classification with data distillation alternatives

<p align="center"><img src='Data\Use\Valid\VIOLET GREEN SWALLOW\1.jpg' width=200></p>

This is the final project for the course 'Intro to ANNs using Tensorflow'. Please read the [Documentation](Documentation\Documentation.md) for more information.

<br />

## Introduction

Training models are bad for the environment.... 

We read about a newly found method for training deep neural networks with very little data and achieving very good results using something called [data distillation via kernel induced points (KIP)](https://ai.googleblog.com/2021/12/training-machine-learning-models-more.html).

<p align="center"><img src='Documentation\Media\KIP_distillation.png' width=800></p>

We wanted to try out something different that also perserves the features from each class at a low dimensional space, while being able to be used for training a classifier. We also wanted to see if we could then use this distillation method in a classifier directly, not as a preprocessing step and compare the results.

<br />

## Description

We used a lightweight (<100,000 param.) Convolutational Neural Network as the benchmark to classify a subset of images of birds from the [Bird Species](https://www.kaggle.com/gpiosenka/100-bird-species) dataset. A detailed description of our used data can be found in the [Documentation](Documentation\Documentation.md). The images are stored in the `/Data/` folder and are resized to a size of 224x224 pixels. For our purposes, we decided to use the top 10 classes with most images, and out of those images kept 100 per class. 

<br />

For the first part, we applied regular data augmentation methods, such as:

- Brightness
- Contrast
- Satuation
- Horizontal flip
- Random (gaussian) noise
- Salt and pepper noise
- "Grid" noise

One random augmentation was performed per image and saved to the `/Data/Preprocessed/Augmented` folder, along with the original images. We then had a total of 200 images per class.

<br />

For the second part, we applied our experimental data distillation methods, such as:

- Averaging over the images of the same class
- Using PCA to extract the most important features of each class
- Using the embeddings of an autoencoder to extract the most important features of each class

This resulted in low dimensional embeddings of the images, which we saved into respective `/Data/Preprocessed/Distilled/...` folders. 

<br />

Finally, we compared our results to check if data distillation is indeed able to train and perform more efficiently on our lightweight CNN: 

| Method | Accuracy on Lightweight CNN | Epochs | Training Time | Energy usage | 
|--------|------------------------------|-------|-------|--------------|
| No preprocessing | 0.9 | 1.5 hours | ~0.5 GB | ~0.5 GB |
| Normal preprocessing | 0.97 | 50 | 140s | ~ 3688W / 0.1KWh |
| Distillation: Average | 0.9 | 1.5 hours | ~0.5 GB | ~0.5 GB |
| Distillation: PCA | 0.9  | 1.5 hours | ~0.5 GB |  ~0.5 GB |
| Distillation: Autoencoder | 0.9  | 1.5 hours | ~0.5 GB | ~0.5 GB |

