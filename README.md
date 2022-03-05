# Bird species classification with data distillation alternatives

<p align="center"><img src='Data\Use\Valid\VIOLET GREEN SWALLOW\1.jpg' width=200></p>

This is the final project for the course 'Intro to ANNs using Tensorflow'. Please read the [Documentation](Documentation\Documentation.md) for more information.

## Introduction

Training models are bad for the environment.... 

We read about a newly found method for training deep neural networks with very little data and achieving very good results using something called [data distillation via kernel induced points (KIP)](https://ai.googleblog.com/2021/12/training-machine-learning-models-more.html).

<p align="center"><img src='Documentation\Media\KIP_distillation.png' width=800></p>

We wanted to try out something different that also perserves the features from each class at a low dimensional space, while being able to be used for training a classifier. We also wanted to see if we could then use this distillation method in a classifier directly, not as a preprocessing step and compare the results.

## Description

We used a lightweight (<100,000 param.) Convolutational Neural Network as the benchmark to classify a subset of images of birds from the [Bird Species](https://www.kaggle.com/gpiosenka/100-bird-species) dataset. A detailed description of our used data can be found in the [Documentation](Documentation\Documentation.md). The images are stored in the `/Data/` folder and are resized to a size of 224x224 pixels. 

We first applied regular data preprocessing steps, such as:

- Normalization of the images to the range [0,1]
- Augmentation of the images with random transformations
    - Translation
    - Rotation
    - Scaling
    - Shearing
    - Horizontal flip
- Adding random (gaussian) noise

And saved the resulting images to the `/Data/Preprocessed/Normal` folder.

We then applied our experimental data distillation methods, such as:

- Averaging over the images of the same class
- Using PCA to extract the most important features of each class
- Using the embeddings of an autoencoder to extract the most important features of each class

This resulted in low dimensional embeddings of the images, which we saved into respective `/Data/Preprocessed/Distilled/...` folders. 

Finally, we compared our results to check if data distillation is indeed able to train and perform more efficiently on our lightweight CNN: 

| Method | Accuracy on Lightweight CNN | Training Time | Energy usage |
|--------|------------------------------|---------------|--------------|
| No preprocessing | 0.9 | 1.5 hours | ~0.5 GB |
| Normal preprocessing | 0.9 | 1.5 hours | ~0.5 GB |
| Distillation: Average | 0.9 | 1.5 hours | ~0.5 GB |
| Distillation: PCA | 0.9  )| 1.5 hours | ~0.5 GB |
| Distillation: Autoencoder | 0.9  | 1.5 hours | ~0.5 GB |

