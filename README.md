# Bird Species with Data Distillation

<p align="center"><img src='Data\valid\ABBOTTS BABBLER\1.jpg' width=200></p>

This is the final project for the course 'Intro to ANNs with Tensorflow'.

## Description

Our project is to use a Lightweight* Convolutational Neural Network to classify images of birds from the [Bird Species](https://www.kaggle.com/gpiosenka/100-bird-species) dataset. The dataset contains 385 different types of birds, and each image is a 224x224x3 colored pixel image. The images are stored in the `Data/` folder.

We want to use 2 types of preprocessing techniques, namely data augmentation and *data distillation*. And finally compare results.

Data distillation is a fairly new technique used to improve the performance of a model by reducing required memory and compute. The idea is to use a 'distilled' dataset, containing a subset of the real dataset, with one distilled image per class (i.e. one image per bird). The model is then trained on the distilled dataset, and the model is then tested on the real dataset. The model is then used to classify the real images, without having used the large original dataset.



