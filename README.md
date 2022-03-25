# Old vs New: an image classification model comparison based on energy consumption and usability

<p align="center"><img src='Data\Use\Validation\VIOLET GREEN SWALLOW\1.jpg' width=200></p>

This is the final project for the course 'Intro to ANNs using Tensorflow', WiSe 2021/2022. Please read the [Documentation](Documentation\Documentation.md) for more information.

<br />

## Introduction

As progress goes on, larger and more powerful neural networks are being developed and tested. Among the Convolutional Neural Networks (CNN) for image classification, a recent one presenting major improvements was Meta's [ConvNeXt](https://arxiv.org/abs/1806.07795). This network is based and built on top of the older [ResNet](https://arxiv.org/abs/1512.03385) architecture, and has shown to reach state-of-the-art performance on the ImageNet dataset, even competing with transformers on tasks such as object detection, image segmentation, and classification.

For simple use cases however, ...

Instead of using smaller lightweight models, people seem to lean towards creating larger models...

We therefore decided to compare the performance and energy consumption of ConvNeXt and ResNet50 on a real-world dataset.

<br />

## Description

We used the [Bird Species](https://www.kaggle.com/gpiosenka/100-bird-species) dataset to perform classification on both models. A detailed description of our used data can be found in the [Documentation](Documentation\Documentation.md). The images are stored in the `/Data/Use/` folder and are resized to a size of 224x224 pixels. The dataset consists of 400 bird classes and about 120-140 images per class. This brings the training set to a total of 58,388 images.

<br />

In our data pipeline, we use the following preprocessing and augmentation steps via the `tf.keras.preprocessing.image.ImageDataGenerator`:

- rescale (Normalization)
- rotation_range (Random rotation)
- width_shift_range (Random horizontal offset)
- height_shift_range (Random vertical offset)
- shear_range   (Cropping) 
- zoom_range (Random zoom)
- horizontal_flip (Flip horizontally)
- zca_epsilon (ZCA whitening)
- brightness_range  (Random brightness)
- fill_mode='nearest' (Points outside the input are filled according to: aaaaaaaa|abcd|dddddddd)

<br />

While training both models, we make use of `nvidia-smi` to measure the energy consumption of the GPU.

<br />

After training, we evaluate the models on the test set (2000 Images) and finally compare our results to check if using a smaller and simpler model would still have reasonable performance on a specific real-world task: 

| Model | Parameters | Val accuracy | Epochs | Average time per epoch | Energy usage | 
|--------|-----------|--------------|--------|---------------|--------------|
| ResNet-50 | 24,407,312 | 0.84 | 10 | 810 s | ~ 93474.32 W / 21.11 KWh |
| ConvNext | 28,127,728 | - | 10 | 1881 s | ~ - W / - KWh |
