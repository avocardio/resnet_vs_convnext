TODOs:

- Try autoencoder for 10 images per class
- Confusion matrix
- Write on Documentation file in parallel
- Try Eigenfaces on bird dataset
- Transfer documentation to Overleaf
- Think about how to store and provide the dataset to tutors
- Need some sort of abstract
- Include GPU analysis



Structure
=========
Intro
Idea of the Project
Dataset
Models
Findings & results
Evaluation
Conclusion

# Documentation 

## Introduction

This project aims at increasing the training efficiency of a standard CNN (or heavyweight VGG-16??) in a multi-label classification task and thus cut down on energy consumption and costs. We attempted to show that by firstly, changing the standard CNN to an Ultralightweight CNN, and secondly, by applying data distillation to our chosen dataset can indeed reach these goals.

The project falls under the classification tasks of Machine Learning. In particular, it involves multi-label classification of input examples. First, we train a standard CNN to classify images of birds from the [Bird Species](https://www.kaggle.com/gpiosenka/100-bird-species) dataset reaching a test accuracy of ***%. To show that the same results can be achieved with less training effort, we deploy a Lightweight* Convolutational Neural Network. As the next step of your project, we attempt to deploy 2 types of preprocessing techniques, namely data augmentation and *data distillation*. Data distillation is a fairly new technique used to improve the performance of a model by reducing required memory and computations. The idea is to use a 'distilled' dataset, containing a somewhat synthesized subset of the original dataset, with one distilled image per class (i.e. one image per bird). The model is then trained on the distilled dataset, and tested on the real dataset. With our results we want to check, if both models can be trained more efficiently with a subset of the original data but reaching the same test accuracy when compared with training and testing on the whole, original dataset.

In data_exploration.ipynb, we check and take a look at a few example images of each class. It can be noted that the images are already cropped so that all birds can be found in the center of the images.


## Preparation Phase

### Dataset

The original dataset contains 385 different types/classes of birds, and each image is a 224x224x3 coloured pixel image. To save time and energy overall, we decided to bring down the number of types/classes to 10. These 10 types/classes comprise of the bird species of which there exist at least 200 images in the original dataset each. Thereby, while reducing the overall amount of data, we assure that the model is seeing enough data during the training phase in order to be able to generalise and perform well.


### Model Design Decisions

1) VGG16 Model or VGGNet - Very Deep Convolutional Neural Network
====================================================================
(https://viso.ai/deep-learning/vgg-very-deep-convolutional-networks/)
https://medium.com/@mygreatlearning/what-is-vgg16-introduction-to-vgg16-f2d63849f615#:~:text=VGG16%20is%20a%20simple%20and,visual%20object%20recognition%20software%20research.


As the next step, we are presenting the preprocessed data to a VGG16 Model. ‘VGG’ is short for 'Visual Geometry Group', who are the developers of this specific CNN architecture. The ‘16’ refers to the number of layers that are being used in the network. VGGs are inspired by the basic convolution neural network design however, larger kernels that are usually integrated in CNNs are replaced with multiple smaller 3x3 kernels. In total, it comprises of 13 convolutional layers and three fully connected layers. While VGGs can differentiate "1000 objects", with a "total of around 138 million parameters", VGG16 "is a [very] extensive network." As a benchmark, it is said that "[t]he VGG16 model achieves almost 92.7% top-5 test accuracy in ImageNet." "The number of filters that we [can] use doubles on every step or through every stack of the convolution layer. This is a major principle used to design the architecture of the VGG16 network. One of the crucial downsides of the VGG16 network is that it is a huge network, which means that it takes more time to train its parameters."

2) Vanilla CNN - Convolutional Neural Network
==============================================
COMMENT: do we still need this or shall we focus on 2 network architectures so that it doesn't get messy?

3) Ultra Lightweight Convolutional Neural Network
=================================================
COMMENT: input from miguel



### Data pre-processing

We took the original data set and sorted all images by their labels. We then took 100 random images per class to have an even and balanced amount of inputs per bird.

- What other pre-processing steps did we use on the images and why
- Maybe show some examples of data before and after pre-processing?
- Cross validation etc

### Data Augmentation

In order to have access to more diverse data, we decided to apply several data augmentation techniques. We hoped that by presenting the models with a more diverse data set, it would be able to generalise and perform better. The results can be found in the 'results' section. We applied a total of 11 augmentation techniques which are listed and explained hereunder:

1) Brightness: we introduced a simple function to change the brightness values of an input image to random brightness values.
2) Contrast: we introduced a simple function to change the global contrast of an input image to a random global contrast.
3) Saturation: we introduced a simple function to randomly change the saturation of an input image.
4) Horizontal flip: a simple function that flips the original image to either the right or the left.
5) Random (gaussian) noise: a simple function that adds noise to the original image in the form of random colour values in the range of 0-255.
6) Median blur: a kernel that smoothens an original image by shifting a median filter over it.
7) AHE: adaptive histogram equilisation is applid to increase the overall entropy in an original image.
8) Bitmap compression: this function attempts to cut down on storage by compressing the original image on a bitmap level.
9) Edge images: this function detects the edges of objects in the original image.
10) Salt & Pepper noise: same as Gaussian noise but with added noise values o either 0 or 255 (black or white).
11) “Grid” noise: COMMENT: input from marlena


### Data Distillation
COMMENT: input from miguel



## Execution Phase

### Model training

- How did we implement the training procedure?
- Pick up on differences in length and effort in training processes.


## Evaluation Phase

### Results

- Testing of models
- How many epochs
- Include plots/visualisations
- Describe accurary/error

### Comparison of efficiency of different models

- Which model is performing better?


### Interpretation

- Why do we think this is the case?

________________________________________________

ConvNext: aka ConvNets vs Transformers 

- Take improvements from transformers and implement into a convnet 
- They are competitive with transformers in tasks of image segmentation and object detection


Paraphrasing form paper 'A ConvNet for the 2020's':

- They start praising that convnets drove the deepnet renaissance in 2012
- All made networks from that time for image recognition used convolutions 
- Convolutions are not shifting (translational) variant but shiting equivatriant
    - It takes spatial pooling layers to make it approximately equivariant (have objects be in the same spot)
- Transformers (from 2017) have shown some application on image generation or image understanding
- The transformer architecture is usable out of the box for images
- The solution space of transformers is larger than for CNNs, due to their image splitting for sequences and not being translation equivariant    
    - The transformer architecture is also more flexible than CNNs
- Transformers are dependent on a lot of training data augmentation
- Splitting a high quality image for a transformer would take a lot of memory and computational space, specially when splitting sequences into 16x16px subimages. Therefore newer architectures like the swin transformer were introduced that use sliding window approaches, almost like ConvNets too 

Architecture: Start with a ResNet50 

- AdamW
- More epochs
- Heavy data augmentation
- Regularization

'Macro design'
- Stage ratio (how many blocks in each stage)
- Larger strides, like non-overlapping patches in transformers

- Depthwise convolutions
- Widening network

- Inverted bottleneck: hidden dimension of the MLP block is four times wider than the input dimension

- Larger kernel sizes: global receptive fields are bigger (but didnt really improve performance, but its proven that the opposite does worse)

'Micro design'
- relu -> gelu
- Less activation functions
- Fewer normalizations (like transformers), no batch normalization, but layer normalization (really something for RNNs, not CNNs)
- Seperate downsampling layers

Results -> better than the swin transformer

'Modern ConvNets scale too in terms of size in data'

- ConvNets can compete in semtic segmentation and object detection tasks
- Questions for the future: **Are other architectures mordernizable too?**

________________________________________________

### Some notes about energy consumption

#### About large models and consumption

" OpenAI trained its GPT-3 model on 45 terabytes of data. To train the final version of MegatronLM, a language model similar to but smaller than GPT-3, Nvidia ran 512 V100 GPUs over nine days.

A single V100 GPU can consume between 250 and 300 watts. If we assume 250 watts, then 512 V100 GPUS consumes 128,000 watts, or 128 kilowatts (kW). Running for nine days means the MegatronLM's training cost 27,648 kilowatt hours (kWh).

The average household uses 10,649 kWh annually, according to the U.S. Energy Information Administration. Therefore, training the final version of MegatronLM used almost the amount of energy three homes use in a year. " (https://www.techtarget.com/searchenterpriseai/feature/Energy-consumption-of-AI-poses-environmental-problems#:~:text=AI%20energy%20consumption%20during%20training&text=A%20single%20V100%20GPU%20can,27%2C648%20kilowatt%20hours%20(kWh).)

- Large ML models for broad tasks, like GPT-3 and MegatronLM, consume a lot of energy
- The consumption is almost the same as the average household 3 times

#### State of the art training, servers and CO2

- Most big tech companies have a state of the art training and massive servers that run 24/7 
- (Insert stats about the energy consumption of the servers, and why they have to train and infer on a 24/7 basis) 
- Energy consumption will scale with the demand for AI, not only for tranining, but also building the supercomputers, collecting and storing the data. (Gerry McGovern, author of the book World Wide Waste.)



#### Arguments to be made about usability vs energy consumption

- Large models obviously cover more usability than smaller models
    - GTP-3 can do things that it was not even designed to do
    - It can ..  past text generation and completion, solving math problems, translating text, describing objects, etc. with the proper prompt engineering
    - But just infereing with the model is very computationally expensive

- Smaller models can be used for very specific tasks
    - Like we saw in our project, a small ConvNet can still relatively perform well on the task of predicting the class of a bird
    - Infereing is very cheap and takes very little time
    - If we think ahead to the use cases and physical objects with restraint memory and a limited amount of time, we can use smaller models instead of bigger ones to solve niche problems
        - Take for example a safari camera that also can predict the class of a birds

#### About energy consumption and the future of machine learning 

- Without getting too political, rising prices in energy will also affect the electric infrastructure without a doubt, and seeking cheaper training options will be more desirable in the future
- (Insert info about rising prices in energy and the number of new cloud computing users / ML engineers) 
