TODO:

- Try autoencoder for 10 images per class
- Confusion matrix
- Write on Documentation file in parallel
- Try Eigenfaces on bird dataset
- Transfer documentation to Overleaf
- Think about how to store and provide the dataset to tutors




# Documentation 

The project falls under the classification tasks of Machine Learning. In particular, it involves multi-label classification of input examples. We use a Lightweight* Convolutational Neural Network to classify images of birds from the [Bird Species](https://www.kaggle.com/gpiosenka/100-bird-species) dataset. As the next step of your project, we attempt to deploy 2 types of preprocessing techniques, namely data augmentation and *data distillation*. Data distillation is a fairly new technique used to improve the performance of a model by reducing required memory and computations. The idea is to use a 'distilled' dataset, containing a somewhat synthesized subset of the original dataset, with one distilled image per class (i.e. one image per bird). The model is then trained on the distilled dataset, and tested on the real dataset. With our results we want to check, if the models can be trained more efficiently with a subset of the original data but reaching the same test accuracy when compared with training and testing on the whole, original dataset.


## Preparation Phase

### Dataset

What does our dataset contain and why did we decide to use it?

The original dataset contains 385 different types/classes of birds, and each image is a 224x224x3 coloured pixel image. To save time and energy, we decided to bring down the number of types/classes to 10. These 10 types/classes comprise of the bird species of which there exist at least 200 images in the original dataset each. Thereby, while reducing the overall amount of data, we assure that the model is seeing enough data during the training phase in order to be able to generalise and perform well.


### Model Design Decisions

1) VGG16 Model or VGGNet - Very Deep Convolutional Neural Network
====================================================================
(https://viso.ai/deep-learning/vgg-very-deep-convolutional-networks/)
https://medium.com/@mygreatlearning/what-is-vgg16-introduction-to-vgg16-f2d63849f615#:~:text=VGG16%20is%20a%20simple%20and,visual%20object%20recognition%20software%20research.


As the next step, we are presenting the preprocessed data to a VGG16 Model. ‘VGG’ is short for 'Visual Geometry Group', who are the developers of this specific CNN architecture. The ‘16’ refers to the number of layers that are being used in the network. VGGs are inspired by the basic convolution neural network design however, larger kernels that are usually integrated in CNNs are replaced with multiple smaller 3x3 kernels. In total, it comprises of 13 convolutional layers and three fully connected layers. While VGGs can differentiate "1000 objects", with a "total of around 138 million parameters", VGG16 "is a [very] extensive network. As a benchmark, it is said that "[t]he VGG16 model achieves almost 92.7% top-5 test accuracy in ImageNet."


- "Convolutional Layers: VGG’s convolutional layers leverage a minimal receptive field, i.e., 3×3, the smallest possible size that still captures up/down and left/right. Moreover, there are also 1×1 convolution filters acting as a linear transformation of the input. This is followed by a ReLU unit, which is a huge innovation from AlexNet that reduces training time."
- "The convolution stride is fixed at 1 pixel to keep the spatial resolution preserved after convolution (stride is the number of pixel shifts over the input matrix)."
- "Hidden Layers: All the hidden layers in the VGG network use ReLU. VGG does not usually leverage Local Response Normalization (LRN) as it increases memory consumption and training time. Moreover, it makes no improvements to overall accuracy."
- "Fully-Connected Layers: The VGGNet has three fully connected layers. Out of the three layers, the first two have 4096 channels each, and the third has 1000 channels, 1 for each class."
- "The number of filters that we can use doubles on every step or through every stack of the convolution layer. This is a major principle used to design the architecture of the VGG16 network. One of the crucial downsides of the VGG16 network is that it is a huge network, which means that it takes more time to train its parameters."

2) Vanilla CNN - Convolutional Neural Network
==============================================


3) Ultra Lightweight Convolutional Neural Network
=================================================



- Which evaluation metrics do we use on our model(s)
- How do we compare them

## Execution Phase

### Pre Processing

- Explain how we extracted top ten classes
- What pre-processing steps did we use on the images and why
- Maybe show some examples of data before and after pre-processing?
- Cross validation etc

### Data Augmentation

- 


### Model training

- How did we implement the training procedure?


## Evaluation Phase

### Results

- Testing of models
- How many epochs
- Include plots/visualisations
- Describe accurary/error

### Comparison of models

- Which model is performing better?


### Interpretation

- Why do we think this is the case?




### Model Design Decisions
### Model Distillation
or
### Data Distillation