TODO:

- Try autoencoder for 10 images per class
- Confusion matrix
- Write on Documentation file in parallel
- Try Eigenfaces on bird dataset
- Try to implement 2018 paper
- Transfer documentation to Overleaf
- Think about how to store and provide the dataset to tutors




# Documentation 

The project falls under the classification tasks of Machine Learning. In particular, it involves multi-label classification of input examples. We use a Lightweight* Convolutational Neural Network to classify images of birds from the [Bird Species](https://www.kaggle.com/gpiosenka/100-bird-species) dataset. As the next step of your project, we attempt to deploy 2 types of preprocessing techniques, namely data augmentation and *data distillation*. Data distillation is a fairly new technique used to improve the performance of a model by reducing required memory and computations. The idea is to use a 'distilled' dataset, containing a somewhat synthesized subset of the original dataset, with one distilled image per class (i.e. one image per bird). The model is then trained on the distilled dataset, and tested on the real dataset. With our results we want to check, if data distillation is able to train and perform more efficiently than a/our Lightweight* Convolutional Network.

# STEP A: Lightweight CNN

## Preparation Phase

### Dataset

What does our dataset contain and why did we decide to use it?

The original dataset contains 385 different types/classes of birds, and each image is a 224x224x3 coloured pixel image. To save time and energy, we decided to bring down the number of types/classes to 10. These 10 types/classes comprise of the bird species of which there exist at least 200 images in the original dataset each. Thereby, while reducing the overall amount of data, we assure that the model is seeing enough data during the training phase in order to be able to generalise and perform well.

- comment JCH: say more about the dataset? reasons for using it important or leave out?

### Model Design Decisions

- Say which Lightweight CNN we used and why (vgg_cnn vs. normal_cnn)
- Which evaluation metrics do we use on our model(s)
- How do we compare them

## Execution Phase

### Pre Processing

- Explain how we extracted top ten classes
- What pre-processing steps did we use on the images and why
- Maybe show some examples of data before and after pre-processing?
- Cross validation etc

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
- Why do we think this is the case?

### Interpretation


# STEP B: Data distillation

### Dataset

- same as above


### Model Design Decisions
### Model Distillation
or
### Data Distillation