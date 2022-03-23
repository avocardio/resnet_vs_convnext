# preprocessing pipelines

print("this is the python script for preprocessing")

import tensorflow as tf
import numpy as np

DIR_TRAIN = '../../Data/Use/Train/'
DIR_TEST = '../../Data/Use/Test/'
DIR_VALID = '../../Data/Use/Validation/'

def preprocessing():
    """
    Preprocessing pipeline for the dataset.
    """
    # Create an image generator, augment the training data
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255, # Normalize
        rotation_range=40, # Random rotation
        width_shift_range=0.2, # Random horizontal offset
        height_shift_range=0.2, # Random vertical offset
        shear_range=0.2, # Random cropping 
        zoom_range=0.2, # Random zoom
        horizontal_flip=True, # Flip horizontally
        zca_epsilon=1e-6, # ZCA whitening
        brightness_range=[0.5, 1.5], # Random brightness
        fill_mode='nearest') # Points outside the input are filled according to: aaaaaaaa|abcd|dddddddd

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    # Find files from directory
    train_data = train_datagen.flow_from_directory(
        DIR_TRAIN,
        target_size=(224, 224),
        batch_size=32,
        color_mode='rgb',
        class_mode='categorical',
        shuffle=True)

    test_data = test_datagen.flow_from_directory(
        DIR_TEST,
        target_size=(224, 224),
        batch_size=32,
        color_mode='rgb',
        class_mode='categorical',
        shuffle=True)

    valid_data = valid_datagen.flow_from_directory(
        DIR_VALID,
        target_size=(224, 224),
        batch_size=32,
        color_mode='rgb',
        class_mode='categorical',
        shuffle=True)

    return train_data, test_data, valid_data