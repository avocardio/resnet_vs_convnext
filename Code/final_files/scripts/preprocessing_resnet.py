# preprocessing pipeline for resnet model
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

"""
Helper Functions
------------------------------------------------------------
"""

seed = 2

def dataset_split(ds, ds_size, train_prop=0.7, val_prop=0.15, test_prop=0.15):
    """
    Costum dataset split into training, validation and test data.

    Parameters:
    ----------
        ds : tf.data.Dataset
            dataset to split
        ds_size : int
            size of the dataset
        train_prop, val_prop, test_prop : float
            split proportions

    Returns:
    -------
        the resulting train, validation and test datasets
    """

    # proportions must add up to 1
    if(train_prop + val_prop + test_prop != 1):
        return print("split sizes must sum up to 1")

    train_size = int(train_prop * ds_size)
    val_size = int(val_prop * ds_size)

    # take the respective numbers of examples, make sure the sets don't overlap
    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    datasets = (train_ds, val_ds, test_ds)

    return datasets


def create_and_split():
    """
    Creates Tensorflow dataset with tf.from_tensor_slices.
    Splits ds into training, validation and test.

    Parameter:
    ---------
        data : ###
            raw data

    Returns:
    ------
        splitted training, validation and test tf datasets
    """
    # create tf dataset and apply augmentation
    data_directory = '..\\marle\\tf_birds\\selected_birds\\train\\'
    datagen = ImageDataGenerator(rescale = 1/255)
    ds = datagen.flow_from_directory(data_directory, target_size=(224,224), color_mode='rgb', class_mode='sparse', save_to_dir='../marle/tf_birds/selected_birds/neu')

    # get dataset size
    ds_size = sum(1 for _ in ds)

    datasets = dataset_split(ds, ds_size)

    return datasets


def data_augmentation(ds):
    """
    Performs basic data augmentation.

    Parameters:
    -----------
        ds : tf.data.Dataset

    Returns:
    --------
        augmented dataset
    """
    image, label = ds
    image = tf.image.random_flip_left_right(image)
    image = tf.image.resize_with_crop_or_pad(image, 224 + 6, 224 + 6)
    # make a new seed
    new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
    # random crop back to the original size
    image = tf.image.stateless_random_crop(
        image, size=[224, 224, 3], seed=seed)
    # random brightness
    image = tf.image.stateless_random_brightness(
        image, max_delta=0.5, seed=new_seed)
    image = tf.clip_by_value(image, 0, 1)
    return image, label   



"""Preprocessing Pipeline"""
def prepare_data(batch_size):
    """
    Parameters:
    ----------
        data : tf.data.Dataset
            data to preprocess
        batch_size : int
            which batch size to use

    Returns:
    -------
        ds : tf.data.Dataset
            the resulting preprocessed dataset
    """
    # create and split train, val and test datasets
    datasets = create_and_split()

    ds_list = []

    # modify each dataset seperately
    for ds in datasets:

        image, label = ds
        image = tf.cast(image, tf.float32)
        
        # data augmentation only on train data
        #if ds == datasets[0]:
            #ds = ds.map(data_augmentation)
        
        # shuffle, batch, prefetch
        ds = ds.shuffle(1000)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(20)
        
        ds_list.append(ds)

    return ds_list


prepare_data(64)