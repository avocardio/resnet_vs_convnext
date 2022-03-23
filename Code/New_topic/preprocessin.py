
# - read from_from_slices
# - split dataset
# - preprocessing steps + augmentation
# - batch, prefetch

import tensorflow as tf
import numpy as np

"""
Helper Functions
------------------------------------------------------------
"""

def dataset_split(ds, ds_size, train_prop=0.7, val_prop=0.15, test_prop=0.15):
    """
    Costum dataset split into training, validation and test data

    Parameters:
    ----------
        ds : tensorflow dataset
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


def create_and_split(data):
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
    # create tf dataset by converting to tensorflow slices
    ds = tf.data.Dataset.from_tensor_slices(data)

    # get dataset size
    ds_size = sum(1 for _ in ds)

    train_ds, val_ds, test_ds = dataset_split(ds, ds_size)

    return train_ds, val_ds, test_ds

def  resize_and_rescale(ds):
    """
    Resizes and rescales dataset.

    Parameters:
    -----------
        ds : tf.data.Dataset
            the dataset to resize and rescale
    
    Returns:
    -------
        the modified dataset
    """
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = (image / 255.0)
    return image, label


def data_augmentation(ds):
    """
    Performs basic data augmentation.

    Parameters:
    -----------
        ds : tensorflow dataset

    Returns:
    --------
        augmented dataset
    """
    image, label = ds
    image, label = resize_and_rescale(image, label)
    image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 6, IMG_SIZE + 6)
    # Make a new seed.
    new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
    # Random crop back to the original size.
    image = tf.image.stateless_random_crop(
        image, size=[IMG_SIZE, IMG_SIZE, 3], seed=seed)
    # Random brightness.
    image = tf.image.stateless_random_brightness(
        image, max_delta=0.5, seed=new_seed)
    image = tf.clip_by_value(image, 0, 1)
    return image, label    



"""Preprocessing Pipeline"""
def prepare_data(data, batch_size):
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
    datasets = create_and_split(data)

    # modify each dataset seperately
    for ds in datasets:
        ds_list = []
        # data augmentation only on train data, else resize and rescale only
        if ds == datasets[0]:
            ds = ds.map(data_augmentation)
        else:
            ds = ds.map(resize_and_rescale)
        # shuffle, batch, prefetch
        ds = ds.shuffle(1000)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(20)
        
        ds_list.append(ds)

    return ds_list

# create and preprocess the datasets
ds_list = prepare_data(data, 64)
train_ds = ds_list[0]
val_ds = ds_list[1]
test_ds = ds_list[2]


"""Preprocessing Pipeline"""
def prepare_data(data, batch_size):
  """
  Data pipeline for quickdraw dataset:
  convert into tf dataset with dtype float32,
  resize images and normalize input
  shuffle, batch, prefetch

  Parameters:
  ----------
    quickdraw_data : np.ndarray
      dataset to preprocess
    batch_size : int
      which batch size to use

  Returns:
  -------
    quickdraw_data : tf.data.Dataset
      the resulting preprocessed dataset
  """
  # convert from uint8 to float32 dtype
  quickdraw_data = quickdraw_data.astype('float32')
  # expand dimensions
  #quickdraw_data = quickdraw_data.map(lambda img: (tf.expand_dims(img, -1)))
  # resize the images from 1x784 to 28x28x1
  quickdraw_data = quickdraw_data.reshape(quickdraw_data.shape[0], 28, 28, 1)
  # input normalization
  quickdraw_data = (quickdraw_data/127.5) - 1.
  # convert to tensorflow slices
  quickdraw_data = tf.data.Dataset.from_tensor_slices(quickdraw_data)