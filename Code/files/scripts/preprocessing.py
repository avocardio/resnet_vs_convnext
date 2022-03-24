# preprocessing pipelines

import tensorflow as tf
import numpy as np

DIR_TRAIN = '../../Data/Use/Train/'
DIR_TEST = '../../Data/Use/Test/'
DIR_VALID = '../../Data/Use/Validation/'

def preprocessing_resnet():
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

def preprocessing_convnext():
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
    train_data = MixupImageDataGenerator(
        generator=train_datagen,
        directory=DIR_TRAIN,
        batch_size=32,
        img_height=224,
        img_width=224)

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


    return None

# Class for mix-up implementation

class MixupImageDataGenerator():
# inspired by https://medium.com/swlh/how-to-do-mixup-training-from-image-files-in-keras-fe1e1c1e6da6

    def __init__(self, generator, directory, batch_size, img_height, img_width, alpha=0.2, subset=None):
        """Constructor for mixup image data generator.
        Arguments:
            generator {object} -- An instance of Keras ImageDataGenerator.
            directory {str} -- Image directory.
            batch_size {int} -- Batch size.
            img_height {int} -- Image height in pixels.
            img_width {int} -- Image width in pixels.
        Keyword Arguments:
            alpha {float} -- Mixup beta distribution alpha parameter. (default: {0.2})
            subset {str} -- 'training' or 'validation' if validation_split is specified in
            `generator` (ImageDataGenerator).(default: {None})
        """

        self.batch_index = 0
        self.batch_size = batch_size
        self.alpha = alpha

        # First iterator yielding tuples of (x, y)
        self.generator1 = generator.flow_from_directory(directory,
                                                        target_size=(
                                                            img_height, img_width),
                                                        class_mode="categorical",
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        subset=subset)

        # Second iterator yielding tuples of (x, y)
        self.generator2 = generator.flow_from_directory(directory,
                                                        target_size=(
                                                            img_height, img_width),
                                                        class_mode="categorical",
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        subset=subset)

        # Number of images across all classes in image directory.
        self.n = self.generator1.samples

    def reset_index(self):
        """Reset the generator indexes array.
        """

        self.generator1._set_index_array()
        self.generator2._set_index_array()

    def on_epoch_end(self):
        self.reset_index()

    def reset(self):
        self.batch_index = 0

    def __len__(self):
        # round up
        return (self.n + self.batch_size - 1) // self.batch_size

    def get_steps_per_epoch(self):
        """Get number of steps per epoch based on batch size and
        number of images.
        Returns:
            int -- steps per epoch.
        """

        return self.n // self.batch_size

    def __next__(self):
        """Get next batch input/output pair.
        Returns:
            tuple -- batch of input/output pair, (inputs, outputs).
        """

        if self.batch_index == 0:
            self.reset_index()

        current_index = (self.batch_index * self.batch_size) % self.n
        if self.n > current_index + self.batch_size:
            self.batch_index += 1
        else:
            self.batch_index = 0

        # random sample the lambda value from beta distribution.
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)

        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        # Get a pair of inputs and outputs from two iterators.
        X1, y1 = self.generator1.next()
        X2, y2 = self.generator2.next()

        # Perform the mixup.
        X = X1 * X_l + X2 * (1 - X_l)
        y = y1 * y_l + y2 * (1 - y_l)
        return X, y

    def __iter__(self):
        while True:
            yield next(self)