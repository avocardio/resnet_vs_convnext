# resnet implementation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import requests

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Blocks for the ResNet50

# [1x1, 64]
# [3x3, 64]  x 3 
# [1x1, 256]

# [1x1, 128]
# [3x3, 128] x 4
# [1x1, 512]

# [1x1, 256]
# [3x3, 256] x 6
# [1x1, 1024]

# [1x1, 512]
# [3x3, 512] x 3
# [1x1, 2048]

class Block(tf.keras.Model):
    def __init__(self, input_channels, output_channels, identity_block=False, identity_strides=1):
        super(Block, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(input_channels, (1, 1), strides=identity_strides)
        self.conv2 = tf.keras.layers.Conv2D(input_channels, (3, 3), padding='same')
        self.conv3 = tf.keras.layers.Conv2D(output_channels, (1, 1), padding='same')

        if identity_block is False:
            self.shortcut = tf.keras.layers.Conv2D(output_channels, (1, 1), strides=identity_strides, padding='same')

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.bn4 = tf.keras.layers.BatchNormalization()

    def call(self, input):

        x = self.conv1(input)
        x = self.bn1(x)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = tf.nn.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if hasattr(self, 'shortcut'):
            input = self.shortcut(input)
            input = self.bn4(input)
            
        y = tf.keras.layers.Add()([x, input])
        y = tf.nn.relu(y)

        return y

# ResNet50

# Base layer
# ConvBlock
# IdentityBlock x2
# ConvBlock
# IdentityBlock x3
# ConvBlock
# IdentityBlock x5
# ConvBlock
# IdentityBlock x2
# AvgPool
# FC

class ResNet50(tf.keras.Model):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(input_shape=(224,224,3), filters=64, kernel_size=(7, 7), strides=2, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.maxpool1 = tf.keras.layers.MaxPool2D((3, 3), strides=2, padding='same')

        self.block1_1 = Block(64, 256, False, 1)
        self.block1_2 = Block(64, 256, True, 1)
        self.block1_3 = Block(64, 256, True, 1)

        self.block2_1 = Block(128, 512, False, 2)
        self.block2_2 = Block(128, 512, True, 1)
        self.block2_3 = Block(128, 512, True, 1)
        self.block2_4 = Block(128, 512, True, 1)

        self.block3_1 = Block(256, 1024, False, 2)
        self.block3_2 = Block(256, 1024, True, 1)
        self.block3_3 = Block(256, 1024, True, 1)
        self.block3_4 = Block(256, 1024, True, 1)
        self.block3_5 = Block(256, 1024, True, 1)
        self.block3_6 = Block(256, 1024, True, 1)

        self.block4_1 = Block(512, 2048, False, 2)
        self.block4_2 = Block(512, 2048, True, 1)
        self.block4_3 = Block(512, 2048, True, 1)

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        x = self.maxpool1(x)

        x = self.block1_1(x)
        x = self.block1_2(x)
        x = self.block1_3(x)

        x = self.block2_1(x)
        x = self.block2_2(x)
        x = self.block2_3(x)
        x = self.block2_4(x)

        x = self.block3_1(x)
        x = self.block3_2(x)
        x = self.block3_3(x)
        x = self.block3_4(x)
        x = self.block3_5(x)
        x = self.block3_6(x)

        x = self.block4_1(x)
        x = self.block4_2(x)
        x = self.block4_3(x)

        x = self.avgpool(x)
        x = self.fc(x)

        return x

