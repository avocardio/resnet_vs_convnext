""" Pipelines for loading models and testing on test set. """

# imports
import argparse
import tensorflow as tf
import numpy as np
from preprocessing import load_test_set
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# setting up CLI
parser = argparse.ArgumentParser(description = "Testing models")
parser.add_argument("-r", "--resnet", action = "store_true", help = "starts testing for resnet")
parser.add_argument("-c", "--convnext", action = "store_true", help = "starts testing for convnext")
parser.add_argument("-b", "--both", action = "store_true", help = "starts testing for both")

args = parser.parse_args()

print("loading test set...")

# Load test set via preprocessing.py
test_set = load_test_set()

# Only test resnet
if args.resnet:
    if os.path.exists('../../Data/Models/resnet50'):
        print("loading resnet50...")
        model = tf.keras.models.load_model('../../Data/Models/resnet50')
        print("\n")
        loss, acc = model.evaluate(test_set, verbose=2)
        print('Loaded ResNet-50, accuracy: {:5.2f}%'.format(100 * acc))   
        print("\n")

# Only test convnext
if args.convnext:
    if os.path.exists('../../Data/Models/convnext'):
        print("loading convnext...")
        model = tf.keras.models.load_model('../../Data/Models/convnext')
        print("\n")
        loss, acc = model.evaluate(test_set, verbose=2)
        print('Loaded ResNet-50, accuracy: {:5.2f}%'.format(100 * acc))   
        print("\n")

# Test both if fragments are saved in the models folder
if args.both:
    if os.path.exists('../../Data/Models/resnet50'):
        print("loading resnet50...")
        model = tf.keras.models.load_model('../../Data/Models/resnet50')
        print("\n")
        loss, acc = model.evaluate(test_set, verbose=1)
        print('Loaded ResNet-50, accuracy: {:5.2f}%'.format(100 * acc))  
        print("\n")
    else:
        print("ResNet model not trained or saved.\n")
    if os.path.exists('../../Data/Models/convnext'):
        print("loading convnext...")
        model = tf.keras.models.load_model('../../Data/Models/convnext')
        print("\n")
        loss, acc = model.evaluate(test_set, verbose=1)
        print('Loaded ResNet-50, accuracy: {:5.2f}%'.format(100 * acc))  
        print("\n")
    else:
        print("ConvNext model not trained or saved.")