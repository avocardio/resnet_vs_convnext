# train pipelines

# imports
import argparse

from resnet50 import ResNet50
from convnext import ConvNeXt
from preprocessing import preprocessing_resnet, preprocessing_convnext
from tensorflow_addons.optimizers import AdamW

# setting up CLI
parser = argparse.ArgumentParser(description = "Training models")
parser.add_argument("-r", "--resnet", action = "store_true", help = "starts training for resnet")
parser.add_argument("-c", "--convnext", action = "store_true", help = "starts training for convnext")
parser.add_argument("-p", "--preprocessed", action = "store_true", help = "uses preprocessed data")

MODEL_FRAGMENTS = '../../Data/Models/'

args = parser.parse_args()

if args.resnet and args.preprocessed:
    train_data, _, valid_data = preprocessing_resnet()
    print("\nStarting training for ResNet50:")
    model = ResNet50(num_classes=400)
    model.build(input_shape=(None, 224, 224, 3))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_data, epochs=10, validation_data=valid_data)
    model.save(MODEL_FRAGMENTS + 'resnet50')
    
if args.convnext and args.preprocessed:
    train_data, _, valid_data = preprocessing_convnext()
    print("\nStarting training for ConvNeXt:")
    model = ConvNeXt(num_classes=400, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], include_top=True, drop_path_rate=0.1)
    model.build(input_shape=(None, 224, 224, 3))
    # need to check on optimizer
    model.compile(optimizer=AdamW(learning_rate=0.01, weight_decay=1e-2), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_data, epochs=10, validation_data=valid_data)
    model.save(MODEL_FRAGMENTS + 'convnext')
