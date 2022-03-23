# train pipelines

# imports
import argparse

# setting up CLI
parser = argparse.ArgumentParser(description = "Training models")
parser.add_argument("-r", "--resnet", action = "store_true", help = "starts training for resnet")
parser.add_argument("-c", "--convnext", action = "store_true", help = "starts training for convnext")

args = parser.parse_args()

if args.resnet:
    print("training function for resnet")
    
if args.convnext:
    print("training function for convnext")