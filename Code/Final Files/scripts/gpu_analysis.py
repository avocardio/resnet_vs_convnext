# gpu analysis

# imports
import argparse

# setting up CLI
parser = argparse.ArgumentParser(description = "Testing models")
parser.add_argument("-r", "--resnet", action = "store_true", help = "starts gpu analysis on resnet")
parser.add_argument("-c", "--convnext", action = "store_true", help = "starts gpu analysis on convnext")

args = parser.parse_args()

if args.resnet:
    print("gpu analysis resnet")
    
if args.convnext:
    print("gpu analysis on convnext")