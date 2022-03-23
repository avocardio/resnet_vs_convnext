# test pipelines

# imports
import argparse

# setting up CLI
parser = argparse.ArgumentParser(description = "Testing models")
parser.add_argument("-r", "--resnet", action = "store_true", help = "starts testing for resnet")
parser.add_argument("-c", "--convnext", action = "store_true", help = "starts testing for convnext")

args = parser.parse_args()

if args.resnet:
    print("testing function for resnet")
    
if args.convnext:
    print("testing function for convnext")