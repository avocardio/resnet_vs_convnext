# train pipelines

# imports
import argparse
import sys

# setting up CLI
parser = argparse.ArgumentParser(description = "Training models")
parser.add_argument("-r", "--resnet", action = "store_true", help = "starts training for resnet")
parser.add_argument("-c", "--convnext", action = "store_true", help = "starts training for convnext")
#parser.add_argument("-o", "--output_variable", type = int, help = "output of this script")


args = parser.parse_args()

if args.resnet:
    print("training function for resnet")
    
if args.convnext:
    sys.argv[1] = 42
    print("training function for convnext")
    


# save parameters in file
# use tf.