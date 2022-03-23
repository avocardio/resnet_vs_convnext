#!/bin/bash

# create directory?

# install/download packages required for preprocessing steps (if not done via environment file)

# preprocessing steps for resnet
echo "  preprocessing resnet" 
python scripts/preprocessing.py data/... data/... --resnet # paths to original data and to file where the new data needs to be stored (if required)