#!/bin/bash
# overall pipeline for loading and preprocessing the data, running the models and and applying the GPU energy analysis

echo "loading data"
./load_data.sh
echo "preprocessing for resnet"
./preprocessing_resnet.sh
echo "preprocessing for convnext"
./preprocessing_convnext.sh
echo "training resnet"
./train_resnet.sh
echo "training convnext"
./train_convnext.sh
echo "testing resnet"
./test_resnet.sh
echo "testing convnext"
./test_convnext.sh
echo "gpu analysis for resnet"
./gpu_resnet.sh
echo "gpu analysis for convnext"
./gpu_convnext.sh