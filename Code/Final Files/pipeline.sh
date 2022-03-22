#!/bin/bash
# overall pipeline for loading and preprocessing the data, running the models and and applying the GPU energy analysis

echo "loading data"
scripts/load_data.sh
echo "preprocessing for resnet"
scripts/preprocessing_resnet.sh
echo "preprocessing for convnext"
scripts/preprocessing_convnext.sh
echo "training resnet"
scripts/train_resnet.sh
echo "training convnext"
scripts/train_convnext.sh
echo "testing resnet"
scripts/test_resnet.sh
echo "testing convnext"
scripts/test_convnext.sh
echo "gpu analysis for resnet"
scripts/gpu_resnet.sh
echo "gpu analysis for convnext"
scripts/gpu_convnext.sh