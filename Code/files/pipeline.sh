#!/usr/bin/env python
# overall pipeline for loading and preprocessing the data, running the models and and applying the GPU energy analysis

echo "training resnet"
./train_resnet.sh
echo "training convnext"
./train_convnext.sh
echo "testing models"
./test_models.sh