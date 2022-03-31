#!/usr/bin/env python

# Overall pipeline for loading and preprocessing the data, running the models and testing the results

echo "training resnet"
./train_resnet.sh
echo "training convnext"
./train_convnext.sh
echo "testing models"
./test_models.sh