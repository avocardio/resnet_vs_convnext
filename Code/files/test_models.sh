#!/bin/bash

# test convnext
echo "  testing models" 

# --convnext for convnext
# --resnet for resnet
# --both for both
python scripts/test_models.py --both