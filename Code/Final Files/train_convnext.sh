#!/bin/bash

# import package tfadams AdamW

# train convnext
echo "  train convnext"
python scripts/train_models.py --convnext $1
myvar="$1"
echo "$myvar"
