#!/bin/bash

echo "Reading GPU..."

nvidia-smi --query-gpu=index,timestamp,power.draw,clocks.sm,clocks.mem,clocks.gr --format=csv -l 10 -f ../../Data/GPU/GPU-stats.

sleep 20

./train_resnet.sh