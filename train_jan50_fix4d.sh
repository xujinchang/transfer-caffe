#!/bin/bash

./build/tools/caffe train \
    -solver models/JAN/resnet/solver.prototxt \
    -weights models/JAN/resnet/SE-ResNet-50.caffemodel \
    -gpu 0,1 \
    2>&1 | tee JAN-SE-50_result.txt
