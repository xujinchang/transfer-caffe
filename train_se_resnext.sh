#!/bin/bash

./build/tools/caffe train \
    -solver models/JAN/resnext/solver.prototxt \
    -weights /home/xujinchang/taskcv_model/stage1_renext_iter_10000.caffemodel \
    -gpu 0,1,2,3 \
    2>&1 | tee JAN-SE-ResNext-50_10_20result.txt
    
#-weights models/JAN/resnet/SE-ResNet-50.caffemodel \
#-weights /home/xujinchang/taskcv_model/stage1_iter_7000.caffemodel \
