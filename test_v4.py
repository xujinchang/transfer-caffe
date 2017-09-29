#!/usr/bin/env python
import numpy as np
import time
import os
import json
import sys
import socket
import logging
import matplotlib.pyplot as plt
from PIL import Image
import cv2
sys.path.insert(0,'./python')
import caffe
import random
import copy
import pickle

caffe.set_mode_gpu()
caffe.set_device(1)
#MODEL_DEF = "/home/xujinchang/share/caffe-center-loss/attri_dg/v3/deploy_inception_v3_our.prototxt"
#MODEL_DEF = "/home/xujinchang/share/caffe-center-loss/attri_dg/resnet269_v2/deploy_resnet269_v2.prototxt"
MODEL_DEF = "/home/xujinchang/share/project/taskcv/transfer-caffe/models/JAN/resnet/deploy_JAN_SE50.prototxt"
MODEL_PATH = "/home/xujinchang/taskcv_model/stage2_iter_1000.caffemodel"
MODEL_DEF1 = "/home/xujinchang/share/project/taskcv/transfer-caffe/models/JAN/resnext/depoly_SE-ResNeXt-50.prototxt"                                                                         
MODEL_PATH1 = "/home/xujinchang/taskcv_model/stage2_renext_iter_1500.caffemodel" 
MODEL_DEF2 = "/home/xujinchang/share/project/taskcv/transfer-caffe/deploy_inception_v4.prototxt"
MODEL_PATH2 = "/home/xujinchang/share/project/taskcv/transfer-caffe/inception_v4_iter_26000.caffemodel"
mean = np.array((104., 117., 123.), dtype=np.float32)
mean1 = np.array((128., 128., 128.), dtype=np.float32)

def predict(image,the_net,SIZE):
    inputs = []
    try:
        tmp_input = image
        tmp_input = cv2.resize(tmp_input,(SIZE,SIZE))
        #tmp_input = tmp_input[13:13+224,13:13+224]
        tmp_input = np.subtract(tmp_input,mean)
        #tmp_input = tmp_input*0.0078125
        tmp_input = tmp_input.transpose((2, 0, 1))
        tmp_input = np.require(tmp_input, dtype=np.float32)
    except Exception as e:
        raise Exception("Image damaged or illegal file format")
        return
    the_net.blobs['data'].reshape(1, *tmp_input.shape)
    the_net.reshape()
    the_net.blobs['data'].data[...] = tmp_input
    the_net.forward()
    scores = the_net.blobs['prob'].data[0]
    return copy.deepcopy(scores)

def predict_v4(image,the_net,SIZE):
    inputs = []
    try:
        tmp_input = image
        tmp_input = cv2.resize(tmp_input,(SIZE,SIZE))
        #tmp_input = tmp_input[13:13+224,13:13+224]
        tmp_input = np.subtract(tmp_input,mean1)
        tmp_input = tmp_input*0.0078125
        tmp_input = tmp_input.transpose((2, 0, 1))
        tmp_input = np.require(tmp_input, dtype=np.float32)
    except Exception as e:
        raise Exception("Image damaged or illegal file format")
        return
    the_net.blobs['data'].reshape(1, *tmp_input.shape)
    the_net.reshape()
    the_net.blobs['data'].data[...] = tmp_input
    the_net.forward()
    scores = the_net.blobs['prob'].data[0]
    return copy.deepcopy(scores)

if __name__=="__main__":
    f = open("/local/home/share/xujinchang/project/taskcv/transfer-caffe/data/VisDA2017/test_pick.txt","rb")
    net = caffe.Net(MODEL_DEF, MODEL_PATH, caffe.TEST)
    net1 = caffe.Net(MODEL_DEF1, MODEL_PATH1, caffe.TEST)
    net2 = caffe.Net(MODEL_DEF2, MODEL_PATH2, caffe.TEST)
    # dump1 = "/home/xujinchang/share/project/taskcv/transfer-caffe/SE_50/model1"
    img_label = dict()
    for line in f.readlines():
        line = line.strip().split(" ")
        img_label[line[0]] = line[1]
    count = 0
    acc = 0
    imgs = img_label.keys()
    indexs = range(len(imgs))
    random.shuffle(indexs)
    imgs = [imgs[i] for i in indexs]
    for img in imgs:
        count += 1
        if count==1:
            start_time = time.time()
        cv_img = cv2.imread(img)
        cv_img_flip = cv2.flip(cv_img, 1)
        scores1 = predict(cv_img,net,224)
        scores1_flip = predict(cv_img_flip,net,224)
        scores2 = predict(cv_img,net1,224)
        scores2_flip = predict(cv_img_flip,net1,224)
        scores3 = predict_v4(cv_img,net2,299)
        scores3_flip = predict_v4(cv_img_flip,net2,299)
        # pickle.dump(scores1, open(dump1+img.split("/")[-1], "w"))
        #scores1_flip = predict(cv_img_flip,net,256)
        scores = scores1+scores1_flip+scores2+scores2_flip+scores3+scores3_flip
        if int(scores.argmax(axis=0)) == int(img_label[img]):
            acc += 1
        print "count: ",count
    f.close()
    end_time = time.time()
    run_time = end_time - start_time

    print "run_time: ",run_time
    print "per_run_time: ",float(run_time)/count
    print "accuracy: ", float(acc)/len(imgs)
