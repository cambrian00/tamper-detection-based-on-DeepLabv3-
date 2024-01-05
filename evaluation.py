#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 22:01:29 2019

@author: dell
"""
from glob import glob
import skimage.io as io
from sklearn.metrics import f1_score
import skimage.transform as trans
import numpy as np
import cv2
import math

def eval_f1_score(path1,path2,file_end = ".jpg",dso = False):
    predict_image_list = glob(path1+"*")
    number = len(predict_image_list)
    print(number)
    all_f1score = 0
    count = 1
    for image in predict_image_list:
        image_name = image.split("/")[-1]
        image_first = image_name.split(".")[0]
        image_first = image_first.replace("_label","")
        mask_name = path2 + image_first + file_end
        count_TP = 0 
        count_FP = 0 
        count_TN = 0 
        count_FN = 0
        label = io.imread(image)
        if(dso == True):
            label = 255 - label[:,:,0]
#        label = cv2.resize(label,(256,256),interpolation=cv2.INTER_NEAREST)
        mask = io.imread(mask_name)
        
        w,h = label.shape
#        print(w,h)
        for i in range(w):
            for j in range(h):
                if label[i][j] == 255 and mask[i][j] == 255:
                    count_TP += 1
                elif label[i][j] == 255 and mask[i][j] == 0:
                    count_FN += 1
                elif label[i][j] == 0 and mask[i][j] == 255:
                    count_FP += 1
                else:
                    count_TN += 1
#        count_TP = count(label == mask and label == 255)
#        count_FP = count(label != mask and label == 0)
#        count_FN = count(label != mask and label == 255)
#        print(count_TP,count_FN,count_FP)
        f1score = (2*count_TP)/(2*count_TP + count_FN + count_FP)
#        f1score = f1_score(img/255,mask/255,average='weighted')
        all_f1score = all_f1score + f1score
#        if(count % 100 == 0):
#        print(image_name," ",count," ",f1score," ",all_f1score/count)
        count = count + 1
        
    return all_f1score/number

def eval_f1_mcc_score(path1,path2,file_end = ".jpg",dso = False):
    predict_image_list = glob(path1+"*")
    number = len(predict_image_list)
    print(number)
    all_f1score = 0
    all_mcc = 0
    count = 1
    for image in predict_image_list:
        image_name = image.split("/")[-1]
        image_first = image_name.split(".")[0]
        image_first = image_first.replace("_label","")
        mask_name = path2 + image_first + file_end
        count_TP = 0 
        count_FP = 0 
        count_TN = 0 
        count_FN = 0
        label = io.imread(image)
        if(dso == True):
            label = 255 - label[:,:,0]
#        label = cv2.resize(label,(256,256),interpolation=cv2.INTER_NEAREST)
        mask = io.imread(mask_name)
        
        w,h = label.shape
#        print(w,h)
        for i in range(w):
            for j in range(h):
                if label[i][j] == 255 and mask[i][j] == 255:
                    count_TP += 1.0
                elif label[i][j] == 255 and mask[i][j] == 0:
                    count_FN += 1.0
                elif label[i][j] == 0 and mask[i][j] == 255:
                    count_FP += 1.0
                else:
                    count_TN += 1.0
#        count_TP = count(label == mask and label == 255)
#        count_FP = count(label != mask and label == 0)
#        count_FN = count(label != mask and label == 255)
#        print(count_TP,count_FN,count_FP)
        f1score = (2*count_TP)/(2*count_TP + count_FN + count_FP)
        
        mcc_fenzi  = (count_TP * count_TN) - (count_FP * count_FN)
#        try:
        mcc_fenmu = math.sqrt((count_TP+count_FP)*(count_TP+count_FN)*(count_TN+count_FP)*(count_TN+count_FN))
#        if(mcc_fenzi <= 0):
#            mcc = 0
#        else:
        mcc = mcc_fenzi / mcc_fenmu
#        except:
#            print(mask_name,count_TP,count_FN,count_FP,count_TN)

            
        all_f1score = all_f1score + f1score
        all_mcc = all_mcc + mcc
#        if(count % 100 == 0):
#        print(image_name," ",count," ",f1score," ",all_f1score/count)
        count = count + 1
#    return all_f1score/number   
    return all_f1score/number,all_mcc/number


             