# python final.py

import os, glob, cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import keras
from keras.layers import Input
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications import DenseNet121, InceptionV3, VGG16, VGG19
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import SGD
from keras.utils import multi_gpu_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras import backend as K


os.environ["CUDA_VISIBLE_DEVICES"] = "7"



def distance(x1, y1, x2, y2):
    result = math.sqrt( math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))
    return result

def recall(y_target, y_pred): # sensitivity, recall
    # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max 이외 가장자리를 깎아 낸다
    # round : 반올림한다
    y_target_yn = K.round(K.clip(y_target, 0, 1)) # 실제값을 0(Negative) 또는 1(Positive)로 설정한다
    y_pred_yn = K.round(K.clip(y_pred, 0, 1)) # 예측값을 0(Negative) 또는 1(Positive)로 설정한다

    # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다
    count_true_positive = K.sum(y_target_yn * y_pred_yn) 

    # (True Positive + False Negative) = 실제 값이 1(Positive) 전체
    count_true_positive_false_negative = K.sum(y_target_yn)

    # Recall =  (True Positive) / (True Positive + False Negative)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    recall = count_true_positive / (count_true_positive_false_negative + K.epsilon())

    # return a single tensor value
    return recall

def img_crop_step1(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray,180,255, cv2.THRESH_BINARY_INV)
    
    k=np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((15,15), np.uint8))
    
    contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    for ii, cnt in enumerate(contours):
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        d=distance(int(img.shape[1]/2), int(img.shape[0]/2), cx, cy)
    
        if ii==0:
            min_dist = (ii, d, cnt, (cx, cy))
        else:
            if d<min_dist[1] and 3000<cv2.contourArea(cnt)<1000000:
                min_dist=(ii, d, cnt, (cx, cy))
                
    crop_sizex = 700
    crop_sizey = 700
    cx, cy = min_dist[3][0], min_dist[3][1]
    img_crop = img[cy-crop_sizey:cy+crop_sizey,cx-crop_sizex:cx+crop_sizex]
    
    return img_crop

def img_crop_step2(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray,180,255, cv2.THRESH_BINARY_INV)
    
    k=np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((15,15), np.uint8))
    
    contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for ii, cnt in enumerate(contours):
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        d=distance(int(img.shape[1]/2), int(img.shape[0]/2), cx, cy)
        if ii==0:
            min_dist = (ii, d, cnt, (cx, cy))
        else:
            if d<min_dist[1] and abs(int(img.shape[1]/2)-cx)<100:
                min_dist=(ii, d, cnt, (cx, cy))

    crop_sizex = 256
    crop_sizey = 512
    cx, cy = min_dist[3][0], min_dist[3][1]
    img_crop = img[cy-crop_sizey if cy-crop_sizey>0 else 0:cy+crop_sizey if cy+crop_sizey>0 else 0,cx-crop_sizex if cx-crop_sizex>0 else 0:cx+crop_sizex if cx+crop_sizex>0 else 0]
    
    return img_crop


if __name__ == '__main__':
    
    PATH = './example/'
    
    model_name_class = 'class_exp2_vgg16'
    
    print('*'*30)
    print('Load data')
    print('*'*30)

    row, col = 224,224
    img_path_tmp = sorted(glob.glob(PATH+'/*'))
    imgs = []
    for path in img_path_tmp:
#         print(path)
        name = path.split('/')[-1]
        img = cv2.imread(path)
        crop_img=img_crop_step1(img)
        crop_img2=img_crop_step2(crop_img)
#         img_wb = (crop_img2*1.0 / crop_img2.mean(axis=(0,1)))*255
        img_f = cv2.resize(crop_img2, (row, col))
        imgs.append(img_f)
        
    imgs = np.array(imgs)
    print('data shape: ', imgs.shape)
    
    test_x=preprocess_input(imgs)
    
    model_class = keras.models.load_model('./{}.h5'.format(model_name_class), custom_objects={'recall':recall})
    print('*'*30)
    print('classification model run...')
    print('*'*30)
    ypred_class = model_class.predict(test_x)
    
    print('*'*30)
    print('Result')
    print('*'*30)
    pred_label=[]
    for i,v in enumerate(ypred_class):
        pre_ans=v.argmax()
        pred_label.append((pre_ans+1)*10)

    for j in range(len(test_x)):
        print('case{}'.format(j))
        print('classification_prediction : ', pred_label[j])