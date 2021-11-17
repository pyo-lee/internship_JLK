import os, glob, cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
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
from keras.utils import to_categorical
from keras import backend as K

def add_new_last_layer(base_model, nb_classes):
    x = base_model.layers[-2].output  
    x = GlobalAveragePooling2D()(x)
  #x = Dense(FC_SIZE, activation='relu')(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(base_model.input, predictions)
  
    return model

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

PATH = '../01_data/PPM_data_aug/'
label_folder = os.listdir(PATH)
label_li = [float(l.split('_')[-1]) for l in label_folder]

row, col = 224,224
img_path_li = list()
data_img = [[],[],[]]
data_label = [[],[],[]]

print('*'*30)
print('Load data')
print('*'*30)
for label_path in label_folder:
    label = float(label_path.split('_')[-1])
    img_path_tmp = glob.glob(PATH+label_path+'/*')
#     print('label :',label, 'number :',len(img_path_tmp))
    
#     print('triain')
    for path in img_path_tmp[:-8]:
        img = cv2.resize(cv2.imread(path), (row, col))
        data_img[0].append(img)
        data_label[0].append(label//10-1)
        
#     print('validation')
    for path in img_path_tmp[-8:-4]:
#         print(path)
        img = cv2.resize(cv2.imread(path), (row, col))
        data_img[1].append(img)
        data_label[1].append(label//10-1)
        
#     print('test')
    for path in img_path_tmp[-4:]:
#         print(path)
        img = cv2.resize(cv2.imread(path), (row, col))
        data_img[2].append(img)
        data_label[2].append(label//10-1)
        
        
train_x, train_y = np.array(data_img[0]), np.array(data_label[0])
val_x, val_y = np.array(data_img[1]), np.array(data_label[1])
test_x, test_y = np.array(data_img[2]), np.array(data_label[2])

train_x=preprocess_input(train_x)
val_x=preprocess_input(val_x)
test_x=preprocess_input(test_x)

class_num = 20
train_y = to_categorical(train_y,class_num)
val_y = to_categorical(val_y,class_num)
test_y = to_categorical(test_y,class_num)

print(train_x.shape, train_y.shape)
print(val_x.shape, val_y.shape)
print(test_x.shape, test_y.shape)

print('*'*30)
print('Setup model')
print('*'*30)
# setup model
inputs = Input(shape=(row, col, 3))
base_model = VGG16(input_tensor=inputs, weights='imagenet', include_top=False)
# base_model  ResNet50(input_tensor=inputs, weights='imagenet', include_top=False)
# base_model == ResNet50(input_tensor=inputs, weights=None, include_top=False)
# base_model = DenseNet121(input_tensor=inputs, weights='imagenet', include_top=False)
# base_model = InceptionV3(input_tensor=inputs, weights='imagenet', include_top=False)
# base_model = ResNet152(input_tensor=inputs, weights='imagenet', include_top=False)
NB_IV3_LAYERS_TO_FREEZE=0
model = add_new_last_layer(base_model, class_num)
for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
    layer.trainable = False
for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
    layer.trainable = True
    
# model.summary()
# model.compile(loss="mse", optimizer=Adam(lr=1e-4), metrics=['mse'])
model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=1e-5), metrics=['acc', recall])



model_name = 'class_exp2_vgg16'
model_checkpoint = ModelCheckpoint('../checkpoint/{}.h5'.format(model_name), monitor='val_loss',verbose=1, save_best_only=True, mode='min')
earlystopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1)

# def sch(epoch):
#     if epoch>50:
#         return 1e-6
#     else:
#         return 1e-4
    
# sc=LearningRateScheduler(sch)

hist = model.fit(train_x, train_y, validation_data=(val_x, val_y), 
                 batch_size=32,epochs=100, verbose=1, callbacks=[model_checkpoint, earlystopping])

model.save('../{}.h5'.format(model_name))

# imgs_mask_test = model.predict(test_x, batch_size=1, verbose=1)

# ypred = model.predict(test_x)
# print(model.evaluate(test_x, test_y))
# print("MSE: %.4f" % mean_squared_error(test_x, ypred))

# print(model.evaluate(test_x, test_y, batch_size=1))

# print('*'*30)
# print('Best model evaluate')
# print('*'*30)
# model_best = keras.models.load_model('../exp1_mae.h5')
# ypred_best = model_best.predict(test_x)
# print(model_best.evaluate(test_x, test_y))
# print("MSE: %.4f" % mean_squared_error(test_y, ypred_best))
# print("MAE: %.4f" % mean_absolute_error(test_y, ypred_best))
