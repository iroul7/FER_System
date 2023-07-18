import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential, load_model, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Dropout, Flatten, BatchNormalization, SeparableConv2D, Input, Concatenate
from keras.optimizers import Adam
from keras.initializers import  RandomNormal
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import keras

from parameters import MODEL_INFO

def getModel():
    if MODEL_INFO.model == 'A':
        return getModelA()
    elif MODEL_INFO.model == 'B':
        return getModelB()
    elif MODEL_INFO.model == 'C':
        return getModelC()
    elif MODEL_INFO.model == 'D':
        return getModelA()
    elif MODEL_INFO.model == 'E':
        return getModelA()
    elif MODEL_INFO.model == 'F':
        return getModelA()
    else:
        print( "ERROR : not valid model " + str(MODEL_INFO.model))
        exit()

def getModelA():
    # Input Image Layers
    input1 = Input(shape=(MODEL_INFO.input_size, MODEL_INFO.input_size, 1))
    conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', bias_initializer=RandomNormal(stddev=1), kernel_initializer=RandomNormal(stddev=1))(input1)
    conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same', bias_initializer=RandomNormal(stddev=1), kernel_initializer=RandomNormal(stddev=1))(conv1_1)
    batch1_1 = BatchNormalization()(conv1_2)
    pool1_1 = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(batch1_1)
    drop1_1 = Dropout(0.25)(pool1_1)

    conv1_3 = Conv2D(64, (3, 3), activation='relu', padding='same', bias_initializer=RandomNormal(stddev=1), kernel_initializer=RandomNormal(stddev=1))(drop1_1)
    conv1_4 = Conv2D(64, (3, 3), activation='relu', padding='same', bias_initializer=RandomNormal(stddev=1), kernel_initializer=RandomNormal(stddev=1))(conv1_3)
    batch1_2 = BatchNormalization()(conv1_4)
    pool1_2 = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(batch1_2)
    drop1_2 = Dropout(0.25)(pool1_2)

    conv1_5 = Conv2D(128, (3, 3), activation='relu', padding='same', bias_initializer=RandomNormal(stddev=1), kernel_initializer=RandomNormal(stddev=1))(drop1_2)
    conv1_6 = Conv2D(128, (3, 3), activation='relu', padding='same', bias_initializer=RandomNormal(stddev=1), kernel_initializer=RandomNormal(stddev=1))(conv1_5)
    batch1_3 = BatchNormalization()(conv1_6)
    pool1_3 = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(batch1_3)
    drop1_3 = Dropout(0.25)(pool1_3)

    conv1_7 = Conv2D(256, (3, 3), activation='relu', padding='same', bias_initializer=RandomNormal(stddev=1), kernel_initializer=RandomNormal(stddev=1))(drop1_3)
    conv1_8 = Conv2D(256, (3, 3), activation='relu', padding='same', bias_initializer=RandomNormal(stddev=1), kernel_initializer=RandomNormal(stddev=1))(conv1_7)
    batch1_4 = BatchNormalization()(conv1_8)
    pool1_4 = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(batch1_4)
    drop1_4 = Dropout(0.5)(pool1_4)

    flatten1 = Flatten()(drop1_4)
    Dense1_1 = Dense(2048, activation='relu')(flatten1)
    drop1_5 = Dropout(0.5)(Dense1_1)

    # if use Features(Landmark,HOG)
    Dense1_2 = Dense(128, activation='relu')(drop1_5)

    # Input Features(Landmark, HOG) Layers
    if MODEL_INFO.use_landmarks or MODEL_INFO.use_hog_and_landmarks:
        if MODEL_INFO.use_hog_and_landmarks:
            input2 = Input(shape=(208,))
            Dense2_1 = Dense(1024, activation='relu')(input2)
        else:
            input2 = Input(shape=(68, 2))
            flatten2 = Flatten()(input2)
            Dense2_1 = Dense(1024, activation='relu')(flatten2)

        Dense2_2 = Dense(128, activation='relu')(Dense2_1)
        concat = Concatenate()([Dense1_2, Dense2_2])

        output = Dense(MODEL_INFO.output_size, activation='softmax')(concat)
        model = Model(inputs=[input1, input2], outputs=output)
    else:
        output = Dense(MODEL_INFO.output_size, activation='softmax')(drop1_5)
        model = Model(inputs=input1, outputs=output)

    return model

def getModelB():
    # Input Image Layers
    input1 = Input(shape=(MODEL_INFO.input_size, MODEL_INFO.input_size, 1))
    conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', bias_initializer=RandomNormal(stddev=1), kernel_initializer=RandomNormal(stddev=1))(input1)
    conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same', bias_initializer=RandomNormal(stddev=1), kernel_initializer=RandomNormal(stddev=1))(conv1_1)
    batch1_1 = BatchNormalization()(conv1_2)
    pool1_1 = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(batch1_1)
    drop1_1 = Dropout(0.25)(pool1_1)

    conv1_3 = Conv2D(128, (3, 3), activation='relu', padding='same', bias_initializer=RandomNormal(stddev=1), kernel_initializer=RandomNormal(stddev=1))(drop1_1)
    conv1_4 = Conv2D(128, (3, 3), activation='relu', padding='same', bias_initializer=RandomNormal(stddev=1), kernel_initializer=RandomNormal(stddev=1))(conv1_3)
    batch1_2 = BatchNormalization()(conv1_4)
    pool1_2 = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(batch1_2)
    drop1_2 = Dropout(0.25)(pool1_2)

    conv1_5 = Conv2D(256, (3, 3), activation='relu', padding='same', bias_initializer=RandomNormal(stddev=1), kernel_initializer=RandomNormal(stddev=1))(drop1_2)
    conv1_6 = Conv2D(256, (3, 3), activation='relu', padding='same', bias_initializer=RandomNormal(stddev=1), kernel_initializer=RandomNormal(stddev=1))(conv1_5)
    batch1_3 = BatchNormalization()(conv1_6)
    pool1_3 = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(batch1_3)
    drop1_3 = Dropout(0.25)(pool1_3)

    conv1_7 = Conv2D(512, (3, 3), activation='relu', padding='same', bias_initializer=RandomNormal(stddev=1), kernel_initializer=RandomNormal(stddev=1))(drop1_3)
    conv1_8 = Conv2D(512, (3, 3), activation='relu', padding='same', bias_initializer=RandomNormal(stddev=1), kernel_initializer=RandomNormal(stddev=1))(conv1_7)
    batch1_4 = BatchNormalization()(conv1_8)
    pool1_4 = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(batch1_4)
    drop1_4 = Dropout(0.5)(pool1_4)

    flatten1 = Flatten()(drop1_4)
    Dense1_1 = Dense(2048, activation='relu')(flatten1)
    drop1_5 = Dropout(0.5)(Dense1_1)

    # if use Features(Landmark,HOG)
    Dense1_2 = Dense(128, activation='relu')(drop1_5)

    # Input Features(Landmark, HOG) Layers
    if MODEL_INFO.use_landmarks or MODEL_INFO.use_hog_and_landmarks:
        if MODEL_INFO.use_hog_and_landmarks:
            input2 = Input(shape=(208,))
            Dense2_1 = Dense(1024, activation='relu')(input2)
        else:
            input2 = Input(shape=(68, 2))
            flatten2 = Flatten()(input2)
            Dense2_1 = Dense(1024, activation='relu')(flatten2)

        Dense2_2 = Dense(128, activation='relu')(Dense2_1)
        concat = Concatenate()([Dense1_2, Dense2_2])

        output = Dense(MODEL_INFO.output_size, activation='softmax')(concat)
        model = Model(inputs=[input1, input2], outputs=output)
    else:
        output = Dense(MODEL_INFO.output_size, activation='softmax')(drop1_5)
        model = Model(inputs=input1, outputs=output)

    return model

def getModelC():
    # Input Image Layers
    input1 = Input(shape=(MODEL_INFO.input_size, MODEL_INFO.input_size, 1))
    conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', bias_initializer=RandomNormal(stddev=1), kernel_initializer=RandomNormal(stddev=1))(input1)
    batch1_1 = BatchNormalization()(conv1_1)
    pool1_1 = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(batch1_1)
    drop1_1 = Dropout(0.25)(pool1_1)

    conv1_2 = Conv2D(128, (3, 3), activation='relu', padding='same', bias_initializer=RandomNormal(stddev=1), kernel_initializer=RandomNormal(stddev=1))(drop1_1)
    batch1_2 = BatchNormalization()(conv1_2)
    pool1_2 = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(batch1_2)
    drop1_2 = Dropout(0.25)(pool1_2)

    conv1_3 = Conv2D(256, (3, 3), activation='relu', padding='same', bias_initializer=RandomNormal(stddev=1), kernel_initializer=RandomNormal(stddev=1))(drop1_2)
    batch1_3 = BatchNormalization()(conv1_3)
    pool1_3 = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(batch1_3)
    drop1_3 = Dropout(0.25)(pool1_3)

    conv1_4 = Conv2D(512, (3, 3), activation='relu', padding='same', bias_initializer=RandomNormal(stddev=1), kernel_initializer=RandomNormal(stddev=1))(drop1_3)
    batch1_4 = BatchNormalization()(conv1_4)
    pool1_4 = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(batch1_4)
    drop1_4 = Dropout(0.25)(pool1_4)

    flatten1 = Flatten()(drop1_4)
    Dense1_1 = Dense(2048, activation='relu')(flatten1)
    drop1_5 = Dropout(0.5)(Dense1_1)

    # if use Features(Landmark,HOG)
    Dense1_2 = Dense(128, activation='relu')(drop1_5)

    # Input Features(Landmark, HOG) Layers
    if MODEL_INFO.use_landmarks or MODEL_INFO.use_hog_and_landmarks:
        if MODEL_INFO.use_hog_and_landmarks:
            input2 = Input(shape=(208,))
            Dense2_1 = Dense(1024, activation='relu')(input2)
        else:
            input2 = Input(shape=(68, 2))
            flatten2 = Flatten()(input2)
            Dense2_1 = Dense(1024, activation='relu')(flatten2)

        Dense2_2 = Dense(128, activation='relu')(Dense2_1)
        concat = Concatenate()([Dense1_2, Dense2_2])

        output = Dense(MODEL_INFO.output_size, activation='softmax')(concat)
        model = Model(inputs=[input1, input2], outputs=output)
    else:
        output = Dense(MODEL_INFO.output_size, activation='softmax')(drop1_5)
        model = Model(inputs=input1, outputs=output)

    return model

def getModelD():
    # Input Image Layers
    input1 = Input(shape=(MODEL_INFO.input_size, MODEL_INFO.input_size, 1))
    conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', bias_initializer=RandomNormal(stddev=1), kernel_initializer=RandomNormal(stddev=1))(input1)
    batch1_1 = BatchNormalization()(conv1_1)
    pool1_1 = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(batch1_1)
    drop1_1 = Dropout(0.25)(pool1_1)

    conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same', bias_initializer=RandomNormal(stddev=1), kernel_initializer=RandomNormal(stddev=1))(drop1_1)
    batch1_2 = BatchNormalization()(conv1_2)
    pool1_2 = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(batch1_2)
    drop1_2 = Dropout(0.25)(pool1_2)

    conv1_3 = Conv2D(128, (3, 3), activation='relu', padding='same', bias_initializer=RandomNormal(stddev=1), kernel_initializer=RandomNormal(stddev=1))(drop1_2)
    batch1_3 = BatchNormalization()(conv1_3)
    pool1_3 = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(batch1_3)
    drop1_3 = Dropout(0.25)(pool1_3)

    conv1_4 = Conv2D(256, (3, 3), activation='relu', padding='same', bias_initializer=RandomNormal(stddev=1), kernel_initializer=RandomNormal(stddev=1))(drop1_3)
    batch1_4 = BatchNormalization()(conv1_4)
    pool1_4 = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(batch1_4)
    drop1_4 = Dropout(0.25)(pool1_4)

    flatten1 = Flatten()(drop1_4)
    Dense1_1 = Dense(2048, activation='relu')(flatten1)
    drop1_5 = Dropout(0.5)(Dense1_1)

    # if use Features(Landmark,HOG)
    Dense1_2 = Dense(128, activation='relu')(drop1_5)

    # Input Features(Landmark, HOG) Layers
    if MODEL_INFO.use_landmarks or MODEL_INFO.use_hog_and_landmarks:
        if MODEL_INFO.use_hog_and_landmarks:
            input2 = Input(shape=(208,))
            Dense2_1 = Dense(1024, activation='relu')(input2)
        else:
            input2 = Input(shape=(68, 2))
            flatten2 = Flatten()(input2)
            Dense2_1 = Dense(1024, activation='relu')(flatten2)

        Dense2_2 = Dense(128, activation='relu')(Dense2_1)
        concat = Concatenate()([Dense1_2, Dense2_2])

        output = Dense(MODEL_INFO.output_size, activation='softmax')(concat)
        model = Model(inputs=[input1, input2], outputs=output)
    else:
        output = Dense(MODEL_INFO.output_size, activation='softmax')(drop1_5)
        model = Model(inputs=input1, outputs=output)

    return model

def getModelE():
    # Input Image Layers
    input1 = Input(shape=(MODEL_INFO.input_size, MODEL_INFO.input_size, 1))
    conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', bias_initializer=RandomNormal(stddev=1), kernel_initializer=RandomNormal(stddev=1))(input1)
    conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same', bias_initializer=RandomNormal(stddev=1), kernel_initializer=RandomNormal(stddev=1))(conv1_1)
    batch1_1 = BatchNormalization()(conv1_2)
    pool1_1 = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(batch1_1)
    drop1_1 = Dropout(0.25)(pool1_1)

    conv1_3 = Conv2D(128, (3, 3), activation='relu', padding='same', bias_initializer=RandomNormal(stddev=1), kernel_initializer=RandomNormal(stddev=1))(drop1_1)
    conv1_4 = Conv2D(128, (3, 3), activation='relu', padding='same', bias_initializer=RandomNormal(stddev=1), kernel_initializer=RandomNormal(stddev=1))(conv1_3)
    batch1_2 = BatchNormalization()(conv1_4)
    pool1_2 = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(batch1_2)
    drop1_2 = Dropout(0.25)(pool1_2)

    conv1_5 = Conv2D(256, (3, 3), activation='relu', padding='same', bias_initializer=RandomNormal(stddev=1), kernel_initializer=RandomNormal(stddev=1))(drop1_2)
    conv1_6 = Conv2D(256, (3, 3), activation='relu', padding='same', bias_initializer=RandomNormal(stddev=1), kernel_initializer=RandomNormal(stddev=1))(conv1_5)
    batch1_3 = BatchNormalization()(conv1_6)
    pool1_3 = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(batch1_3)
    drop1_3 = Dropout(0.25)(pool1_3)

    conv1_7 = Conv2D(512, (3, 3), activation='relu', padding='same', bias_initializer=RandomNormal(stddev=1), kernel_initializer=RandomNormal(stddev=1))(drop1_3)
    conv1_8 = Conv2D(512, (3, 3), activation='relu', padding='same', bias_initializer=RandomNormal(stddev=1), kernel_initializer=RandomNormal(stddev=1))(conv1_7)
    batch1_4 = BatchNormalization()(conv1_8)
    pool1_4 = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(batch1_4)
    drop1_4 = Dropout(0.25)(pool1_4)

    conv1_9 = Conv2D(1024, (3, 3), activation='relu', padding='same', bias_initializer=RandomNormal(stddev=1), kernel_initializer=RandomNormal(stddev=1))(drop1_4)
    conv1_10 = Conv2D(1024, (3, 3), activation='relu', padding='same', bias_initializer=RandomNormal(stddev=1), kernel_initializer=RandomNormal(stddev=1))(conv1_9)
    batch1_5 = BatchNormalization()(conv1_10)
    pool1_5 = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(batch1_5)
    drop1_5 = Dropout(0.5)(pool1_5)

    flatten1 = Flatten()(drop1_5)
    Dense1_1 = Dense(2048, activation='relu')(flatten1)
    drop1_5 = Dropout(0.5)(Dense1_1)

    # if use Features(Landmark,HOG)
    Dense1_2 = Dense(128, activation='relu')(drop1_5)

    # Input Features(Landmark, HOG) Layers
    if MODEL_INFO.use_landmarks or MODEL_INFO.use_hog_and_landmarks:
        if MODEL_INFO.use_hog_and_landmarks:
            input2 = Input(shape=(208,))
            Dense2_1 = Dense(1024, activation='relu')(input2)
        else:
            input2 = Input(shape=(68, 2))
            flatten2 = Flatten()(input2)
            Dense2_1 = Dense(1024, activation='relu')(flatten2)

        Dense2_2 = Dense(128, activation='relu')(Dense2_1)
        concat = Concatenate()([Dense1_2, Dense2_2])

        output = Dense(MODEL_INFO.output_size, activation='softmax')(concat)
        model = Model(inputs=[input1, input2], outputs=output)
    else:
        output = Dense(MODEL_INFO.output_size, activation='softmax')(drop1_5)
        model = Model(inputs=input1, outputs=output)

    return model

def getModelF():
    # Input Image Layers
    input1 = Input(shape=(MODEL_INFO.input_size, MODEL_INFO.input_size, 1))
    conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', bias_initializer=RandomNormal(stddev=1), kernel_initializer=RandomNormal(stddev=1))(input1)
    batch1_1 = BatchNormalization()(conv1_1)
    pool1_1 = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(batch1_1)
    drop1_1 = Dropout(0.25)(pool1_1)

    conv1_2 = Conv2D(128, (3, 3), activation='relu', padding='same', bias_initializer=RandomNormal(stddev=1), kernel_initializer=RandomNormal(stddev=1))(drop1_1)
    batch1_2 = BatchNormalization()(conv1_2)
    pool1_2 = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(batch1_2)
    drop1_2 = Dropout(0.25)(pool1_2)

    conv1_3 = Conv2D(256, (3, 3), activation='relu', padding='same', bias_initializer=RandomNormal(stddev=1), kernel_initializer=RandomNormal(stddev=1))(drop1_2)
    batch1_3 = BatchNormalization()(conv1_3)
    pool1_3 = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(batch1_3)
    drop1_3 = Dropout(0.25)(pool1_3)

    conv1_4 = Conv2D(512, (3, 3), activation='relu', padding='same', bias_initializer=RandomNormal(stddev=1), kernel_initializer=RandomNormal(stddev=1))(drop1_3)
    batch1_4 = BatchNormalization()(conv1_4)
    pool1_4 = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(batch1_4)
    drop1_4 = Dropout(0.25)(pool1_4)

    conv1_5 = Conv2D(1024, (3, 3), activation='relu', padding='same', bias_initializer=RandomNormal(stddev=1), kernel_initializer=RandomNormal(stddev=1))(drop1_4)
    batch1_5 = BatchNormalization()(conv1_5)
    pool1_5 = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(batch1_5)
    drop1_5 = Dropout(0.5)(pool1_5)

    flatten1 = Flatten()(drop1_5)
    Dense1_1 = Dense(2048, activation='relu')(flatten1)
    drop1_5 = Dropout(0.5)(Dense1_1)

    # if use Features(Landmark,HOG)
    Dense1_2 = Dense(128, activation='relu')(drop1_5)

    # Input Features(Landmark, HOG) Layers
    if MODEL_INFO.use_landmarks or MODEL_INFO.use_hog_and_landmarks:
        if MODEL_INFO.use_hog_and_landmarks:
            input2 = Input(shape=(208,))
            Dense2_1 = Dense(1024, activation='relu')(input2)
        else:
            input2 = Input(shape=(68, 2))
            flatten2 = Flatten()(input2)
            Dense2_1 = Dense(1024, activation='relu')(flatten2)

        Dense2_2 = Dense(128, activation='relu')(Dense2_1)
        concat = Concatenate()([Dense1_2, Dense2_2])

        output = Dense(MODEL_INFO.output_size, activation='softmax')(concat)
        model = Model(inputs=[input1, input2], outputs=output)
    else:
        output = Dense(MODEL_INFO.output_size, activation='softmax')(drop1_5)
        model = Model(inputs=input1, outputs=output)

    return model
