import numpy as np
import pandas as pd
import os
import errno
import imageio
import dlib
import cv2

from skimage.feature import hog
from parameters import DATASET, MODEL_INFO

# initialization
image_height = MODEL_INFO.input_size
image_width = MODEL_INFO.input_size
ONE_HOT_ENCODING = True
GET_LANDMARKS = True
GET_HOG_FEATURES = True
GET_HOG_IMAGES = False
SELECTED_LABELS = []
IMAGES_PER_LABEL = 35887

# 1. choose the facial expression want to use : 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
input_expressions = "0,1,2,3,4,5,6"

expressions = input_expressions.split(",")
for i in range(0, len(expressions)):
    label = int(expressions[i])
    if (label >=0 and label<=6):
        SELECTED_LABELS.append(label)

if SELECTED_LABELS == []:
    SELECTED_LABELS = [0,1,2,3,4,5,6]

print(str(len(SELECTED_LABELS)) + " expressions")

# 2. loading Dlib predictor and preparing arrays
print("preparing")
predictor = dlib.shape_predictor(DATASET.shape_predictor_path)
original_labels = [0, 1, 2, 3, 4, 5, 6]
new_labels = list(set(original_labels) & set(SELECTED_LABELS))
nb_images_per_label = list(np.zeros(len(new_labels), 'uint8'))
try:
    os.makedirs(DATASET.features_folder)
except OSError as e:
    if e.errno == errno.EEXIST and os.path.isdir(DATASET.features_folder):
        pass
    else:
        raise

# get landmark features function
def get_landmarks(image, rects):
    if len(rects) > 1:
        raise BaseException("TooManyFaces")
    if len(rects) == 0:
        raise BaseException("NoFaces")
    return np.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])

# get new label function
def get_new_label(label, one_hot_encoding=False):
    if one_hot_encoding:
        new_label = new_labels.index(label)
        label = list(np.zeros(len(new_labels), 'uint8'))
        label[new_label] = 1
        return label
    else:
        return new_labels.index(label)

# 3. import fer2013 csv file
# 0 ~ 28708 (28709) : Training Data
# 28709 ~ 32297 (3589) : Public Test Data
# 32298 ~ 35886 (3589) : Private Test Data
print("importing csv file")
data = pd.read_csv(DATASET.fer_file_path)

# 4. classify by Usage : Training, PublicTet, PrivateTest / Save
for category in data['Usage'].unique():
    print( "converting set: " + category + "...")
    # create catgory folder
    if not os.path.exists(category):
        try:
            os.makedirs(DATASET.features_folder + '/' + category)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(DATASET.features_folder):
               pass
            else:
                raise
    
    # get samples and labels of the actual category
    category_data = data[data['Usage'] == category]
    samples = category_data['pixels'].values
    labels = category_data['emotion'].values
    
    # get images and extract features
    images = []
    labels_list = []
    landmarks = []
    hog_features = []
    hog_images = []
    for i in range(len(samples)):
        try:
            if labels[i] in SELECTED_LABELS and nb_images_per_label[get_new_label(labels[i])] < IMAGES_PER_LABEL:
                image = np.fromstring(samples[i], dtype=int, sep=" ").reshape((image_height, image_width))
                images.append(image)
                
                # Get HOG features & images
                if GET_HOG_FEATURES:
                    features, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                                            cells_per_block=(1, 1), visualize=True)
                    hog_features.append(features)
                    if GET_HOG_IMAGES:
                        hog_images.append(hog_image)

                # Get Landmark featrues
                if GET_LANDMARKS:
                    imageio.imwrite('temp.jpg', image)
                    image2 = cv2.imread('temp.jpg')
                    face_rects = [dlib.rectangle(left=1, top=1, right=47, bottom=47)]
                    face_landmarks = get_landmarks(image2, face_rects)
                    landmarks.append(face_landmarks)  
                              
                labels_list.append(get_new_label(labels[i], one_hot_encoding=ONE_HOT_ENCODING))
                nb_images_per_label[get_new_label(labels[i])] += 1

        except Exception as e:
            print( "error in image: " + str(i) + " - " + str(e))

    np.save(DATASET.features_folder + '/' + category + '/images.npy', images)
    if ONE_HOT_ENCODING:
        np.save(DATASET.features_folder + '/' + category + '/labels.npy', labels_list)
    else:
        np.save(DATASET.features_folder + '/' + category + '/labels.npy', labels_list)
    if GET_LANDMARKS:
        np.save(DATASET.features_folder + '/' + category + '/landmarks.npy', landmarks)
    if GET_HOG_FEATURES:
        np.save(DATASET.features_folder + '/' + category + '/hog_features.npy', hog_features)
        if GET_HOG_IMAGES:
            np.save(DATASET.features_folder + '/' + category + '/hog_images.npy', hog_images)
