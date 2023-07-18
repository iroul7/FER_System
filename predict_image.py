# %%
import numpy as np
import tensorflow as tf
import cv2
import os
import dlib
from keras.models import load_model
from skimage.feature import hog
from parameters import DATASET, TRAINING, MODEL_INFO, VIDEO_PREDICTOR

predictor = dlib.shape_predictor(DATASET.shape_predictor_path)
cascade_classifier = cv2.CascadeClassifier(DATASET.haarcascade_path)

def format_image(image):
    #이미지를 Gray로 바꿈
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = cascade_classifier.detectMultiScale(image, scaleFactor=1.3, minNeighbors = 5)

    # 이미지에 얼굴이 없으면 none 리턴
    if not len(faces) > 0:
        return None, None
    
    # faces리스트의 첫번째를 max_are_face로 설정
    max_are_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_are_face[2] * max_are_face[3]:
            max_are_face = face

    # face를 이미지로 바꿈
    face_coor =  max_are_face
    image = image[face_coor[1]:(face_coor[1] + face_coor[2]), face_coor[0]:(face_coor[0] + face_coor[3])]

    # 이미지 사이즈를 48,48로 줄임
    try:
        image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_CUBIC)

    #오류 메시지 출력
    except Exception:
        print("[+} Problem during resize")
        return None, None

    #이미지, 가장 큰 facd object 리턴
    return  image, face_coor

def image_to_tensor(image):
  tensor = np.asarray(image).reshape(-1, 2304) * 1 / 255.0
  return tensor

def get_landmarks(image, rects):
    if len(rects) > 1:
        raise BaseException("TooManyFaces")
    if len(rects) == 0:
        raise BaseException("NoFaces")
    return np.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])

#def predict(image, tensor_image, model):
def predict(image, model):
    cv2.imshow('face2', image)
    if MODEL_INFO.use_landmarks or MODEL_INFO.use_hog_and_landmarks:
        face_rects = [dlib.rectangle(left=0, top=0, right=MODEL_INFO.input_size, bottom=MODEL_INFO.input_size)]
        face_landmarks = np.array([get_landmarks(image, face_rects)])
        features = face_landmarks
        hog_features, _ = hog(image, orientations=8, pixels_per_cell=(16, 16),
                              cells_per_block=(1, 1), visualize=True)
        hog_features = np.asarray(hog_features)
        face_landmarks = face_landmarks.flatten()

        features = np.concatenate((face_landmarks, hog_features))
        tensor_image = image.reshape([-1, MODEL_INFO.input_size, MODEL_INFO.input_size, 1])
        predicted_label = model.predict([tensor_image, features.reshape((1, -1))])
        return predicted_label[0]

    else:
        tensor_image = image.reshape([-1, MODEL_INFO.input_size, MODEL_INFO.input_size, 1])
        predicted_label = model.predict(tensor_image)
        return predicted_label[0]

def main():
    model_path = TRAINING.checkpoint_dir + '/1.ModelB_Land_HOG.55-0.5938.hdf5'
    image_path = 'Data/Predict_images/happy.jpg'

    if os.path.isfile(model_path):
        model = load_model(model_path)
    else:
        print("Error: file '{}' not found".format(model_path))
        exit()
    
    if os.path.isfile(image_path):
        face_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        face_image_2 = cv2.imread(image_path, 0)
    else:
        print("Error: file '{}' not found".format(image_path))
        exit()
    
    detected_face, face_coor = format_image(face_image)
    
    temp_face = detected_face

    if detected_face is not None:
        detected_face = tf.image.convert_image_dtype(detected_face, dtype=float)
        
        three_d_image = tf.expand_dims(detected_face, axis=-1)
        four_d_image = tf.expand_dims(three_d_image, axis=0)

        #result = predict(face_image_2, four_d_image, model)
        result = predict(temp_face, model)

        for index, emotion in enumerate(VIDEO_PREDICTOR.emotions):
            print(emotion.rjust(10) + ' : ' + str(result[index] * 100) + ' %')
            cv2.putText(face_image, emotion, (10, index * 20 + 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
            cv2.rectangle(face_image, (130, index * 20 + 10), (130 + int(result[index] * 100), (index + 1) * 20 + 4),
                        (255, 0, 0), -1)  
        
        cv2.imshow('face', face_image)
        cv2.waitKey()
        cv2.destroyAllWindows()

    else:
        print("Error: Not detected face")
        exit()

if __name__ == '__main__':
  main()    