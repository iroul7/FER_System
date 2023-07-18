import numpy as np
from keras.models import load_model
from data_loader import load_data
from parameters import TRAINING, MODEL_INFO

Train, Validation, Test = load_data()

# Ensemble Test
if MODEL_INFO.use_landmarks or MODEL_INFO.use_hog_and_landmarks:
    if MODEL_INFO.use_hog_and_landmarks:
        modelA = load_model(TRAINING.checkpoint_dir + '/1.ModelA_Land_HOG.74.hdf5')
        modelB = load_model(TRAINING.checkpoint_dir + '/1.ModelB_Land_HOG.55-0.5938.hdf5')
        modelC = load_model(TRAINING.checkpoint_dir + '/1.ModelC_Land_HOG.66-0.5821.hdf5')
        modelD = load_model(TRAINING.checkpoint_dir + '/1.ModelD_Land_HOG.93-0.6222.hdf5')
        modelE = load_model(TRAINING.checkpoint_dir + '/1.ModelE_Land_HOG.78-0.6063.hdf5')
        modelF = load_model(TRAINING.checkpoint_dir + '/1.ModelF_Land_HOG.62-0.5952.hdf5')
    else:
        modelA = load_model(TRAINING.checkpoint_dir + '/1.ModelA_Land_notHOG.78-0.6077.hdf5')
        modelB = load_model(TRAINING.checkpoint_dir + '/1.ModelB_Land_notHOG.68-0.5729.hdf5')
        modelC = load_model(TRAINING.checkpoint_dir + '/1.ModelC_Land_notHOG.54-0.5790.hdf5')
        modelD = load_model(TRAINING.checkpoint_dir + '/1.ModelD_Land_notHOG.88-0.6177.hdf5')
        modelE = load_model(TRAINING.checkpoint_dir + '/1.ModelE_Land_notHOG.88-0.6244.hdf5')
        modelF = load_model(TRAINING.checkpoint_dir + '/1.ModelF_Land_notHOG.87-0.6144.hdf5')
        
else:
    modelA = load_model(TRAINING.checkpoint_dir + '/1.ModelA_notLand_notHOG.45-0.5793.hdf5')
    modelB = load_model(TRAINING.checkpoint_dir + '/1.ModelB_notLand_notHOG.36-0.5687.hdf5')
    modelC = load_model(TRAINING.checkpoint_dir + '/1.ModelC_notLand_notHOG.32-0.5358.hdf5')
    modelD = load_model(TRAINING.checkpoint_dir + '/1.ModelD_notLand_notHOG.73-0.6158.hdf5')
    modelE = load_model(TRAINING.checkpoint_dir + '/1.ModelE_notLand_notHOG.44-0.5743.hdf5')
    modelF = load_model(TRAINING.checkpoint_dir + '/1.ModelF_notLand_notHOG.51-0.5862.hdf5')

models = [modelA, modelB, modelC, modelD, modelE, modelF]

if MODEL_INFO.use_landmarks:
    y_preds = [model.predict([Test['X'], Test['X2']]) for model in models]
    y_preds_avg = np.mean(y_preds, axis=0)
    y_preds_classes = np.argmax(y_preds_avg, axis=1)
    y_test_classes = np.argmax(Test['Y'], axis=1)

else:
    y_preds = [model.predict(Test['X']) for model in models]
    y_preds_avg = np.mean(y_preds, axis=0)
    y_preds_classes = np.argmax(y_preds_avg, axis=1)
    y_test_classes = np.argmax(Test['Y'], axis=1)


accuracy = np.mean(y_preds_classes == y_test_classes)
print("Ensemble Accuracy : " + str(accuracy))