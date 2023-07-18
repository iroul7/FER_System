import numpy as np
from keras.models import load_model
from data_loader import load_data
from parameters import TRAINING, MODEL_INFO

Train, Validation, Test = load_data()

# Ensemble Test
# 1. Model load
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

# 2. Get Accuracy from each model
if MODEL_INFO.use_landmarks:
    y_accs = [model.evaluate([np.array(Test['X']), np.array(Test['X2'])], np.array(Test['Y']), batch_size=1024) for model in models]
else:
    y_accs = [model.evaluate(np.array(Test['X']), np.array(Test['Y']), batch_size=1024) for model in models]

# 3. Get Predicts from each model
if MODEL_INFO.use_landmarks:
    y_preds = [model.predict([Test['X'], Test['X2']]) for model in models]
else:
    y_preds = [model.predict(Test['X']) for model in models]

# 4. Basic Ensembles methods
y_max_classes = [np.argmax(y_pred, axis=1) for y_pred in y_preds]
y_preds_avg = np.mean(y_preds, axis=0)
y_preds_classes = np.argmax(y_preds_avg, axis=1)

# 5. Find Duplicated Emotion 
for index in range(0, len(y_max_classes[0])):
    y_avgs = [[False for col in range(0)] for row in range(7)]
    for y_max_class_index, y_max_class in enumerate(y_max_classes):
        y_avgs[y_max_class[index]].append(y_accs[y_max_class_index][1])

    max_value = 0
    for index2 in range(0, len(y_avgs)):
        if len(y_avgs[index2]) > max_value:
            max_value = len(y_avgs[index2])

    is_duplicated = False
    duplicated_counter = 0
    for index2 in range(0, len(y_avgs)):
        if max_value == len(y_avgs[index2]):
            duplicated_counter += 1

    if duplicated_counter > 1:
        avg = 0
        y_result = -1
        for index2 in range(0, len(y_avgs)):
            if y_avgs[index2]:
                y_Avg = np.mean(y_avgs[index2], axis=0)
                if y_Avg > avg:
                    avg = y_Avg
                    y_result = index2
        y_preds_classes[index] = y_result

# 6. Test Data 
y_test_classes = np.argmax(Test['Y'], axis=1)    

accuracy = np.mean(y_preds_classes == y_test_classes)
print("Ensemble Accuracy : " + str(accuracy))