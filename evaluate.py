import numpy as np
from keras.models import load_model
from data_loader import load_data
from parameters import TRAINING, MODEL_INFO

# F1-Score Test
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

Train, Validation, Test = load_data()
model = load_model(TRAINING.checkpoint_dir + '/1.ModelF_Land_HOG.62-0.5952.hdf5')
'''
if MODEL_INFO.use_landmarks:
    score = model.evaluate([np.array(Test['X']), np.array(Test['X2'])], np.array(Test['Y']), batch_size=1024)
    valscore = model.evaluate([np.array(Validation['X']), np.array(Validation['X2'])], np.array(Validation['Y']), batch_size=1024)
else:
    score = model.evaluate(np.array(Test['X']), np.array(Test['Y']), batch_size=1024)
    valscore = model.evaluate(np.array(Validation['X']), np.array(Validation['Y']), batch_size=1024)

print("Test Loss : " + str(score[0]))
print("Test Accu : " + str(score[1]))
print("validation Loss : " + str(valscore[0]))
print("validation Accu : " + str(valscore[1]))
'''



# F1-Score Test

if MODEL_INFO.use_landmarks:
    y_pred = model.predict([Test['X'], Test['X2']])
    y_pred_class = np.argmax(y_pred, axis=1)
    y_test_class = np.argmax(Test['Y'], axis=1)

else:
    y_pred = model.predict(Test['X'])
    y_pred_class = np.argmax(y_pred, axis=1)
    y_test_class = np.argmax(Test['Y'], axis=1)

    
acc = accuracy_score(y_test_class, y_pred_class)
print(acc)

f1score = f1_score(y_test_class, y_pred_class, average='micro')
print(f1score)
f1score = f1_score(y_test_class, y_pred_class, average='macro')
print(f1score)
f1score = f1_score(y_test_class, y_pred_class, average='weighted')
print(f1score)