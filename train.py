# %%
import tensorflow as tf
import pandas as pd 
import time as tm
import datetime as dt
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator

from data_loader import load_data 
from model import getModel
from parameters import TRAINING, MODEL_INFO

# 1. loading data
print("loading dataset fer2013 ...")
Train, Validation, Test = load_data()

# 2. building model
print("building model ...")
model = getModel()
model.summary()
model.compile(loss=MODEL_INFO.loss, optimizer=Adam(), metrics=['accuracy'])

# 3. setting callback module
filepathLand = '_notLand'
filepathHOG = '_notHOG'
if MODEL_INFO.use_landmarks:
    filepathLand = '_Land'
if MODEL_INFO.use_hog_and_landmarks:
    filepathHOG = '_HOG'
filepath = TRAINING.checkpoint_dir + '/Model' + MODEL_INFO.model + filepathLand + filepathHOG + '.{epoch:02d}-{val_accuracy:.4f}.hdf5'
checkpointer = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
tensorboard = TensorBoard(log_dir=TRAINING.logs_dir)

# 4. training model
print("start training ...")
print( "  - emotions = {}".format(MODEL_INFO.output_size))
print( "  - model = {}".format(MODEL_INFO.model))
print( "  - batch size = {}".format(TRAINING.batch_size))
print( "  - epochs = {}".format(TRAINING.epochs))
print( "  - use landmarks = {}".format(MODEL_INFO.use_landmarks))
print( "  - use hog + landmarks = {}".format(MODEL_INFO.use_hog_and_landmarks))

# start time check
start = tm.time()

if MODEL_INFO.use_landmarks:
    history = model.fit([Train['X'], Train['X2']], Train['Y'],
                        batch_size=TRAINING.batch_size,
                        epochs=TRAINING.epochs,
                        validation_data=([Validation['X'], Validation['X2']], Validation['Y']),
                        callbacks=[checkpointer, tensorboard])
else:
    history = model.fit(Train['X'], Train['Y'],
                        batch_size=TRAINING.batch_size,
                        epochs=TRAINING.epochs,
                        validation_data=(Validation['X'], Validation['Y']),
                        callbacks=[checkpointer, tensorboard])
# end time check
end = tm.time()
sec = (end - start)
tResult = dt.timedelta(seconds=sec)
print("Training time : ")
print(tResult)

# 5. save history file
historyfilepath = TRAINING.history_dir + '/Model' + MODEL_INFO.model + filepathLand + filepathHOG + '_history.csv'
pd.DataFrame(history.history).to_csv(historyfilepath)



# %%
