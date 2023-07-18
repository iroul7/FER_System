import pandas as pd
import matplotlib.pyplot as plt
from parameters import TRAINING

history = pd.read_csv(TRAINING.history_dir + '/ModelB_Land_HOG_history.csv' , usecols = ['accuracy','loss','val_accuracy','val_loss'])

def plot_accuracy(data, size = (20,10)):
    plt.figure(figsize=size) 
    plt.plot(data['accuracy'])
    plt.plot(data['val_accuracy'])
    plt.title('Model Accuracy', fontsize = 18)
    plt.ylabel('Accuracy', fontsize = 18)
    plt.xlabel('Epoch', fontsize = 18)
    plt.legend(['Train', 'Test'], loc='upper left', fontsize = 16)
    plt.tick_params(axis='both', labelsize=14)
    plt.show()
    
plot_accuracy(history)

def plot_loss(data, size = (20,10)):
    plt.figure(figsize=size) 
    plt.plot(data['loss'])
    plt.plot(data['val_loss'])
    plt.title('Model Loss', fontsize = 18)
    plt.ylabel('Loss', fontsize = 18)
    plt.xlabel('Epoch', fontsize = 18)
    plt.ylim(0.9,2)
    plt.legend(['Train', 'Test'], loc='upper left', fontsize = 16)
    plt.tick_params(axis='both', labelsize=14)
    plt.show()
    
plot_loss(history)