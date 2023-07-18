from parameters import DATASET, MODEL_INFO
import numpy as np

def load_data():
    
    train_dict = dict()
    validation_dict = dict()
    test_dict = dict()

    # load train set
    train_dict['X'] = np.load(DATASET.train_folder + '/images.npy')
    train_dict['X'] = train_dict['X'].reshape([-1, MODEL_INFO.input_size, MODEL_INFO.input_size, 1])

    if MODEL_INFO.use_landmarks:
        train_dict['X2'] = np.load(DATASET.train_folder + '/landmarks.npy')
    if MODEL_INFO.use_hog_and_landmarks:
        train_dict['X2'] = np.load(DATASET.train_folder + '/landmarks.npy')
        train_dict['X2'] = np.array([x.flatten() for x in train_dict['X2']])
        train_dict['X2'] = np.concatenate((train_dict['X2'], np.load(DATASET.train_folder + '/hog_features.npy')), axis=1)

    train_dict['Y'] = np.load(DATASET.train_folder + '/labels.npy')

    # load validation set
    validation_dict['X'] = np.load(DATASET.validation_folder + '/images.npy')
    validation_dict['X'] = validation_dict['X'].reshape([-1, MODEL_INFO.input_size, MODEL_INFO.input_size, 1])

    if MODEL_INFO.use_landmarks:
        validation_dict['X2'] = np.load(DATASET.validation_folder + '/landmarks.npy')
    if MODEL_INFO.use_hog_and_landmarks:
        validation_dict['X2'] = np.load(DATASET.validation_folder + '/landmarks.npy')
        validation_dict['X2'] = np.array([x.flatten() for x in validation_dict['X2']])
        validation_dict['X2'] = np.concatenate((validation_dict['X2'], np.load(DATASET.validation_folder + '/hog_features.npy')), axis=1)

    validation_dict['Y'] = np.load(DATASET.validation_folder + '/labels.npy')


    # load test set
    test_dict['X'] = np.load(DATASET.test_folder + '/images.npy')
    test_dict['X'] = test_dict['X'].reshape([-1, MODEL_INFO.input_size, MODEL_INFO.input_size, 1])

    if MODEL_INFO.use_landmarks:
        test_dict['X2'] = np.load(DATASET.test_folder + '/landmarks.npy')
    if MODEL_INFO.use_hog_and_landmarks:
        test_dict['X2'] = np.load(DATASET.test_folder + '/landmarks.npy')
        test_dict['X2'] = np.array([x.flatten() for x in test_dict['X2']])
        test_dict['X2'] = np.concatenate((test_dict['X2'], np.load(DATASET.test_folder + '/hog_features.npy')), axis=1)

    test_dict['Y'] = np.load(DATASET.test_folder + '/labels.npy')

    return train_dict, validation_dict, test_dict
