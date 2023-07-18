import os

class Dataset:
    fer_file_path = 'Data/Resource/fer2013.csv'
    features_folder = 'Data/Resource/fer2013_features'
    train_folder = 'Data/Resource/fer2013_features/Training'
    validation_folder = 'Data/Resource/fer2013_features/PublicTest'
    test_folder = 'Data/Resource/fer2013_features/PrivateTest'
    shape_predictor_path = 'Data/Resource/shape_predictor_68_face_landmarks.dat'
    haarcascade_path = 'Data/Resource/haarcascade_files/haarcascade_frontalface_default.xml'

class ModelInfo:
    model = 'D'
    input_size = 48
    output_size = 7
    activation = 'relu'
    loss = 'categorical_crossentropy'
    use_landmarks = True
    use_hog_and_landmarks = True

class Training:
    batch_size = 128
    epochs = 200
    logs_dir = 'Data/Result/logs'
    checkpoint_dir = 'Data/Result/models'
    history_dir = 'Data/Result/historys'

class VideoPredictor:
    emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    print_emotions = False
    camera_source = 0
    face_detection_classifier = "lbpcascade_frontalface.xml"
    show_confidence = False
    time_to_wait_between_predictions = 0.5

DATASET = Dataset()
MODEL_INFO = ModelInfo()
TRAINING = Training()
VIDEO_PREDICTOR = VideoPredictor()

def make_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

make_dir(TRAINING.logs_dir)
make_dir(TRAINING.checkpoint_dir)
make_dir(TRAINING.history_dir)
