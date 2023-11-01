import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

custom_objects = {'Adam': Adam} # variable for formatting the model loading, optimizer type


def open_camera(camera_index=1):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    return cap

def preprocess_frame(frame, target_size=(48, 48)):
    """Convert frame to grayscale, resize, and scale pixel values."""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, target_size)
    preprocessed_frame = resized_frame / 255.0
    preprocessed_frame = np.expand_dims(np.expand_dims(preprocessed_frame, -1), 0)
    return preprocessed_frame

def load_emotion_model(model_path='../model/emotionIndicatorV3.keras'):
    """Load the pre-trained emotion classification model."""
    return load_model(model_path, compile=False)

def predict_emotion(model, frame):
    """Predict the emotion of the person in the frame."""
    preprocessed_frame = preprocess_frame(frame)
    emotion_prediction = model.predict(preprocessed_frame)
    return emotion_prediction

def display_emotion(frame, emotion_label):
    """Overlay the predicted emotion on the frame."""
    cv2.putText(frame, emotion_label, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

def capture_frames(cap, model, emotion_labels):
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot receive frame (stream end?). Exiting ...")
            break

        # Predict the emotion of the person in the frame
        emotion_prediction = predict_emotion(model, frame)
        emotion_label = emotion_labels[np.argmax(emotion_prediction)]

        # Display the predicted emotion on the frame
        display_emotion(frame, emotion_label)

        cv2.imshow('Camera Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def release_camera(cap):
    cap.release()
    cv2.destroyAllWindows()

def main():
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    model = load_emotion_model()
    cap = open_camera()
    try:
        capture_frames(cap, model, emotion_labels)
    finally:
        release_camera(cap)

if __name__ == '__main__':
    main()
