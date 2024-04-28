import sys
import numpy as np
import librosa
from skimage.transform import resize
from keras.models import load_model

def preprocess_audio(audio_path):
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        resized_spectrogram = resize(spectrogram, (200, 200))
        return resized_spectrogram.reshape(1, 200, 200, 1)
    except Exception as e:
        print(f"Error processing audio file {audio_path}: {e}")
        return None

def predict_audio(audio_path, model):
    preprocessed_audio = preprocess_audio(audio_path)
    if preprocessed_audio is not None:
        prediction = model.predict(preprocessed_audio)
        predicted_class = np.argmax(prediction, axis=1)
        label_map = {0: "Human Male", 1: "Human Female", 2: "Bot Male", 3: "Bot Female"}
        return label_map[predicted_class[0]]
    return "Error in prediction."

if __name__ == '__main__':
        model_path = 'sheeesh.keras'
        model = load_model(model_path)
        audio_path = sys.argv[1]
        result = predict_audio(audio_path, model)
        print("Predicted Class:", result)
        

