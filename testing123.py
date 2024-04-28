import sys
from tokenize import Ignore
import librosa
import numpy as np
import random
import pymysql.cursors
from keras.models import Sequential
from keras.layers import Input, Dense, MaxPooling2D, Conv2D, Flatten
from keras.optimizers import Adam
import warnings
from skimage.transform import resize



def get_audio_data_from_db():
    connection = pymysql.connect(host='127.0.0.1',
                                 user='root',
                                 password='CSCI4400',
                                 database='CapstoneProject',
                                 cursorclass=pymysql.cursors.DictCursor)
    try:
        with connection.cursor() as cursor:
            sql = "SELECT file_path, label_audio FROM M2audio"
            cursor.execute(sql)
            return cursor.fetchall()
    finally:
        connection.close()

def load_data():
    train_data = []
    data = get_audio_data_from_db()
    for item in data:
        try:
            audio_path = item['file_path']
            label = item['label_audio']
            audio_file, sr = librosa.load(audio_path, sr=None)
            spectrogram = librosa.feature.melspectrogram(y=audio_file, sr=sr)
            spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
            resized_spectrogram = resize(spectrogram, (200, 200))
            train_data.append([resized_spectrogram.reshape(200, 200, 1), label])
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
    return train_data

train = load_data()
random.shuffle(train)
audios = np.array([item[0] for item in train]).reshape(-1, 200, 200, 1)
labels = np.array([item[1] for item in train])

model = Sequential([
    Input(shape=(200, 200, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(4, activation='softmax')
])
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(audios, labels, batch_size=16, epochs=35, validation_split=0.20)

#print("\nTraining Summary:")
#print(f"Final Training Accuracy: {history.history['accuracy'][-1]}")
#print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]}")
#overall_accuracy = model.evaluate(audios, labels)[1]
#print(f"Overall Model Accuracy: {overall_accuracy}")

model.save('Wagwan.keras')