import sys
import numpy as np
import tensorflow as tf
from keras import layers, models
from sklearn.metrics import accuracy_score
import librosa
import pymysql
from sklearn.model_selection import train_test_split


def get_audio_data_from_db():
    connection = pymysql.connect(host='127.0.0.1',
                                 user='root',
                                 password='CSCI4400',
                                 database='CapstoneProject',
                                 cursorclass=pymysql.cursors.DictCursor)
    try:
        with connection.cursor() as cursor:
            sql = "SELECT aud, answer FROM audioo"
            cursor.execute(sql)
            result = cursor.fetchall()
            audio_paths = [row['aud'] for row in result]
            labels = [row['answer'] for row in result]
            return audio_paths, labels
    finally:
        connection.close()

# Load audio file and extract features
def load_and_extract_features(audio_paths):
    mfccs_list = []
    for audio_path in audio_paths:
        y, sr = librosa.load(audio_path)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_list.append(mfccs)
    return mfccs_list

# Truncate arrays to a fixed length and pad if necessary
def process_features(mfccs_list, max_length=500):
    return [arr[:, :max_length] if arr.shape[1] >= max_length else np.pad(arr, ((0, 0), (0, max_length - arr.shape[1])), mode='constant') for arr in mfccs_list]

# Fetch audio paths and labels from database
audio_paths, labels = get_audio_data_from_db()
mfccs_list = load_and_extract_features(audio_paths)
mfccs_processed = process_features(mfccs_list)

# Prepare data for the model
data = np.array(mfccs_processed)
data = data.reshape((-1, 13, data.shape[2], 1))  # Reshape for CNN input
labels = np.array(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Model Definition
num_mfcc = X_train.shape[1]
num_frames = X_train.shape[2]

model = models.Sequential([
    layers.Input(shape=(num_mfcc, num_frames, 1)),  # Input shape for MFCC features
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Model Training
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32)
old_stdout = sys.stdout
sys.stdout = sys.stderr
sys.stdout = old_stdout

# Inferencing
def classify_audio(audio_path, model=model, max_length=500):
    y, sr = librosa.load(audio_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_processed = mfccs[:, :max_length] if mfccs.shape[1] >= max_length else np.pad(mfccs, ((0, 0), (0, max_length - mfccs.shape[1])), mode='constant')
    features = mfccs_processed.reshape((1, 13, max_length, 1))
    predicted_label = model.predict(features)
    return "Bot" if predicted_label > 0.5 else "Human"

# Model Evaluation
y_pred = (model.predict(X_test) > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}%')


#main
if __name__ == '__main__':
    audio_path = sys.argv[1]
    result = classify_audio(audio_path)
    accuracy_str = "{:.2f}%".format(accuracy) 
    print(f'{{"result": "{result}"}}')


