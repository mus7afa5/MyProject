
#user input
import sys
import pandas as pd
import pymysql as mysql
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

def connection():
    con = mysql.connect(
        host="127.0.0.1",
        user="root",
        password="CSCI4400",
        database="CapstoneProject"
    )
    with con.cursor() as cursor:
        cursor.execute("SELECT result, sett FROM naive")
        data = cursor.fetchall()
    con.close()
    return data

data = connection()

df = pd.DataFrame(data, columns=['result', 'sett'])


# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['sett'], df['result'], test_size=0.2, random_state=42)

# Convert text to feature vectors
vectorizer = CountVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)


# Train the Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_vectors, y_train)

# Function to predict 
def predict_message(message):
    message_vector = vectorizer.transform([message])
    prediction = clf.predict(message_vector)
    return 'Bot' if prediction[0] == 1 else 'Human'

# model's accuracy on the test set
y_pred = clf.predict(X_test_vectors)
#print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

if __name__ == "__main__":
    input_text = sys.argv[1]
    prediction2 = predict_message(input_text)
    print(prediction2)


