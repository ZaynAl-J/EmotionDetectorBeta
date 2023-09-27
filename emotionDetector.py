import keras
# keras.__version__
print(keras.__version__)
import pandas as pd 
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps = PorterStemmer()
import re 
import numpy as np
import tensorflow as tf

from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Neural Network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import pickle 
from preprocess import *

# print("Emotion Detector")
# print("Creating Training, Validation, and Testing datasets:")

# # Create Training, Validation & Testing datasets
# train = pd.read_table('train.txt', delimiter=';', header=None, )
# val = pd.read_table('val.txt', delimiter=';', header=None, )
# test = pd.read_table('test.txt', delimiter=';', header=None, )

# # print("training \n", train)
# # print("validation \n", val)
# # print("testing \n", test)

# # Connect the datasets
# data = pd.concat([train, val, test])
# data.columns = ["text","label"]

# Save to file
# data.to_csv('data.csv')
# print(data)
# print(data.shape)
# Check if dataset has missing values:
#  data.isna().any(axis=1).sum()

# Text Preprocessing
# ps = PorterStemmer()

# def preprocess(line):
#     # Leave only characters from a to z
#     # print("removing characters")
#     review = re.sub('[^a-zA-Z]', ' ', line)
#     # Lowertext
#     review = review.lower()
#     # Turn string into list of words
#     review = review.split()
#     # Apply Stemming
#     # For example changing, changed, change --> chang or studying, studies, study --> studi

#     # Delete stop words like I, and, or
#     review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
#     # Turn list into sentences
#     return " ".join(review)

def main():
    print("Emotion Detector")
    print("Creating Training, Validation, and Testing datasets:")

    # Create Training, Validation & Testing datasets
    train = pd.read_table('train.txt', delimiter=';', header=None, )
    val = pd.read_table('val.txt', delimiter=';', header=None, )
    test = pd.read_table('test.txt', delimiter=';', header=None, )

    # print("training \n", train)
    # print("validation \n", val)
    # print("testing \n", test)

    # Connect the datasets
    data = pd.concat([train, val, test])
    data.columns = ["text","label"]

    # Save to file
    # data.to_csv('data.csv')

    # Apply text preprocessing:
    data['text'] = data['text'].apply(lambda x: preprocess(x))
    # data.to_csv('Preprocessed.csv')

    # Convert label values to numerical ones
    label_encoder = preprocessing.LabelEncoder()
    data['N_label'] = label_encoder.fit_transform(data['label'])

    # data.to_csv('labelUpdated.csv')

    # Create the Bag of Words model by applying CountVectorizer -convert textual data to numerical data
    # For Example: ****Look into this more*** the course was long-> [the,the course,the course was,course....
    cv = CountVectorizer(max_features=5000, ngram_range=(1,3))

    data_cv = cv.fit_transform(data['text']).toarray()

    # Split data set to train and test
    x_train, x_test, y_train, y_test = train_test_split(data_cv, data['N_label'], test_size=0.25, random_state=42)

    # Load the dataset, split into input (x) & output (y) variables, & define the keras model
    model = Sequential()
    model.add(Dense(12, input_shape=(x_train.shape[1], ), activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(6, activation='softmax'))

    # Compile the keras model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the keras model on the dataset
    model.fit(x_train, y_train, epochs=10, batch_size=10)

    # Evaluate the model
    # _, accuracy = model.evaluate(x_train, y_train)

    # print('Accuracy for training set: %.2f' % (accuracy*100))

    # _, accuracy = model.evaluate(x_test, y_test)
    # print('Accuracy for testing set: %.2f' % (accuracy*100))

    # Test on a input that is not in the dataset
    text = "I feel sad"
    text = preprocess(text)
    # Apply counter vectorizer
    array = cv.transform([text]).toarray()
    # Use model to predict the text
    pred = model.predict(array)
    # Return the max number aka the predicted value
    a = np.argmax(pred, axis=1)
    # Transform the label to the original value
    prediction = label_encoder.inverse_transform(a)[0]
    print(prediction)

    # Save the CNN model
    tf.keras.models.save_model(model, 'emotionDetector.keras')

    # Save the encoder
    pickle.dump(label_encoder, open('encoder.pk1', 'wb'))
    pickle.dump(cv, open('CountVectorizer.pk1', 'wb'))
    pickle.dump(preprocess, open('preprocess.pkl', 'wb'))
    

main()