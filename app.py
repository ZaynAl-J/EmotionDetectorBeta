import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf 
from preprocess import *

st.write("""
    Prediction
""")

st.sidebar.header('User Input')

def user_input():
    text = st.sidebar.text_input("Enter a sentence: ")
    return text 

def main():
    input = user_input()

    st.write(input)
    encoder = pickle.load(open('encoder.pk1', 'rb'))
    cv = pickle.load(open('CountVectorizer.pk1', 'rb'))
    model = tf.keras.models.load_model('emotionDetector.keras')

    input = preprocess(input)
    array = cv.transform([input]).toarray()

    pred = model.predict(array)
    a = np.argmax(pred, axis=1)
    prediction = encoder.inverse_transform(a)[0]

    st.subheader('Prediction')
    if input == '':
        st.write('The emotion of this text is...')
    else:
        st.write(prediction)

main()
