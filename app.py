import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re
import string
import nltk

# Load the tokenizer and model
tokenizer = joblib.load('tokenizer_file.pkl')
model = tf.keras.models.load_model('job_prediction.h5')

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Preprocessing functions
stop_words = set(stopwords.words("english"))
punctuation = list(string.punctuation)
stop_words.update(punctuation)
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def text_preprocessing(text):
    text = text.lower()
    text = re.sub(r"\\n", " ", text)
    text = text.strip()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"www.\S+", " ", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in tokens]
    text = ' '.join(tokens)
    return text

# Streamlit UI
st.title("Job Recruitment Fraud Detection")

st.write("Enter the job posting details below to check if it's fraudulent or not:")

title = st.text_input("Job Title")
location = st.text_input("Location")
company_profile = st.text_area("Company Profile")
requirements = st.text_area("Job Requirements")
industry = st.text_input("Industry")

if st.button("Predict"):
    combined_text = f"{title} {location} {company_profile} {description} {requirements} {benefits} {industry}"
    processed_text = text_preprocessing(combined_text)
    tokenized_text = tokenizer.texts_to_sequences([processed_text])
    padded_text = pad_sequences(tokenized_text, padding='pre', maxlen=tokenizer.maxlen)
    
    prediction = model.predict(padded_text)
    fraudulent = prediction[0][0] > 0.5

    if fraudulent:
        st.error("The job posting is likely fraudulent!")
    else:
        st.success("The job posting appears to be legitimate.")

if __name__ == '__main__':
    st.run()
