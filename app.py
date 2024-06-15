import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import joblib
import os

# Function to preprocess text
def text_preprocessing(text):
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('stopwords', quiet=True)

    stop_words = set(stopwords.words("english"))
    punctuation = list(string.punctuation)
    stop_words.update(punctuation)
    lemmatizer = WordNetLemmatizer()

    def get_wordnet_pos(word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

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

# Load tokenizer
def load_tokenizer(tokenizer_path):
    try:
        tokenizer = joblib.load(tokenizer_path)
    except FileNotFoundError:
        st.error('Tokenizer file not found. Please make sure tokenizer_file.pkl exists.')
        st.stop()
    return tokenizer

# Function to load the model
@st.cache(allow_output_mutation=True)
def load_dl_model(model_path):
    try:
        model = load_model(model_path)
    except FileNotFoundError:
        st.error('Model file not found. Please make sure job_prediction.h5 exists.')
        st.stop()
    return model

# Streamlit App
def main():
    st.title('Job Posting Fraud Detection')
    st.write('Enter a job description to detect if it\'s potentially fake.')

    # Input job description
    job_description = st.text_area("Enter job description here:")

    if st.button('Predict'):
        if job_description:
            # Preprocess the input
            job_description_processed = text_preprocessing(job_description)
            tokenizer = load_tokenizer('tokenizer_file.pkl')
            max_features = 10000  # adjust based on your tokenizer configuration
            sent_length = 500  # adjust based on your tokenizer configuration

            job_description_encoded = tokenizer.texts_to_sequences([job_description_processed])
            job_description_padded = pad_sequences(job_description_encoded, padding='pre', maxlen=sent_length)

            # Load the deep learning model
            with st.spinner("Loading Model...."):
                model = load_dl_model('job_prediction.h5')

            # Make prediction
            prediction = model.predict(job_description_padded)[0][0]

            if prediction >= 0.5:
                st.error('Prediction Result: Fake Job')
            else:
                st.success('Prediction Result: Genuine Job')
        else:
            st.warning('Please enter a job description.')

    # Optional: Visualization or additional analysis
    st.sidebar.subheader('Explore Data Insights')
    st.sidebar.write('Choose an option to explore:')
    option = st.sidebar.selectbox('Select an option',
                                  ['Fraudulent vs Non-Fraudulent Distribution', 'Most Common Job Titles',
                                   'Most Job Posted by Country', 'Most Required Experience'])

    # Load data for visualization
    df = pd.read_csv('fake_job_postings.csv')
    df['country'] = df['location'].apply(lambda x: x.split(',')[0])
    df.fillna(' ', inplace=True)
    df = df.drop_duplicates()

    if option == 'Fraudulent vs Non-Fraudulent Distribution':
        st.subheader('Distribution of Fraudulent vs Non-Fraudulent Jobs')
        fig, ax = plt.subplots()
        sns.countplot(x='fraudulent', data=df, ax=ax)
        plt.xlabel('Fraudulent')
        plt.ylabel('Count')
        st.pyplot(fig)

    elif option == 'Most Common Job Titles':
        st.subheader('Most Common Job Titles')
        fig, ax = plt.subplots()
        sns.countplot(y='title', data=df, order=df['title'].value_counts().iloc[:5].index, ax=ax)
        plt.xlabel('Count')
        plt.ylabel('Job Title')
        st.pyplot(fig)

    elif option == 'Most Job Posted by Country':
        st.subheader('Most Job Posted by Country')
        fig, ax = plt.subplots()
        sns.countplot(y='country', data=df, order=df['country'].value_counts().iloc[:10].index, ax=ax)
        plt.xlabel('Count')
        plt.ylabel('Country')
        st.pyplot(fig)

    elif option == 'Most Required Experience':
        st.subheader('Most Required Experience')
        fig, ax = plt.subplots()
        sns.countplot(y='required_experience', data=df, order=df['required_experience'].value_counts().index, ax=ax)
        plt.xlabel('Count')
        plt.ylabel('Experience Level')
        st.pyplot(fig)

    st.sidebar.subheader('Word Cloud')
    cloud_option = st.sidebar.selectbox('Select a Word Cloud',
                                        ['Word Cloud for Fraudulent Job Postings', 'Word Cloud for Genuine Job Postings'])

    if cloud_option == 'Word Cloud for Fraudulent Job Postings':
        st.subheader('Word Cloud for Fraudulent Job Postings')
        fraudjobs_text_cleaned = df[df.fraudulent == 1]['description']
        wc = WordCloud(background_color="white", max_words=300, stopwords=STOPWORDS).generate(' '.join(fraudjobs_text_cleaned))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        st.pyplot()

    elif cloud_option == 'Word Cloud for Genuine Job Postings':
        st.subheader('Word Cloud for Genuine Job Postings')
        actualjobs_text_cleaned = df[df.fraudulent == 0]['description']
        wc = WordCloud(background_color="white", max_words=300, stopwords=STOPWORDS).generate(' '.join(actualjobs_text_cleaned))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        st.pyplot()

if __name__ == '__main__':
    main()
