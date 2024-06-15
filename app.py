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
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import joblib

# Streamlit App
st.title('Job Posting Fraud Detection')
st.write('Analisis untuk mendeteksi penipuan dalam posting pekerjaan.')

# Upload the job description
job_description = st.text_area("Masukkan deskripsi pekerjaan di sini:")

if st.button('Prediksi'):
    if job_description:
        # Text Preprocessing Functions
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('stopwords')

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

        job_description_processed = text_preprocessing(job_description)
        tokenizer = joblib.load('tokenizer_file.pkl')
        max_features = 10000
        sent_length = 500  # Ensure you set this correctly based on your data preprocessing
        job_description_encoded = tokenizer.texts_to_sequences([job_description_processed])
        job_description_padded = pad_sequences(job_description_encoded, padding='pre', maxlen=sent_length)
        
        model = tf.keras.models.load_model('job_prediction.h5')
        prediction = model.predict(job_description_padded)[0][0]
        
        if prediction >= 0.5:
            st.error('Hasil Prediksi: Pekerjaan Palsu')
        else:
            st.success('Hasil Prediksi: Pekerjaan Asli')
    else:
        st.warning('Silakan masukkan deskripsi pekerjaan terlebih dahulu.')

# Visualization
# Load data
df = pd.read_csv('fake_job_postings.csv')
df['country'] = df['location'].apply(lambda x: x.split(',')[0])
df.fillna(' ', inplace=True)
df = df.drop_duplicates()

st.subheader('Distribusi Fraudulent vs Non-Fraudulent')
fig, ax = plt.subplots()
sns.barplot(x=df['fraudulent'].value_counts().index, y=df['fraudulent'].value_counts().values, ax=ax)
plt.xlabel('Fraudulent')
plt.ylabel('Count')
st.pyplot(fig)

st.subheader('Most Common Job Titles')
fig, ax = plt.subplots()
sns.barplot(x=df.title.value_counts()[:5].index, y=df.title.value_counts()[:5], ax=ax)
plt.xlabel('Title')
plt.ylabel('Count')
plt.xticks(rotation=90)
st.pyplot(fig)

country = dict(df.country.value_counts()[:11])
if ' ' in country:
    del country[' ']

st.subheader('Most Job Posted by Country')
fig, ax = plt.subplots()
plt.bar(country.keys(), country.values())
plt.xlabel('Countries')
plt.ylabel('No. of Jobs')
st.pyplot(fig)

experience = dict(df.required_experience.value_counts())
if ' ' in experience:
    del experience[' ']

st.subheader('Most Required Experience')
fig, ax = plt.subplots()
plt.bar(experience.keys(), experience.values())
plt.xlabel('Experience')
plt.ylabel('No. of Jobs')
st.pyplot(fig)

st.subheader('Word Cloud for Fraudulent Job Postings')
fraudjobs_text_cleaned = df[df.fraudulent == 1]['combined_text_processed']
wc = WordCloud(background_color="white", max_words=3000, width=1600, height=800, stopwords=STOPWORDS).generate(str(" ".join(fraudjobs_text_cleaned)))
fig, ax = plt.subplots()
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
st.pyplot(fig)

st.subheader('Word Cloud for Non-Fraudulent Job Postings')
actualjobs_text_cleaned = df[df.fraudulent == 0]['combined_text_processed']
wc = WordCloud(background_color="white", max_words=3000, width=1600, height=800, stopwords=STOPWORDS).generate(str(" ".join(actualjobs_text_cleaned)))
fig, ax = plt.subplots()
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
st.pyplot(fig)
