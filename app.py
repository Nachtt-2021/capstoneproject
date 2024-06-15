import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow import keras

# Daftarkan initializer Orthogonal
custom_objects = {'Orthogonal': tf.keras.initializers.Orthogonal}

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('job_prediction.h5', custom_objects=custom_objects)
    return model

with st.spinner("Loading Model...."):
    model = load_model()
    
def fraudulent_predict(input_data):
    # Ubah input ke array NumPy dan tambahkan dimensi
    input_data_as_numpy_array = np.array(input_data, dtype=np.float32).reshape(1, -1)
    
    prediction = model.predict(input_data_as_numpy_array)
    
    if prediction[0][0] == 0:
        return 'Not Fraudulent'
    else:
        return 'Fraudulent'
    
def main():
    st.title("Job Posting Prediction")
        
    # Inisialisasi input data        
    Title = st.text_area('Posisi Pekerjaan yang Ditawarkan')
    Location = st.text_area('Lokasi Pekerjaan')
    Department = st.text_area('Divisi Spesifik Lowongan')
    Description = st.text_area('Detail Pekerjaan')
    Company_profile = st.text_area('Deskripsi Perusahaan')
    Requirements = st.text_area('Persyaratan Lowongan Pekerjaan')
    Benefits = st.text_area('Keuntungan yang ditawarkan kepada Pelamar')
    Required_experience = st.text_area('Tingkat pengalaman yang diperlukan untuk pekerjaan')
    Required_education = st.text_area('Tingkat pendidikan minimum yang diperlukan untuk pekerjaan')
    Industry = st.text_area('Perusahaan bergerak di sektor apa?')
    Function = st.text_area('Spesialisasi dalam industri tersebut')
        
    # Membuat combined_text
    combined_text = f"{Title} {Location} {Department} {Company_profile} {Description} {Requirements} {Benefits} {Required_experience} {Required_education} {Industry} {Function}"
        
    # Prediction
    fraudulent_diagnosis = ''
        
    if st.button('Prediction Result :'):
        fraudulent_diagnosis = fraudulent_predict([combined_text])
    
    st.success(fraudulent_diagnosis)
    
if __name__ == '__main__':
    main()
