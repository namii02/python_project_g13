import streamlit as st
import pandas as pd
import joblib
import gzip
import shutil
import os


# Function to decompress the model file
def decompress_model(input_file, output_file):
    try:
        with gzip.open(input_file, 'rb') as f_in:
            with open(output_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    except Exception as e:
        st.error(f"Error decompressing the model file: {e}")
        raise


compressed_model_path = 'joblibs/rf_model.joblib.gz'
decompressed_model_path = 'joblibs/rf_model.joblib'

# Decompress the model file
if not os.path.exists(decompressed_model_path):
    decompress_model(compressed_model_path, decompressed_model_path)

# Load the trained model, scaler, feature columns, and data
try:
    model = joblib.load(decompressed_model_path)
except Exception as e:
    st.error(f"Error loading the decompressed model file: {e}")
    raise

scaler = joblib.load('joblibs/scaler.joblib')
feature_columns = joblib.load('joblibs/feature_columns.joblib')
initial_df = pd.read_csv('data/cleaned_data.csv')
encoded_df = pd.read_csv('data/encoded_data.csv')


st.markdown(
    """
    <style>
    .stApp {
        background: rgb(0,0,0);
        background: radial-gradient(circle, rgba(0,0,0,1) 70%, rgba(52,52,52,1) 100%);
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.header("Kuala Lumpur House Price Prediction")

col1, col2 = st.columns(2)

with col1:
    location_options = [col.replace('Location_', '') for col in feature_columns if 'Location_' in col]
    selected_location = st.selectbox("Select Location", location_options)

with col2:
    property_options = [col.replace('Property Type_', '') for col in feature_columns if 'Property Type_' in col]
    selected_property = st.selectbox("Property Type", property_options)

col3, col4, col5 = st.columns(3)

with col3:
    num_rooms = st.number_input("Number of Rooms", min_value=1, max_value=21, value=2, step=1)

with col4:
    num_bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=20, value=1, step=1)

with col5:
    num_car_parks = st.number_input("Number of Car Parks", min_value=0, max_value=30, value=1, step=1)

col6, col7 = st.columns(2)

with col6:
    size = st.number_input("Size", min_value=750, value=1500, max_value=12000000)
with col7:
    furnishing_options = [col.replace('Furnishing_', '') for col in feature_columns if 'Furnishing_' in col]
    selected_furnishing = st.selectbox("Furnishing", furnishing_options)

    
st.write("---")

data = {
    'Rooms': [num_rooms],
    'Bathrooms': [num_bathrooms],
    'Car Parks': [num_car_parks],
    'Size': [size],
    'Location_' + selected_location: [1],
    'Property Type_' + selected_property: [1],
    'Furnishing_' + selected_furnishing: [1],
}

input_df = pd.DataFrame(data).reindex(columns=feature_columns, fill_value=0)

input_scaled = scaler.transform(input_df)


# Make prediction
if st.button("Calculate"):
    prediction = model.predict(input_scaled)
    predicted_price = prediction[0]
    
    st.write(f"Predicted Price: RM {predicted_price:,.2f}")


