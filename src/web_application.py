import streamlit as st
import cv2
import mahotas
import pickle
import numpy as np
import pandas as pd
import pandas
from PIL import Image
import lime
import lime.lime_tabular
from sklearn.model_selection import train_test_split
import streamlit.components.v1 as components

# carregar o modelo
def get_model():
    return pickle.load(open('C:/Users/cspau/Desktop/coisas do pc/Aprendendo Python/GitHub/leaf-diagnostic/etc/best_random_forest_model.dat', 'rb'))

# converter dados da imagem
def convert_byteio_image(string):
    array = np.frombuffer(string, np.uint8)
    image = cv2.imdecode(array, flags=1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# interpretar e exibir as previsões com interpretação
def interpret_and_display_predictions(model, image, features):
    features = mahotas.features.haralick(image, compute_14th_feature=True, return_mean=True).reshape(1, 14)

    df = pandas.read_csv('C:/Users/cspau/Desktop/coisas do pc/Aprendendo Python/GitHub/leaf-diagnostic/etc/features.csv', delimiter=';')

    X = df.drop('Label', axis=1)
    y = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X.columns, class_names=['Saudável', 'Moscas das Galhas'], 
                                                      feature_selection='lasso_path', discretize_continuous=True)
        
    exp = explainer.explain_instance(features.reshape(14,), model.predict_proba, num_features=14)
    
    st.image(image, caption="", use_column_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        

    with col2:
        st.markdown("<h3 style='text-align: center; color: black;'>Interpretação</h3>", unsafe_allow_html=True)
        components.html((exp.as_html(predict_proba=False)), height=500)

# título 
st.markdown("<h1 style='text-align: center; color: black;'>Aplicação de Diagnóstico de Folhas</h1>", unsafe_allow_html=True)

# barra lateral para upload de imagens
st.sidebar.title('Configurações')
uploaded_images = st.sidebar.file_uploader("Escolha até 10 imagens (JPG ou JPEG)", type=['jpg', 'jpeg'], accept_multiple_files=True, key="upload_images")

model = get_model()

if uploaded_images:
    images_and_features = []
    
    st.markdown("<h3 style='text-align: center; color: black;'>Imagens e Previsões</h3>", unsafe_allow_html=True)

    for uploaded_image in uploaded_images:
        bytes_data = uploaded_image.getvalue()
        image = convert_byteio_image(bytes_data)

        if image.shape != (256, 256):
            image = cv2.resize(image, (256, 256))

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # características da imagem
        features = mahotas.features.haralick(gray_image, compute_14th_feature=True, return_mean=True).reshape(14,)
        
        images_and_features.append((image, features))

    for idx, (img, features) in enumerate(images_and_features, start=1):
        prediction_probs = model.predict_proba([features])
        prediction = "Folha afetada pelas moscas-das-galhas com {:.2%} de certeza".format(prediction_probs[0][1]) if prediction_probs[0][1] > prediction_probs[0][0] else "Folha normal com {:.2%} de certeza".format(prediction_probs[0][0])

        st.markdown(f"<h4 style='text-align: center; color: black;'>Imagem {idx}: {prediction}</h4>", unsafe_allow_html=True)
        
        interpret_and_display_predictions(model, img, features)
