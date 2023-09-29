import streamlit.components.v1 as components
import lime.lime_tabular
import streamlit as st
import mahotas
import pickle
import pandas as pd
import numpy as np
import cv2

from sklearn.model_selection import train_test_split

# Função que carrega o modelo
def get_model():
    return pickle.load(open('C:/Users/cspau/Desktop/coisas do pc/Aprendendo Python/GitHub/leaf-diagnostic/etc/best_random_forest_model.dat', 'rb'))

# Função para converter dados de bytes para imagem
def convert_byteio_image(string):
    array = np.frombuffer(string, np.uint8)
    image = cv2.imdecode(array, flags=1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# Função para interpretar e exibir as previsões
def interpret_and_display_predictions(model, features, X_train):
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X_train.columns, class_names=['Saudável', 'moscas-das-galhas'],
                                                       feature_selection='lasso_path', discretize_continuous=True)
    exp = explainer.explain_instance(features.reshape(14,), model.predict_proba, num_features=14)
    return exp.as_html(predict_proba=False)

# Título web
st.markdown("<h1 style='text-align: center; color: black;'>Aplicação web para diagnosticar moscas-das-galhas nas folhas </h1>", unsafe_allow_html=True)

# Barra lateral para upload de imagens
st.sidebar.title('Configurações')
uploaded_image = st.sidebar.file_uploader("Escolha a imagem", type='jpg', accept_multiple_files=False)

# Carregar o modelo
model = get_model()

# Verificar se a imagem foi carregada
if uploaded_image is not None:
    bytes_data = uploaded_image.getvalue()
    image = convert_byteio_image(bytes_data)

    # redimensiona a imagem
    if (image.shape != (256, 256)):
        image = cv2.resize(image, (256, 256))

    # Extrair características da imagem
    features = mahotas.features.haralick(image, compute_14th_feature=True, return_mean=True).reshape(1, 14)

    # Fazer previsões
    prediction_probs = model.predict_proba(features)
    prediction = "Folha afetada pelas moscas-das-galhas com {:.2%} de certeza".format(prediction_probs[0][1]) if prediction_probs[0][1] > prediction_probs[0][0] else "Folha normal com {:.2%} de certeza".format(prediction_probs[0][0])

    # Exibir imagem e previsões
    st.markdown("<h3 style='text-align: center; color: black;'>Imagem</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([0.2, 5, 0.2])
    col2.image(image, use_column_width=True)
    st.markdown("<h4 style='text-align: center; color: black;'>" + prediction + "</h4>", unsafe_allow_html=True)

    # Botão para exibir interpretação
if st.sidebar.button("Exibir interpretação"):
    st.markdown("<h3 style='text-align: center; color: black;'>Interpretação</h3>", unsafe_allow_html=True)
    with st.spinner('Calculando...'):
        # Carregar dados e treinar o modelo
        df = pd.read_csv('C:/Users/cspau/Desktop/coisas do pc/Aprendendo Python/GitHub/leaf-diagnostic/etc/features.csv', delimiter=';')
        X_train, X_test, _, _ = train_test_split(df.drop('Label', axis=1), df['Label'], test_size=0.2, random_state=42)
        # Interpretar e exibir as previsões
        html_exp = interpret_and_display_predictions(model, features, X_train)
        components.html(html_exp, height=800)
