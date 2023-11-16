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
import SimpleITK as sitk
from radiomics import featureextractor

def extrair_caracteristicas_radiomics(imagem_cv2):
    # Crie o extrator de características Radiomics
    extractor = featureextractor.RadiomicsFeatureExtractor(shape2D=True)

    image = sitk.GetImageFromArray(imagem_cv2)

    # Crie uma máscara que cubra toda a imagem
    mask = sitk.Image(image.GetSize(), sitk.sitkUInt8)
    mask.CopyInformation(image)
    mask = sitk.BinaryThreshold(mask, lowerThreshold=0, upperThreshold=1, insideValue=1, outsideValue=0)

    # Calcule as características radiômicas
    result = extractor.execute(image, mask)

    # Converta os valores para float, atribuindo 0.0 se não for possível converter
    features_values = [float(value) if isinstance(value, (int, float)) else 0.0 for value in list(result.values())[:98]]

    return np.array(features_values)

# carregar o modelo
def get_model():
    return pickle.load(open('C:/Users/cspau/Desktop/coisas do pc/Aprendendo Python/GitHub/leaf-diagnostic/etc/best_xgboost_model.dat', 'rb'))

# converter dados da imagem
def convert_byteio_image(string):
    array = np.frombuffer(string, np.uint8)
    image = cv2.imdecode(array, flags=1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

#def interpret_and_display_predictions(model, image, features):
    features = features.reshape(1, -1)  # Reshape para uma matriz 2D

    df = pd.read_csv('C:/Users/cspau/Desktop/coisas do pc/Aprendendo Python/GitHub/leaf-diagnostic/etc/features.csv', delimiter=';')

    X = df.drop('Label', axis=1)
    y = df['Label']

    # Certifique-se de que o formato dos dados seja compatível com o modelo
    model_input = features.ravel().reshape(1, -1)

    # Ajuste o número de características em model_input[0]
    if model_input.shape[1] != 98:
        # Ajuste o número de características para 98
        model_input = model_input[:, :98]

    # Converta todas as colunas do DataFrame para números
    X = X.apply(pd.to_numeric, errors='coerce')

    # Substitua valores NaN por 0 (ou qualquer valor padrão desejado)
    X = X.fillna(0)

    class_names = ['Saudável', 'Moscas das Galhas', 'Morta', 'Gorgulho', 'Antracnose', 'Cancro Bacteriano', 'Fumagina']

    explainer = lime.lime_tabular.LimeTabularExplainer(X.values,
                                                  mode='classification',
                                                  training_labels=y,
                                                  feature_names=list(X.columns),
                                                  class_names=class_names,)
    
    # O número de features no modelo e no LimeTabular deve coincidir
    num_features = min(X.shape[1], 98)  
    exp = explainer.explain_instance(model_input[0], model.predict_proba, num_features=num_features, top_labels=len(class_names), num_samples=5000)

    col1, col2 = st.columns(2)
    
    with col1: 
        st.image(image, caption="", use_column_width=True)
        
    with col2:
        st.markdown("<h3 style='text-align: center; color: white;'>Interpretação</h3>", unsafe_allow_html=True)
        interpretation_text = exp.as_html(predict_proba=False)
        interpretation_text = interpretation_text.replace("color:#000000;", "color:gray;")
        components.html((exp.as_html(predict_proba=False)), width=800, height=500)

# título 
st.markdown("<h1 style='text-align: center; color: white;'>Aplicação de Diagnóstico de Folhas</h1>", unsafe_allow_html=True)

# barra lateral para upload de imagens
st.sidebar.title('Configurações')
uploaded_images = st.sidebar.file_uploader("Escolha até 10 imagens (JPG ou JPEG)", type=['jpg', 'jpeg'], accept_multiple_files=True, key="upload_images")

model = get_model()

if uploaded_images:
    images_and_features = []
    
    st.markdown("<h3 style='text-align: center; color: white;'>Imagens e Previsões</h3>", unsafe_allow_html=True)

    for uploaded_image in uploaded_images:
        bytes_data = uploaded_image.getvalue()
        image = convert_byteio_image(bytes_data)

        if image.shape != (256, 256):
            image = cv2.resize(image, (256, 256))

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        features = extrair_caracteristicas_radiomics(gray_image)

        images_and_features.append((image, features))

    for idx, (img, features) in enumerate(images_and_features, start=1):
        prediction_probs = model.predict_proba([features]) 

        # class_names com os nomes reais das classes no seu conjunto de dados
        class_names = ['Saudável', 'Moscas das Galhas', 'Morta', 'Gorgulho', 'Antracnose', 'Cancro Bacteriano', 'Fumagina']
    
        class_idx = np.argmax(prediction_probs)
        predicted_class = class_names[class_idx]

        prediction_percentage = prediction_probs[0][class_idx] * 100
        prediction = f"{predicted_class} com {prediction_percentage:.2f}% de certeza"

        st.markdown(f"<h4 style='text-align: center; color: white'>Imagem {idx}: {prediction}</h4>", unsafe_allow_html=True)
        print(prediction_probs)
        #interpret_and_display_predictions(model, img, features)