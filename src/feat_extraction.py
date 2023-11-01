import os
import cv2
import SimpleITK as sitk
from radiomics import featureextractor
import pandas as pd
import numpy as np
import glob

# Define as classes 
labels = {
    'Healthy': 0,
    'Gall Midge': 1,
    'Die Back': 2,
    'Cutting Weevil': 3,
    'Anthracnose': 4,
    'Bacterial Canker': 5,
    'Sooty Mould': 6
}

# extrair características e criar DataFrame
def extrair_caracteristicas(class_path, label):
    df = pd.DataFrame()

    for imagem_nome in os.listdir(class_path):
        if imagem_nome.endswith(".jpg"):
            # Carregue a imagem com OpenCV em escala de cinza
            imagem_path = os.path.join(class_path, imagem_nome)
            image_cv2 = cv2.imread(imagem_path, cv2.IMREAD_GRAYSCALE)

            # Converta para um objeto SimpleITK
            image = sitk.GetImageFromArray(image_cv2)

            # Crie uma máscara que cubra a imagem inteira
            mask = sitk.Image(image.GetSize(), sitk.sitkUInt8)
            mask.CopyInformation(image)
            mask = sitk.Cast(mask, sitk.sitkUInt8)
            mask = sitk.BinaryThreshold(mask, lowerThreshold=0, upperThreshold=1, insideValue=1, outsideValue=0)
            
            # Crie o extrator de características com PyRadiomics
            extractor = featureextractor.RadiomicsFeatureExtractor##

            # Calcule as características radiômicas
            result = extractor.execute(image, mask)

            # Organize as características em um DataFrame
            temp_df = pd.DataFrame(list(result.values()), index=result.keys(), columns=['Valor']).T

            # Adicione o rótulo à coluna 'Label'
            temp_df['Label'] = label

            # Adicione o DataFrame temporário ao DataFrame principal
            df = pd.concat([df, temp_df], ignore_index=True)
    
    return df

# Lista para armazenar as características
features_list = []

# Loop para processar as imagens
for class_name, label in labels.items():
    class_path = f'C:/Users/cspau/Desktop/coisas do pc/Aprendendo Python/GitHub/leaf-diagnostic/images/{class_name}'

    if not os.path.exists(class_path):
        print(f"A class_path '{class_path}' não existe.")
        continue

    df = extrair_caracteristicas(class_path, label)

    if df.empty:
        print(f"Nenhuma imagem encontrada em '{class_path}'.")
    else:
        features_list.append(df)

# Combine todos os DataFrames em um único DataFrame
if features_list:
    df = pd.concat(features_list, ignore_index=True)

    # Salve o DataFrame em um arquivo CSV
    output_path = 'C:/Users/cspau/Desktop/coisas do pc/Aprendendo Python/GitHub/leaf-diagnostic/etc/features.csv'
    df.to_csv(output_path, index=False, sep=';')
