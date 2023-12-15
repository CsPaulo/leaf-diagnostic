import os
import cv2
import SimpleITK as sitk
import pandas as pd
from radiomics import featureextractor
import numpy as np

# Defina os rótulos das classes
labels = {'Healthy': 0, 'Die Back': 1, 'Cutting Weevil': 2}

def extrair_caracteristicas(imagem_path, label):
    # Carregue a imagem em escala de cinza
    image_cv2 = cv2.imread(imagem_path, cv2.IMREAD_GRAYSCALE)
    image = sitk.GetImageFromArray(image_cv2)

    # Crie uma máscara que cubra toda a imagem
    mask = sitk.Image(image.GetSize(), sitk.sitkUInt8)
    mask.CopyInformation(image)
    mask = sitk.BinaryThreshold(mask, lowerThreshold=0, upperThreshold=1, insideValue=1, outsideValue=0)

    # Crie o extrator de características Radiomics
    extractor = featureextractor.RadiomicsFeatureExtractor(shape2D=True)

    # Calcule as características radiômicas
    result = extractor.execute(image, mask)

    # Adicione o rótulo à coluna 'Label'
    result['Label'] = label

    return result

# Lista para armazenar as características
all_features = []

# Loop para processar as imagens de todas as classes
for class_name, label in labels.items():
    class_path = f'C:/Users/cspau/Desktop/coisas do pc/Aprendendo Python/GitHub/leaf-diagnostic/images/{class_name}'

    if not os.path.exists(class_path):
        print(f"A pasta '{class_path}' não existe.")
        continue

    # Extrai características para cada imagem na classe
    for imagem_nome in os.listdir(class_path):
        if imagem_nome.endswith(".jpg"):
            imagem_path = os.path.join(class_path, imagem_nome)
            class_features = extrair_caracteristicas(imagem_path, label)
            all_features.append(class_features)

# Combine todas as características em um único DataFrame
if all_features:
    df = pd.DataFrame(all_features)

    # Salve o DataFrame em um arquivo CSV
    output_path = 'C:/Users/cspau/Desktop/coisas do pc/Aprendendo Python/GitHub/leaf-diagnostic/etc/features.csv'
    df.to_csv(output_path, index=False, sep=';')
