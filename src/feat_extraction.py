import os
import cv2
import SimpleITK as sitk
from radiomics import featureextractor
import pandas as pd

# Defina os rótulos das classes
labels = {
    'Healthy': 0,
    'Gall Midge': 1,
    'Die Back': 2,
    'Cutting Weevil': 3,
    'Anthracnose': 4,
    'Bacterial Canker': 5,
    'Sooty Mould': 6
}

# Função para extrair características radiômicas de uma pasta de imagens
def extrair_caracteristicas(pasta, label):
    features_list = []

    # Crie o extrator de características Radiomics
    extractor = featureextractor.RadiomicsFeatureExtractor(shape2d=True)

    for imagem_nome in os.listdir(pasta):
        if imagem_nome.endswith(".jpg"):
            # Carregue a imagem em escala de cinza
            imagem_path = os.path.join(pasta, imagem_nome)
            image_cv2 = cv2.imread(imagem_path, cv2.IMREAD_GRAYSCALE)

            # Converta para um objeto SimpleITK
            image = sitk.GetImageFromArray(image_cv2)

            # Crie uma máscara que cubra toda a imagem
            mask = sitk.Image(image.GetSize(), sitk.sitkUInt8)
            mask.CopyInformation(image)
            mask = sitk.BinaryThreshold(mask, lowerThreshold=0, upperThreshold=1, insideValue=1, outsideValue=0)

            # Calcule as características radiômicas
            result = extractor.execute(image, mask)

            # Adicione o rótulo à coluna 'Label'
            result['Label'] = label

            # Adicione as características ao vetor
            features_list.append(result)

    return features_list

# Lista para armazenar as características
all_features = []

# Loop para processar as imagens de todas as classes
for class_name, label in labels.items():
    class_path = f'C:/Users/cspau/Desktop/coisas do pc/Aprendendo Python/GitHub/leaf-diagnostic/images/{class_name}'

    if not os.path.exists(class_path):
        print(f"A pasta '{class_path}' não existe.")
        continue

    class_features = extrair_caracteristicas(class_path, label)
    all_features.extend(class_features)

# Combine todas as características em um único DataFrame
if all_features:
    df = pd.DataFrame(all_features)

    # Salve o DataFrame em um arquivo CSV
    output_path = 'C:/Users/cspau/Desktop/coisas do pc/Aprendendo Python/GitHub/leaf-diagnostic/etc/features.csv'
    df.to_csv(output_path, index=False, sep=';')
