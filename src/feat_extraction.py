import mahotas
import pandas as pd
import numpy as np
import glob
import cv2
import os

# Nome das pastas
labels = ['Healthy', 'Gall_Midge']

# Lista que armazena as características
features_list = []

# Função para calcular as características e adiciona-las a lista
def extract_features(image_path, label):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))

    features_img = mahotas.features.haralick(img, compute_14th_feature=True, return_mean=True)
    features_img = np.append(features_img, label)

    return features_img

# Loop das pastas e imagens
for label in labels:
    path = f'C:/Users/cspau/Desktop/coisas do pc/Aprendendo Python/GitHub/leaf-diagnostic/images/{label}'

    if not os.path.exists(path):
        print(f"A pasta '{path}' não existe.")
        continue

    images_list = glob.glob(os.path.join(path, '*.jpg'))

    if not images_list:
        print(f"Nenhuma imagem encontrada em '{path}'.")
        continue

    label_mapping = {'Healthy': 0, 'Gall_Midge': 1}
    label_value = label_mapping.get(label, -1)

    for image_path in images_list:
        features = extract_features(image_path, label_value)
        features_list.append(features)

# Nomes das características
features_names = mahotas.features.texture.haralick_labels
features_names = np.append(features_names, 'Label')

# DataFrame e salvar ele como CSV
df = pd.DataFrame(data=features_list, columns=features_names)

output_path = 'C:/Users/cspau/Desktop/coisas do pc/Aprendendo Python/GitHub/leaf-diagnostic/etc/features.csv'
df.to_csv(output_path, index=False, sep=';')
