import mahotas
import pandas as pd
import numpy as np
import glob
import cv2
import os


labels = ['Healthy', 'Gall Midge', 'Die Back', 'Cutting Weevil', 'Anthracnose', 'Bacterial Canker', 'Powdery Mildew', 'Sooty Mould']

# armazena as características
features_list = []

# calcular as características e adiciona-las a lista
def extract_features(image_path, label):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))

    features_img = mahotas.features.haralick(img, compute_14th_feature=True, return_mean=True)
    features_img = np.append(features_img, label)

    return features_img

# loop das pastas e imagens
for label in labels:
    path = f'C:/Users/cspau/Desktop/coisas do pc/Aprendendo Python/GitHub/leaf-diagnostic/images/{label}'

    if not os.path.exists(path):
        print(f"A pasta '{path}' não existe.")
        continue

    images_list = glob.glob(os.path.join(path, '*.jpg'))

    if not images_list:
        print(f"Nenhuma imagem encontrada em '{path}'.")
        continue

    label_mapping = {'Healthy': 0, 'Gall Midge': 1, 'Die Back': 2, 'Cutting Weevil': 3, 'Anthracnose': 4, 'Bacterial Canker': 5, 'Powdery Mildew': 6, 'Sooty Mould': 7}
    label_value = label_mapping.get(label, -1)

    for image_path in images_list:
        features = extract_features(image_path, label_value)
        features_list.append(features)

# nomes das características
features_names = mahotas.features.texture.haralick_labels
features_names = np.append(features_names, 'Label')

# salvar como CSV
df = pd.DataFrame(data=features_list, columns=features_names)

output_path = 'C:/Users/cspau/Desktop/coisas do pc/Aprendendo Python/GitHub/leaf-diagnostic/etc/features.csv'
df.to_csv(output_path, index=False, sep=';')
