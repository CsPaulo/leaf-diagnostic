import os
import cv2
import SimpleITK as sitk
from radiomics import featureextractor
import pandas as pd
import numpy as np
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

def extrair_caracteristicas(imagem):
    # Verifique se a imagem já está em escala de cinza
    if len(imagem.shape) == 2 or (len(imagem.shape) == 3 and imagem.shape[2] == 1):
        image_cv2 = imagem
    else:
        # A imagem não está em escala de cinza, converta para escala de cinza
        image_cv2 = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)

    # Crie uma imagem SimpleITK
    image = sitk.GetImageFromArray(image_cv2)

    # Crie uma máscara que cubra toda a imagem
    mask = sitk.Image(image.GetSize(), sitk.sitkUInt8)
    mask.CopyInformation(image)
    mask = sitk.BinaryThreshold(mask, lowerThreshold=0, upperThreshold=1, insideValue=1, outsideValue=0)

    # Crie o extrator de características Radiomics
    extractor = featureextractor.RadiomicsFeatureExtractor(shape2D=True)

    # Calcule as características radiômicas
    result = extractor.execute(image, mask)

    # Converta as características para um array numpy
    features_array = np.concatenate([np.atleast_1d(result[key]).ravel() for key in result.keys()])

    return features_array