import os
import cv2
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import pickle
import xgboost as xgb
from radiomics import featureextractor
import streamlit as st
import SimpleITK as sitk

# Função para calcular métricas de desempenho
def get_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    accuracy = np.trace(cm) / np.sum(cm)
    precision = np.diag(cm) / np.sum(cm, axis=0)
    recall = np.diag(cm) / np.sum(cm, axis=1)

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall}

# Carregar dados do arquivo CSV
file_path = 'C:/Users/cspau/Desktop/coisas do pc/Aprendendo Python/GitHub/leaf-diagnostic/etc/features.csv'
data = pd.read_csv(file_path, delimiter=';')

# Remover colunas não numéricas
data = data.select_dtypes(include=['number'])

# Verificar classes únicas nos rótulos
unique_classes = data['Label'].unique()
if len(unique_classes) < 2:
    raise ValueError("Número insuficiente de classes nos rótulos.")

# Lidar com valores NaN nos rótulos
data['Label'] = data['Label'].fillna(0)

# Separar X e Y e aplicar SMOTE para lidar com desbalanceamento
X = data.drop('Label', axis=1)
y = data['Label'].values
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Dividir em conjunto de treinamento e teste com os dados resampleados
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Aplicar SelectKBest para selecionar melhores características
k_best_selector = SelectKBest(f_classif, k=36)
X_train_selected = k_best_selector.fit_transform(X_train, y_train)
X_test_selected = k_best_selector.transform(X_test)

# Parâmetros para otimização do modelo XGBoost
param_grid = {
    'max_depth': [10, 20],
    'min_child_weight': [100, 60],
    'gamma': [0.5, 1],
    'subsample': [0.8, 1.5],
    'colsample_bytree': [0.77, 1],
    'eta': [0.3, 0.1],
    'n_estimators': [200, 400],
}

# Configurar e realizar busca em grade
grid_search = GridSearchCV(
    estimator=xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='merror', random_state=42),
    param_grid=param_grid, 
    scoring='accuracy', 
    cv=3
)
grid_search.fit(X_train_selected, y_train)

# Exibir melhores parâmetros encontrados
print("Melhores parâmetros encontrados:")
print(grid_search.best_params_)

# Usar melhores parâmetros para treinar o modelo
best_xgb_model = grid_search.best_estimator_
best_xgb_model.fit(X_train_selected, y_train)

# Salvar o modelo treinado
model_file_path = 'C:/Users/cspau/Desktop/coisas do pc/Aprendendo Python/GitHub/leaf-diagnostic/etc/best_xgboost_model.dat'
with open(model_file_path, 'wb') as model_file:
    pickle.dump(best_xgb_model, model_file)

# Avaliar o modelo com características selecionadas
y_pred_best_xgb = best_xgb_model.predict(X_test_selected)
metrics_best_xgb = get_metrics(y_test, y_pred_best_xgb)

# Exibir métricas
print("Métricas para XGBoost (com melhores parâmetros e características selecionadas):")
print(metrics_best_xgb)
