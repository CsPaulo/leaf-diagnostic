from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle

def get_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    accuracy = cm.diagonal().sum() / cm.sum()
    precision = cm.diagonal() / cm.sum(axis=0)
    recall = cm.diagonal() / cm.sum(axis=1)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
    }

# Carregar os dados
df = pd.read_csv('C:/Users/cspau/Desktop/coisas do pc/Aprendendo Python/GitHub/leaf-diagnostic/etc/features.csv', delimiter=';')

X = df.drop('Label', axis=1)
y = df['Label']

# Dados de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SMOTE para lidar com o desbalanceamento de classes
balanced = SMOTE(random_state=42)
X_train, y_train = balanced.fit_resample(X_train, y_train)

# Seleção de características com SelectKBest
k_best_features = 14
selector = SelectKBest(f_classif, k=k_best_features)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# RandomForestClassifier com hiperparâmetros ajustados
param_grid = {
    'n_estimators': 2000,
    'max_depth': 160,
    'min_samples_split': 48,
    'min_samples_leaf': 24,
    'max_features': 'sqrt',
    'criterion': 'gini',
    'bootstrap': True
}

rf_model = RandomForestClassifier(**param_grid)
rf_model.fit(X_train_selected, y_train)
y_rf_pred = rf_model.predict(X_test_selected)

rf_metrics = get_metrics(y_test, y_rf_pred)
print("Random Forest Métricas:")
print(rf_metrics)

# Relatório de Classificação
print("Relatório de Classificação Random Forest:")
print(classification_report(y_test, y_rf_pred))

# Salvar o modelo
pickle.dump(rf_model, open('C:/Users/cspau/Desktop/coisas do pc/Aprendendo Python/GitHub/leaf-diagnostic/etc/best_random_forest_model.dat', 'wb'))
