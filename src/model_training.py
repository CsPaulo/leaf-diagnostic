from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
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

# carregar os dados
df = pd.read_csv('C:/Users/cspau/Desktop/coisas do pc/Aprendendo Python/GitHub/leaf-diagnostic/etc/features.csv', delimiter=';')

X = df.drop('Label', axis=1)
y = df['Label']


# dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SMOTE para lidar com o desbalanceamento de classes
balanced = SMOTE(random_state=42)
X_train, y_train = balanced.fit_resample(X_train, y_train)


# RandomForestClassifier com GridSearchCV
rf_params = {
    'n_estimators': [100, 300, 500],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, n_jobs=-1)
rf_grid.fit(X_train, y_train)
best_rf_model = rf_grid.best_estimator_
y_rf_pred = best_rf_model.predict(X_test)
rf_metrics = get_metrics(y_test, y_rf_pred)
print("Random Forest Métricas:")
print(rf_metrics)

# relatório de classificação
print("Relatório de Classificação Random Forest :")
print(classification_report(y_test, y_rf_pred))

# Salve o modelo treinado
pickle.dump(best_rf_model, open('C:/Users/cspau/Desktop/coisas do pc/Aprendendo Python/GitHub/leaf-diagnostic/etc/best_random_forest_model.dat', 'wb'))
