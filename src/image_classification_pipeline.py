from sklearn.metrics import f1_score, cohen_kappa_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import pandas as pd
import pickle
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Função para calcular métricas
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

# Carregue os dados do arquivo CSV
data = pd.read_csv('C:/Users/cspau/Desktop/coisas do pc/Aprendendo Python/GitHub/leaf-diagnostic/etc/features.csv', delimiter=';')

# Identifique as colunas não numéricas
non_numeric_columns = data.select_dtypes(exclude=['number']).columns

# Remova as colunas não numéricas (se necessário)
data = data.drop(non_numeric_columns, axis=1)

# Certifique-se de que a coluna 'Label' seja numérica (binária) e substitua NaN por 0
data['Label'] = data['Label'].map({'classe1': 0, 'classe2': 1})  # Substitua com os rótulos reais
data['Label'] = data['Label'].fillna(0)  # Substitua 0 pelo valor desejado para NaN

# Verifique se há pelo menos duas classes diferentes nos rótulos
unique_classes = data['Label'].unique()
if len(unique_classes) < 2:
    raise ValueError("Número insuficiente de classes nos rótulos.")


# Lide com valores NaN nos rótulos (substitua NaN pelos valores desejados)
data['Label'] = data['Label'].fillna(0)  # Substitua 0 pelo valor desejado para NaN

# Separando X e Y e dividindo em conjunto de treinamento e teste
X = data.drop('Label', axis=1)
y = data['Label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Classificação binária com SVM
svm_model = SVC(random_state=42, kernel='rbf', C=1, gamma='scale')
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
metrics_svm = get_metrics(y_test, y_pred_svm)
print("Métricas para SVM:")
print(metrics_svm)

# Classificação com 5 classificadores
# Para o SVM
svm_model = SVC(random_state=42, kernel='rbf', C=1, gamma='scale')
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
metrics_svm = get_metrics(y_test, y_pred_svm)
print("Métricas para SVM:")
print(metrics_svm)

# Para o Random Forest
rf_model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
metrics_rf = get_metrics(y_test, y_pred_rf)
print("Métricas para Random Forest:")
print(metrics_rf)

# Para o KNN
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
metrics_knn = get_metrics(y_test, y_pred_knn)
print("Métricas para K-Nearest Neighbors (KNN):")
print(metrics_knn)

# Para o AdaBoost
adaboost_model = AdaBoostClassifier(random_state=42, n_estimators=50, learning_rate=1.0)
adaboost_model.fit(X_train, y_train)
y_pred_adaboost = adaboost_model.predict(X_test)
metrics_adaboost = get_metrics(y_test, y_pred_adaboost)
print("Métricas para AdaBoost (Gradient Boosting):")
print(metrics_adaboost)

# Para o XGB
xgb_model = xgb.XGBClassifier(objective='binary:logistic', random_state=42, max_depth=9, 
                              colsample_bytree=0.4033, min_child_weight=6, gamma=0.429, 
                              eta=0.5995, n_estimators=1000, use_label_encoder=False, 
                              eval_metric='merror')
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
metrics_xgb = get_metrics(y_test, y_pred_xgb)
print("Métricas para XGBoost:")
print(metrics_xgb)

# Criar DataFrames e salvar os resultados em um arquivo xlsx
df_metrics_svm = pd.DataFrame([metrics_svm.values()], columns=metrics_svm.keys())
df_metrics_svm.insert(0, 'Classificador', 'SVM')

df_metrics_rf = pd.DataFrame([metrics_rf.values()], columns=metrics_rf.keys())
df_metrics_rf.insert(0, 'Classificador', 'RF')

df_metrics_knn = pd.DataFrame([metrics_knn.values()], columns=metrics_knn.keys())
df_metrics_knn.insert(0, 'Classificador', 'KNN')

df_metrics_adaboost = pd.DataFrame([metrics_adaboost.values()], columns=metrics_adaboost.keys())
df_metrics_adaboost.insert(0, 'Classificador', 'AdaBoost')

df_metrics_xgb = pd.DataFrame([metrics_xgb.values()], columns=metrics_xgb.keys())
df_metrics_xgb.insert(0, 'Classificador', 'XGBoost')

# Concatenar todos os DataFrames
df_all_metrics = pd.concat([df_metrics_svm, df_metrics_rf, df_metrics_knn, df_metrics_adaboost, df_metrics_xgb])

# Salvar o DataFrame em um arquivo Excel
df_all_metrics.to_excel('metrics.xlsx', index=False)

df_all_metrics  # Caso queira ver o DataFrame no notebook
