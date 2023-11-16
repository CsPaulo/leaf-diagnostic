from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle
import xgboost as xgb

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

# Verifique se há pelo menos duas classes diferentes nos rótulos
unique_classes = data['Label'].unique()
if len(unique_classes) < 2:
    raise ValueError("Número insuficiente de classes nos rótulos.")

# Lide com valores NaN nos rótulos (substitua NaN pelos valores desejados)
data['Label'] = data['Label'].fillna(0)  # Substitua 0 pelo valor desejado para NaN

# Separando X e Y e dividindo em conjunto de treinamento e teste
X = data.drop('Label', axis=1)
y = data['Label'].values

# Separando X e Y e dividindo em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Para o XGB
xgb_model = xgb.XGBClassifier(objective='binary:logistic', max_depth=30, 
                              colsample_bytree=0.7033, min_child_weight=18, gamma=0.729, 
                              eta=0.8995, n_estimators=4000, use_label_encoder=False, 
                              eval_metric='merror')
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
metrics_xgb = get_metrics(y_test, y_pred_xgb)
print("Métricas para XGBoost:")
print(metrics_xgb)

# Relatório de Classificação
print("Relatório de Classificação XGBoost:")
print(classification_report(y_test, y_pred_xgb))

# Salvar o modelo
pickle.dump(xgb_model, open('C:/Users/cspau/Desktop/coisas do pc/Aprendendo Python/GitHub/leaf-diagnostic/etc/best_xgboost_model.dat', 'wb'))
