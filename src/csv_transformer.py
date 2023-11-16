import pandas as pd

# Leitura do arquivo CSV original
df_original = pd.read_csv('C:/Users/cspau/Desktop/coisas do pc/Aprendendo Python/GitHub/leaf-diagnostic/etc/features.csv', delimiter=';')

# Seleção das colunas relevantes (excluindo informações não relacionadas às características radiômicas)
df_radiomics = df_original.iloc[:, 16:-1]

# Adição da coluna "Label" ao DataFrame de características radiômicas
df_radiomics['Label'] = df_original['Label']

# Salvar o novo DataFrame em um novo arquivo CSV
df_radiomics.to_csv('C:/Users/cspau/Desktop/coisas do pc/Aprendendo Python/GitHub/leaf-diagnostic/etc/features_ok.csv', index=False)
