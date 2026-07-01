#1. Imports
#================================================================================================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

!pip install scikit-optimize

#2. Project Paths
#================================================================================================================
BASE_DIR = '/content'
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

#3. Dataset Loading
#================================================================================================================
caminho_dados = os.path.join(PROCESSED_DIR, 'dataset_copa_pre_processado.csv')

try:
    df = pd.read_csv(caminho_dados)
    print(f"[+] Dataset carregado com sucesso: {df.shape[0]} partidas históricas.")
except FileNotFoundError:
    print(f"❌ Erro: Não foi possível encontrar o arquivo em {caminho_dados}")
    print("Verifique se o script de pré-processamento gerou o arquivo na pasta 'data/processed/'.")
    exit()

# Separar X (Características), e y (Rótulo: 0=Vitória B, 1=Empate, 2=Vitória A)
X = df.drop(columns=['resultado'])
y = df['resultado'].astype(int)

# Salvar a lista com os nomes originais das colunas antes de transformar para array
cols_features = list(X.columns)

#4. Math Treatment - Input e Scaler
#================================================================================================================
print("[-] Aplicando tratamento matemático (Imputer e Scaler)...")
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_imputed)

#5.  Data Split - Train (80%), Test (20%)
#================================================================================================================
# Divisão em Treino (80%) e Teste (20%) com estratificação de classes
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"-> Matriz de Treino original: {X_train.shape[0]} amostras.")
print(f"-> Matriz de Teste original: {X_test.shape[0]} amostras.")

#6. Training Session (Balanced)
#================================================================================================================
print("\n" + "="*30)
print(" INICIANDO A SESSÃO DE TREINO")
print("="*30)

# Balanceamento sintético via SMOTE (TREINO)
print("[-] Aplicando SMOTE para balancear as classes de treino (Evitando viés de empates)...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print(f"-> Matriz de Treino PÓS-BALANCEAMENTO: {X_train_balanced.shape[0]} amostras.")

print("[-] Treinando o modelo RandomForestClassifier (Comitê de Árvores)...")
# Hiperparâmetros para evitar overfitting no histórico de Copas
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf_model.fit(X_train_balanced, y_train_balanced)
print("[+] Treino finalizado com sucesso!")

#7. Testing Session (output Metrics)
#================================================================================================================
print("\n" + "="*40)
print(" INICIANDO A SESSÃO DE TESTE E MÉRICAS")
print("="*40)

# Predição utilizando a base de teste isolada
y_pred = rf_model.predict(X_test)

# Cálculo e exibição da Acurácia Global solicitada
acuracia = accuracy_score(y_test, y_pred)
print(f"📊 ACURÁCIA GLOBAL DO MODELO: {acuracia * 100:.2f}%\n")

# Relatório detalhado por Classe (Essencial para o preenchimento do relatório Word)
print("--- RELATÓRIO TÉCNICO DE CLASSIFICAÇÃO ---")
print(classification_report(y_test, y_pred, target_names=['Vitória B (0)', 'Empate (1)', 'Vitória A (2)']))

#8. Artifacts and Graph Plot
#================================================================================================================
print("\n[-] Gerando os arquivos visuais e salvando ecossistema...")

# Report 01: Matriz de Confusão
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Vitória B', 'Empate', 'Vitória A'])
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.title("Random Forest - Matriz de Confusão", fontsize=11, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, '01_rf_confusion_matrix.png'), dpi=300)
plt.close()

# Report 02: Feature Importance
fig, ax = plt.subplots(figsize=(8, 13))
importances = rf_model.feature_importances_
indices = np.argsort(importances)

plt.title("Random Forest - Importância das Variáveis", fontsize=11, fontweight='bold', pad=15)
plt.barh(range(len(indices)), importances[indices], color='#2ca02c', align='center')
plt.yticks(range(len(indices)), [cols_features[i] for i in indices])
plt.xlabel("Importância Relativa (Critério de Gini)")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, '02_rf_feature_importance.png'), dpi=300)
plt.close()

# Exportação dos arquivos .pkl para produção e uso no Streamlit
joblib.dump(rf_model, os.path.join(MODELS_DIR, 'rf_model.pkl'))
joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler_copa.pkl'))
joblib.dump(imputer, os.path.join(MODELS_DIR, 'imputer_copa.pkl'))
joblib.dump(cols_features, os.path.join(MODELS_DIR, 'features_esperadas.pkl'))

print("[+] Todos os artefatos salvos! Pronto para integração com o grupo.")


