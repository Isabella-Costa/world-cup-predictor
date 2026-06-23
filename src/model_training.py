import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report
import warnings

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
warnings.filterwarnings("ignore")

print("Comparação de Algoritmos:")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
caminho_dados = os.path.join(PROCESSED_DIR, 'dataset_copa_pre_processado.csv')

try:
    df = pd.read_csv(caminho_dados)
except FileNotFoundError:
    print(f"Erro: Não foi possível encontrar o arquivo em {caminho_dados}")
    exit()

X = df.drop(columns=['resultado'])
y = df['resultado'].astype(int)

# Tratamento e Escalonamento
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Divisão Treino/Teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Balanceamento (SMOTE)
print("A aplicar SMOTE nos dados de treino...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

modelos = {
    "Regressão Logística": LogisticRegression(random_state=42, max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
    "SVM": SVC(random_state=42, probability=True),
    "Rede Neural": MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', max_iter=500, early_stopping=True, random_state=42)
}

resultados = []

print("\n" + "="*50)
print("RESULTADOS DA AVALIAÇÃO MÚLTIPLA")
print("="*50)

for nome, modelo in modelos.items():
    print(f"\nA treinar: {nome}...")
    modelo.fit(X_train_balanced, y_train_balanced)
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    resultados.append({"Modelo": nome, "Acurácia": acc})
    
    print(f"Acurácia: {acc * 100:.2f}%")

# ==========================================
# RANKING FINAL
# ==========================================
print("\n" + "="*50)
print("AVALIAÇÃO FINAL DOS MODELOS")
print("="*50)
ranking_df = pd.DataFrame(resultados).sort_values(by="Acurácia", ascending=False)
print(ranking_df.to_string(index=False))
