import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

print(" A iniciar o pipeline: Random Forest...")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

os.makedirs(MODELS_DIR, exist_ok=True)
caminho_dados = os.path.join(PROCESSED_DIR, 'dataset_copa_pre_processado.csv')

try:
    df = pd.read_csv(caminho_dados)
    print(f"Dataset carregado: {df.shape[0]} partidas.")
except FileNotFoundError:
    print(f"❌ Erro: Não foi possível encontrar o ficheiro em {caminho_dados}")
    exit()

# Separar X (Características) e y (Rótulo)
X = df.drop(columns=['resultado'])
y = df['resultado'].astype(int)

cols_features = list(X.columns)
print("A aplicar o pré-processamento matemático (Imputer e Scaler)...")

imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Divisão em Treino (80%) e Teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Balanceamento (SMOTE)
print("A aplicar o SMOTE para balancear a matriz de treino...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Treino do Modelo Campeão (Random Forest)
print("A treinar a Floresta Aleatória (RandomForestClassifier)...")
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)

rf_model.fit(X_train_balanced, y_train_balanced)

y_pred = rf_model.predict(X_test)
acuracia = accuracy_score(y_test, y_pred)

print("\n" + "="*50)
print(f"RESULTADOS DE PRODUÇÃO: RANDOM FOREST")
print("="*50)
print(f"Acurácia Global: {acuracia * 100:.2f}%\n")
print(classification_report(y_test, y_pred, target_names=['Vitória B (0)', 'Empate (1)', 'Vitória A (2)']))

print("\nA exportar o ecossistema preditivo para a pasta 'models/'...")

joblib.dump(rf_model, os.path.join(MODELS_DIR, 'random_forest_copa.pkl'))
joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler_copa.pkl'))
joblib.dump(imputer, os.path.join(MODELS_DIR, 'imputer_copa.pkl'))
joblib.dump(cols_features, os.path.join(MODELS_DIR, 'features_esperadas.pkl'))

print("Sucesso! O modelo está pronto.")