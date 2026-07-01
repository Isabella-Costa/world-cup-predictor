import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PASTA_MODELO_ESPECIFICA = os.path.join(MODELS_DIR, 'redeNeural')

# Configuração de balanceamento: 'SMOTE', 'ROS' ou 'MIX'
TIPO_BALANCEAMENTO = 'ROS'

os.makedirs(PASTA_MODELO_ESPECIFICA, exist_ok=True)
caminho_dados = os.path.join(PROCESSED_DIR, 'dataset_copa_pre_processado.csv')

try:
    df = pd.read_csv(caminho_dados)
    print(f"Dataset carregado: {df.shape[0]} partidas.")
except FileNotFoundError:
    print(f"❌ Erro: Não foi possível encontrar a pasta em {caminho_dados}")
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

# Balanceamento
print(f"A aplicar o balanceamento {TIPO_BALANCEAMENTO} para balancear a matriz de treino...")
if TIPO_BALANCEAMENTO == 'SMOTE':
    sampler = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = sampler.fit_resample(X_train, y_train)
elif TIPO_BALANCEAMENTO == 'ROS':
    sampler = RandomOverSampler(random_state=42)
    X_train_balanced, y_train_balanced = sampler.fit_resample(X_train, y_train)
elif TIPO_BALANCEAMENTO == 'MIX':
    # A MIX ativa o SMOTE até 50% e depois passa a ser ROS
    contagem = Counter(y_train)
    max_count = max(contagem.values())
    alvo_smote = {k: max(v, int(0.5 * max_count)) for k, v in contagem.items()}
    
    smote = SMOTE(sampling_strategy=alvo_smote, random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    ros = RandomOverSampler(random_state=42)
    X_train_balanced, y_train_balanced = ros.fit_resample(X_train_smote, y_train_smote)
else:
    X_train_balanced, y_train_balanced = X_train, y_train

print("A treinar a Rede Neural (MLP)...")
mlp_model = MLPClassifier(random_state=42, early_stopping=True)

mlp_model.fit(X_train_balanced, y_train_balanced)

y_pred = mlp_model.predict(X_test)
acuracia = accuracy_score(y_test, y_pred)

print("\n" + "="*50)
print(f"RESULTADOS DE PRODUÇÃO: REDE NEURAL (MLP)")
print("="*50)
print(f"Acurácia Global: {acuracia * 100:.2f}%\n")
print(classification_report(y_test, y_pred, target_names=['Vitória B (0)', 'Empate (1)', 'Vitória A (2)']))

print("\nA exportar o ecossistema preditivo para a pasta 'models/redeNeural'...")

joblib.dump(mlp_model, os.path.join(PASTA_MODELO_ESPECIFICA, 'mlp_model.pkl'))
joblib.dump(scaler, os.path.join(PASTA_MODELO_ESPECIFICA, 'scaler_copa.pkl'))
joblib.dump(imputer, os.path.join(PASTA_MODELO_ESPECIFICA, 'imputer_copa.pkl'))
joblib.dump(cols_features, os.path.join(PASTA_MODELO_ESPECIFICA, 'features_esperadas.pkl'))

print("Sucesso! O modelo está pronto.")
