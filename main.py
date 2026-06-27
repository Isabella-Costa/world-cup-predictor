import os
import pandas as pd
import joblib
import warnings
import time

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report

# Modelos
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


# 1. DICIONÁRIO DE MODELOS E PARÂMETROS (GRID)
CONFIGURACOES_MODELOS = {
    "Random Forest": {
        "modelo_base": RandomForestClassifier(random_state=42),
        "parametros": {
            "n_estimators": [50, 100, 200],       # Número de árvores
            "max_depth": [None, 10, 20],          # Profundidade da árvore
            "min_samples_split": [2, 5]           # Amostras mínimas para dividir
        }
    },
    "Rede Neural": {
        "modelo_base": MLPClassifier(random_state=42, early_stopping=True),
        "parametros": {
            "hidden_layer_sizes": [(64,), (128, 64)], # Arquitetura dos neurónios
            "activation": ['relu', 'tanh'],           # Funções de ativação
            "learning_rate_init": [0.001, 0.01]       # Taxa de aprendizagem
        }
    },
    "Regressao Logistica": {
        "modelo_base": LogisticRegression(random_state=42, max_iter=2000),
        "parametros": {
            "C": [0.1, 1.0, 10.0],                # Regularização (força)
            "solver": ['lbfgs', 'liblinear']      # Algoritmo de otimização
        }
    },
    "SVM": {
        "modelo_base": SVC(random_state=42, probability=True),
        "parametros": {
            "C": [0.1, 1, 10],                    # Margem de erro permitida
            "kernel": ['linear', 'rbf']           # Tipo de separação espacial
        }
    }
}

# ==========================================
# 2. FUNÇÃO: PREPARAÇÃO DOS DADOS
# ==========================================
def preparar_dados(caminho_csv):
    print(f"A carregar dados de: {caminho_csv}")
    df = pd.read_csv(caminho_csv)
    
    X = df.drop(columns=['resultado'])
    y = df['resultado'].astype(int)
    features_esperadas = list(X.columns)
    
    print("A aplicar tratamento (Imputer + Scaler)...")
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    print("A aplicar balanceamento SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    return X_train_balanced, X_test, y_train_balanced, y_test, scaler, imputer, features_esperadas

# 3. TUNING 
def tunar_e_avaliar(nome_modelo, config, X_train, y_train, X_test, y_test, pasta_modelos):
    print("\n" + "="*50)
    print(f"INICIAR TUNING: {nome_modelo}")
    print("="*50)
    
    modelo_base = config["modelo_base"]
    parametros = config["parametros"]
    
    # Configurar o GridSearchCV
    # cv=3 significa Validação Cruzada (divide o treino em 3 partes para testar)
    # n_jobs=-1 usa todos os núcleos do teu processador para ser mais rápido
    grid_search = GridSearchCV(estimator=modelo_base, param_grid=parametros, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    
    # Iniciar o treino intensivo
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    tempo_treino = time.time() - start_time
    
    # Extrair os melhores resultados
    melhor_modelo = grid_search.best_estimator_
    melhores_params = grid_search.best_params_
    
    print(f"\n✅ Tuning concluído em {tempo_treino:.2f} segundos!")
    print(f"Melhores Parâmetros Encontrados: {melhores_params}")
    
    # Avaliar o melhor modelo no conjunto de Teste 
    y_pred = melhor_modelo.predict(X_test)
    acuracia = accuracy_score(y_test, y_pred)
    
    print(f"\nAcurácia Final (Teste): {acuracia * 100:.2f}%")
    print("\nRelatório de Classificação Detalhado:")
    print(classification_report(y_test, y_pred))
    
    # Guardar o modelo campeão na pasta models
    caminho_salvar = os.path.join(pasta_modelos, f"{nome_modelo.replace(' ', '_').lower()}_tunnado.pkl")
    joblib.dump(melhor_modelo, caminho_salvar)
    print(f"Modelo guardado em: {caminho_salvar}")
    
    return melhor_modelo

# 4. EXECUÇÃO PRINCIPAL 
if __name__ == "__main__":
    # CONFIGURAÇÕES DE DIRETÓRIO 
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    CAMINHO_DADOS = os.path.join(PROCESSED_DIR, 'dataset_copa_pre_processado.csv')
    
    # ---------------------------------------------------------
    # SELETOR DE MODELO 
    # ---------------------------------------------------------
    # Opções válidas: "Random Forest", "Rede Neural", "Regressao Logistica", "SVM", ou "Todos"
    MODELO_ALVO = "Random Forest" 
    
    try:
        X_train, X_test, y_train, y_test, scaler, imputer, features = preparar_dados(CAMINHO_DADOS)
        
        # Salvar as ferramentas auxiliares
        joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler_copa.pkl'))
        joblib.dump(imputer, os.path.join(MODELS_DIR, 'imputer_copa.pkl'))
        joblib.dump(features, os.path.join(MODELS_DIR, 'features_esperadas.pkl'))
        
        # Executar lógica de tuning
        if MODELO_ALVO == "Todos":
            for nome, config in CONFIGURACOES_MODELOS.items():
                tunar_e_avaliar(nome, config, X_train, y_train, X_test, y_test, MODELS_DIR)
        elif MODELO_ALVO in CONFIGURACOES_MODELOS:
            tunar_e_avaliar(MODELO_ALVO, CONFIGURACOES_MODELOS[MODELO_ALVO], X_train, y_train, X_test, y_test, MODELS_DIR)
        else:
            print(f"❌ Erro: O modelo '{MODELO_ALVO}' não existe no dicionário.")
            
    except Exception as e:
        print(f"❌ Erro crítico na execução: {e}")