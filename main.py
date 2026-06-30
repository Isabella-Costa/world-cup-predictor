import os
import pandas as pd
# pyrefly: ignore [missing-import]
import joblib
import warnings
import time
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
# pyrefly: ignore [missing-import]
from imblearn.over_sampling import SMOTE

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
# 3. TUNING 
def tunar_e_avaliar(nome_modelo, config, X_train, y_train, X_test, y_test, pasta_modelos, pasta_relatorios):
    print("\n" + "="*50)
    print(f"INICIAR TUNING: {nome_modelo}")
    print("="*50)
    
    modelo_base = config["modelo_base"]
    parametros = config["parametros"]
    
    grid_search = GridSearchCV(estimator=modelo_base, param_grid=parametros, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    tempo_treino = time.time() - start_time
    
    melhor_modelo = grid_search.best_estimator_
    melhores_params = grid_search.best_params_
    
    print(f"\n✅ Tuning concluído em {tempo_treino:.2f} segundos!")
    print(f"Melhores Parâmetros Encontrados: {melhores_params}")
    
    y_pred = melhor_modelo.predict(X_test)
    acuracia = accuracy_score(y_test, y_pred)
    
    print(f"\nAcurácia Final (Teste): {acuracia * 100:.2f}%")
    print("\nRelatório de Classificação Detalhado:")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    labels = ['Derrota (0)', 'Empate (1)', 'Vitória (2)']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, cbar=True)
    plt.title(f'Matriz de Confusão - {nome_modelo}', fontsize=14, pad=15)
    plt.ylabel('Resultado Real', fontsize=12)
    plt.xlabel('Previsão do Modelo', fontsize=12)
    plt.tight_layout()
    
    # Guardar a imagem na pasta reports
    nome_ficheiro_matriz = os.path.join(pasta_relatorios, f"matriz_confusao_{nome_modelo.replace(' ', '_').lower()}.png")
    plt.savefig(nome_ficheiro_matriz)
    plt.close() # Fecha o gráfico na memória para não sobrecarregar
    print(f"Gráfico da Matriz de Confusão guardado em: {nome_ficheiro_matriz}")
    # ==========================================
    
    caminho_salvar = os.path.join(pasta_modelos, f"{nome_modelo.replace(' ', '_').lower()}_tunnado.pkl")
    joblib.dump(melhor_modelo, caminho_salvar)
    print(f"Modelo guardado em: {caminho_salvar}")
    
    return melhor_modelo

# 4. EXECUÇÃO PRINCIPAL 
if __name__ == "__main__":
    # CONFIGURAÇÕES DE DIRETÓRIO 
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    os.makedirs(MODELS_DIR, exist_ok=True)

    REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
    os.makedirs(REPORTS_DIR, exist_ok=True)

    CAMINHO_DADOS = os.path.join(PROCESSED_DIR, 'dataset_copa_pre_processado.csv')
    
    # ---------------------------------------------------------
    # SELETOR DE MODELO 
    # ---------------------------------------------------------
    # Opções válidas: "Random Forest", "Rede Neural", "Regressao Logistica", "SVM", ou "Todos"
    MODELO_ALVO = "Regressao Logistica" 
    
    try:
        X_train, X_test, y_train, y_test, scaler, imputer, features = preparar_dados(CAMINHO_DADOS)
        
        # Salvar as ferramentas auxiliares
        joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler_copa.pkl'))
        joblib.dump(imputer, os.path.join(MODELS_DIR, 'imputer_copa.pkl'))
        joblib.dump(features, os.path.join(MODELS_DIR, 'features_esperadas.pkl'))
        
        # Executar lógica de tuning
        if MODELO_ALVO == "Todos":
            for nome, config in CONFIGURACOES_MODELOS.items():
                # Faltava passar o REPORTS_DIR aqui no final 👇
                tunar_e_avaliar(nome, config, X_train, y_train, X_test, y_test, MODELS_DIR, REPORTS_DIR)
        elif MODELO_ALVO in CONFIGURACOES_MODELOS:
            # E faltava passar aqui também 👇
            tunar_e_avaliar(MODELO_ALVO, CONFIGURACOES_MODELOS[MODELO_ALVO], X_train, y_train, X_test, y_test, MODELS_DIR, REPORTS_DIR)
        else:
            print(f"❌ Erro: O modelo '{MODELO_ALVO}' não existe no dicionário.")
            
    except Exception as e:
        print(f"❌ Erro crítico na execução: {e}")