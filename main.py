import os
import pandas as pd
# pyrefly: ignore [missing-import]
import joblib
import warnings
import time
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns
# pyrefly: ignore [missing-import]
from skopt import BayesSearchCV
# pyrefly: ignore [missing-import]
from skopt.space import Real, Categorical, Integer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
# pyrefly: ignore [missing-import]
from imblearn.over_sampling import SMOTE, RandomOverSampler
from collections import Counter

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Modelos
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

warnings.filterwarnings("ignore")



# 1. DICIONÁRIO DE MODELOS E PARÂMETROS (BAYESIANO)
CONFIGURACOES_MODELOS = {
    "Random Forest": {
        "modelo_base": RandomForestClassifier(random_state=42, class_weight='balanced'),
        "parametros": {
            "n_estimators": Integer(50, 300),         # Qualquer inteiro entre 50 e 300
            "max_depth": Integer(5, 30),              # Profundidade entre 5 e 30
            "min_samples_split": Integer(2, 10)       # Amostras entre 2 e 10
        }
    },
    "Rede Neural": {
        "modelo_base": MLPClassifier(random_state=42, early_stopping=True),
        "parametros": {
            "hidden_layer_sizes": Integer(15, 50), 
            "activation": Categorical(['relu', 'tanh']), 
            "solver": Categorical(['adam', 'sgd']),
            "alpha": Real(0.0001, 1.0, prior='log-uniform'),
            "learning_rate_init": Real(0.001, 0.01, prior='log-uniform')
        }
    },
    "Regressao Logistica": {
        "modelo_base": LogisticRegression(random_state=42, max_iter=2000, class_weight='balanced'),
        "parametros": {
            "C": Real(0.001, 100.0, prior='log-uniform'), 
            "penalty": Categorical(['l2']),                    
            "solver": Categorical(['lbfgs', 'newton-cg', 'saga']) 
        }
    },
    "SVM": {
        "modelo_base": SVC(random_state=42, probability=True, class_weight='balanced'),
        "parametros": {
            "C": Real(0.1, 50.0, prior='log-uniform'),                    
            "kernel": Categorical(['linear', 'rbf'])           
        }
    }
}

# 2.  PREPARAÇÃO DOS DADOS
def aplicar_engenharia_features(df):
    # Diferença de Ranking 
    df['dif_rank'] = df['rank_a'] - df['rank_b']

    # Razão de Valor de Mercado 
    df['razao_valor'] = df['valor_a'] / (df['valor_b'] + 1)

    # Eficiência Ofensiva 
    df['eficiencia_ataque_a'] = df['gols_elite_a'] / (df['minutos_elite_a'] + 1)
    df['eficiencia_ataque_b'] = df['gols_elite_b'] / (df['minutos_elite_b'] + 1)

    # Incompatibilidade Tática 
    df['vantagem_ofensiva_a'] = df['eficiencia_ataque_a'] - df['eficiencia_ataque_b']
    
    return df

def preparar_dados(caminho_csv, tipo_balanceamento='ROS'):
    print(f"A carregar dados de: {caminho_csv}")
    df = pd.read_csv(caminho_csv)

    df = aplicar_engenharia_features(df)
    
    X = df.drop(columns=['resultado'])
    y = df['resultado'].astype(int)
    features_esperadas = list(X.columns)
    
    # DIVIDIR PRIMEIRO 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # IMPUTAÇÃO 
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test) # Usar apenas transform!

    # ESCALONAMENTO
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed) # Usar apenas transform!

    # BALANCEAMENTO 
    print(f"A aplicar balanceamento {tipo_balanceamento}...")
    
    if tipo_balanceamento == 'SMOTE':
        sampler = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = sampler.fit_resample(X_train_scaled, y_train)
    elif tipo_balanceamento == 'ROS':
        sampler = RandomOverSampler(random_state=42)
        X_train_balanced, y_train_balanced = sampler.fit_resample(X_train_scaled, y_train)
    elif tipo_balanceamento == 'MIX':
        contagem = Counter(y_train)
        max_count = max(contagem.values())
        alvo_smote = {k: max(v, int(0.5 * max_count)) for k, v in contagem.items()}
        
        smote = SMOTE(sampling_strategy=alvo_smote, random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
        
        ros = RandomOverSampler(random_state=42)
        X_train_balanced, y_train_balanced = ros.fit_resample(X_train_smote, y_train_smote)
    else:
        X_train_balanced, y_train_balanced = X_train_scaled, y_train
    
    return X_train_balanced, X_test_scaled, y_train_balanced, y_test, scaler, imputer, features_esperadas

# 3. TUNING 
def formatar_nome_pasta(nome):
    palavras = nome.split()
    if len(palavras) == 1:
        return palavras[0].lower()
    return palavras[0].lower() + ''.join(p.capitalize() for p in palavras[1:])

def tunar_e_avaliar(nome_modelo, config, X_train, y_train, X_test, y_test, pasta_modelos, pasta_relatorios, scaler, imputer, features):
    nome_pasta_modelo = formatar_nome_pasta(nome_modelo)
    pasta_modelo_especifico = os.path.join(pasta_modelos, nome_pasta_modelo)
    pasta_relatorios_especifico = os.path.join(pasta_relatorios, nome_pasta_modelo)
    
    os.makedirs(pasta_modelo_especifico, exist_ok=True)
    os.makedirs(pasta_relatorios_especifico, exist_ok=True)

    print("\n" + "="*50)
    print(f"INICIAR TUNING BAYESIANO: {nome_modelo}")
    print("="*50)
    
    modelo_base = config["modelo_base"]
    parametros = config["parametros"]
    
    otimizador = BayesSearchCV(
        estimator=modelo_base,
        search_spaces=parametros,
        n_iter=15,             # Quantas combinações ele vai tentar (aumente para 30 ou 50 se tiver tempo)
        cv=3,                  # Cross-validation
        scoring='accuracy',    # Métrica alvo
        n_jobs=-1,             # Usa todos os núcleos do processador
        random_state=42,       # Para reprodutibilidade
        verbose=1
    )
    
    start_time = time.time()
    otimizador.fit(X_train, y_train)
    tempo_treino = time.time() - start_time
    
    melhor_modelo = otimizador.best_estimator_
    melhores_params = otimizador.best_params_
    
    print(f"\n✅ Tuning Bayesiano concluído em {tempo_treino:.2f} segundos!")
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
    nome_pasta_matriz = os.path.join(pasta_relatorios_especifico, f"matriz_confusao_{nome_pasta_modelo}.png")
    plt.savefig(nome_pasta_matriz)
    plt.close() # Fecha o gráfico na memória para não sobrecarregar
    print(f"Gráfico da Matriz de Confusão guardado em: {nome_pasta_matriz}")
    
    caminho_salvar = os.path.join(pasta_modelo_especifico, f"{nome_pasta_modelo}_tunnado.pkl")
    joblib.dump(melhor_modelo, caminho_salvar)
    print(f"Modelo guardado em: {caminho_salvar}")
    
    # Salvar as ferramentas auxiliares específicas do modelo
    joblib.dump(scaler, os.path.join(pasta_modelo_especifico, 'scaler_copa.pkl'))
    joblib.dump(imputer, os.path.join(pasta_modelo_especifico, 'imputer_copa.pkl'))
    joblib.dump(features, os.path.join(pasta_modelo_especifico, 'features_esperadas.pkl'))
    
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
    MODELO_ALVO = "Rede Neural" 
    TIPO_BALANCEAMENTO = "ROS" # Pode ser "SMOTE", "ROS" ou "MIX"
    
    try:
        X_train, X_test, y_train, y_test, scaler, imputer, features = preparar_dados(CAMINHO_DADOS, tipo_balanceamento=TIPO_BALANCEAMENTO)
        
        # Executar lógica de tuning
        if MODELO_ALVO == "Todos":
            for nome, config in CONFIGURACOES_MODELOS.items():
                tunar_e_avaliar(nome, config, X_train, y_train, X_test, y_test, MODELS_DIR, REPORTS_DIR, scaler, imputer, features)
        elif MODELO_ALVO in CONFIGURACOES_MODELOS:
            tunar_e_avaliar(MODELO_ALVO, CONFIGURACOES_MODELOS[MODELO_ALVO], X_train, y_train, X_test, y_test, MODELS_DIR, REPORTS_DIR, scaler, imputer, features)
        else:
            print(f"❌ Erro: O modelo '{MODELO_ALVO}' não existe no dicionário.")
            
    except Exception as e:
        print(f"❌ Erro crítico na execução: {e}")