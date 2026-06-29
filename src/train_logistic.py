import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

def treinar_modelo():
    caminho_dados = 'data/copa2026_dataset.csv'
    pasta_modelos = 'models'
    
    if not os.path.exists(caminho_dados):
        print(f"Erro: Arquivo de dados não encontrado em '{caminho_dados}'.")
        return

    print("Carregando os dados da Copa...")
    df = pd.read_csv(caminho_dados)

    print("Criando a coluna alvo 'is_top_16' baseada no Rank FIFA...")
    df['is_top_16'] = (df['rank_fifa'] <= 16).astype(int)
    coluna_alvo = 'is_top_16'

    colunas_para_remover = [coluna_alvo, 'selecao', 'conf', 'continente', 'rank_fifa']
    X = df.select_dtypes(include=['int64', 'float64']).drop(columns=colunas_para_remover, errors='ignore')
    y = df[coluna_alvo]

    print("\nFeatures numéricas que o modelo vai usar para prever:")
    print(list(X.columns))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()), 
        ('logistic', LogisticRegression(max_iter=1000, random_state=42))
    ])

    print("\nTreinando a LogisticRegression (Gerando a Baseline)...")
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    acuracia = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*40)
    print(f"RESULTADO DA BASELINE (Acurácia): {acuracia:.2%}")
    print("="*40)
    print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred, zero_division=0))
    
    os.makedirs(pasta_modelos, exist_ok=True)
    caminho_salvar = os.path.join(pasta_modelos, 'logistic_model.pkl')
    joblib.dump(pipeline, caminho_salvar)
    print(f"\nSucesso! Modelo salvo em: '{caminho_salvar}'")

if __name__ == "__main__":
    treinar_modelo()