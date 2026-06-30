import os
import pandas as pd
# pyrefly: ignore [missing-import]
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

def treinar_modelo_mlp():
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

    # balanceamento
    print("\nAplicando técnica de acréscimo de objetos à classe minoritária...")
    
    df_temp = pd.concat([X, y], axis=1)
    
  
    df_maioria = df_temp[df_temp[coluna_alvo] == 0]
    df_minoria = df_temp[df_temp[coluna_alvo] == 1]
    
    df_minoria_aumentada = df_minoria.sample(n=len(df_maioria), replace=True, random_state=42)
    
    df_balanceado = pd.concat([df_maioria, df_minoria_aumentada])
    X = df_balanceado.drop(columns=[coluna_alvo])
    y = df_balanceado[coluna_alvo]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()), 
        ('mlp', MLPClassifier(max_iter=500, random_state=42))
    ])

    param_grid = [
        {
            'mlp__hidden_layer_sizes': [(50,), (100,), (50, 25)],
            'mlp__activation': ['logistic', 'relu'],
            'mlp__solver': ['sgd'],
            'mlp__learning_rate_init': [0.001, 0.01],
            'mlp__momentum': [0.9, 0.95],
        },
        {
            'mlp__hidden_layer_sizes': [(50,), (100,), (50, 25)],
            'mlp__activation': ['logistic', 'relu'],
            'mlp__solver': ['adam'],
            'mlp__learning_rate_init': [0.001, 0.01],
            # Adam já adapta a taxa de aprendizado para cada parâmetro e possui mecanismo de momento embutido
        }
    ]

    print("\nIniciando busca exaustiva com GridSearchCV...")
    print("Isso pode levar alguns minutos dependendo da complexidade das redes testadas.")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=3, 
        scoring='accuracy',
        n_jobs=-1, 
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    melhor_modelo = grid_search.best_estimator_
    print(f"\nMelhor arquitetura/parâmetros encontrados: {grid_search.best_params_}")
    
    y_pred = melhor_modelo.predict(X_test)
    acuracia = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*40)
    print(f"RESULTADO DO MELHOR MODELO MLP (Acurácia): {acuracia:.2%}")
    print("="*40)
    print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred, zero_division=0))
    
    print("\nGerando gráficos analíticos...")
    pasta_reports = 'reports'
    os.makedirs(pasta_reports, exist_ok=True)

    # Matriz de Confusão
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(melhor_modelo, X_test, y_test, cmap='Blues', ax=ax)
    plt.title("Matriz de Confusão - MLP")
    plt.tight_layout()
    plt.savefig(os.path.join(pasta_reports, "01_mlp_confusion_matrix.png"))
    plt.close()

    # Curva de Perda (Loss Curve)
    mlp_step = melhor_modelo.named_steps['mlp']
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(mlp_step.loss_curve_, label='Training Loss', color='red')
    ax.set_title("Curva de Perda durante o Treinamento")
    ax.set_xlabel("Iterações (Épocas)")
    ax.set_ylabel("Perda (Loss)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(pasta_reports, "02_mlp_loss_curve.png"))
    plt.close()

    # Curva ROC e AUC
    y_prob = melhor_modelo.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Taxa de Falsos Positivos')
    ax.set_ylabel('Taxa de Verdadeiros Positivos')
    ax.set_title('Curva ROC - MLP')
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(pasta_reports, "03_mlp_roc_curve.png"))
    plt.close()

    # Importância das Features (Permutation Importance)
    print("Calculando a importância das features por permutação...")
    resultado_perm = permutation_importance(melhor_modelo, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    
    importancias = pd.Series(resultado_perm.importances_mean, index=X.columns).sort_values(ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    importancias.plot(kind='barh', ax=ax, color='skyblue')
    ax.set_title("Importância das Features (Permutation Importance) - MLP")
    ax.set_xlabel("Queda média na Acurácia")
    plt.tight_layout()
    plt.savefig(os.path.join(pasta_reports, "04_mlp_feature_importance.png"))
    plt.close()
    
    print("Gráficos salvos com sucesso na pasta 'reports/'.")
    
    # Exportação e salvamento do modelo treinado com os melhores hiperparâmetros
    os.makedirs(pasta_modelos, exist_ok=True)
    caminho_salvar = os.path.join(pasta_modelos, 'mlp_model.pkl')
    joblib.dump(melhor_modelo, caminho_salvar)
    print(f"\nSucesso! Melhor modelo salvo em: '{caminho_salvar}'")

if __name__ == "__main__":
    treinar_modelo_mlp()
