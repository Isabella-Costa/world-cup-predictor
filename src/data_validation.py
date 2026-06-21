import pandas as pd
import numpy as np
import os

print("A iniciar a análise de integridade dos dados...")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

try:
    df = pd.read_csv(os.path.join(PROCESSED_DIR, 'dataset_final.csv'))
    print(f"✅ Dataset carregado: {df.shape[0]} linhas e {df.shape[1]} colunas.\n")
except FileNotFoundError:
    print("❌ Erro: Dataset não encontrado. Executa o script de data_prep primeiro.")
    exit()


print("-" * 50)
print("FASE 1: DADOS EM FALTA")
print("-" * 50)
missing_data = df.isnull().sum()
missing_cols = missing_data[missing_data > 0]

if missing_cols.empty:
    print("Perfeito! Não há dados em falta no dataset.")
else:
    print("Atenção! As seguintes colunas têm valores nulos (NaN):")
    for col, count in missing_cols.items():
        percentagem = (count / len(df)) * 100
        print(f" - {col}: {count} valores nulos ({percentagem:.2f}%)")

print("\n" + "-" * 50)
print("FASE 2: LINHAS DUPLICADAS")
print("-" * 50)
duplicados = df.duplicated().sum()
if duplicados == 0:
    print("Excelente! Não há jogos duplicados na base (sem vazamento de redundância).")
else:
    print(f"⚠️ Aviso Crítico: Foram detetadas {duplicados} linhas duplicadas. Isto vicia o modelo.")

print("\n" + "-" * 50)
print("FASE 3: ANOMALIAS LÓGICAS")
print("-" * 50)
anomalias = 0

# A. Rankings não podem ser menores que 1
if 'rank_a' in df.columns and (df['rank_a'] < 1).any():
    print("⚠️ Anomalia: Encontrado ranking da FIFA inferior a 1 para a Equipa A.")
    anomalias += 1
if 'rank_b' in df.columns and (df['rank_b'] < 1).any():
    print("⚠️ Anomalia: Encontrado ranking da FIFA inferior a 1 para a Equipa B.")
    anomalias += 1

# B. Valores de mercado não podem ser negativos
if 'valor_a' in df.columns and (df['valor_a'] < 0).any():
    print("⚠️ Anomalia: Valores de mercado negativos detetados na Equipa A.")
    anomalias += 1
if 'valor_b' in df.columns and (df['valor_b'] < 0).any():
    print("⚠️ Anomalia: Valores de mercado negativos detetados na Equipa B.")
    anomalias += 1

cols_stats = ['gols_elite_a', 'ast_elite_a', 'minutos_elite_a', 'gols_elite_b', 'ast_elite_b', 'minutos_elite_b']
for col in cols_stats:
    if col in df.columns and (df[col] < 0).any():
        print(f"⚠️ Anomalia: Valores negativos detetados na feature de volume {col}.")
        anomalias += 1

if anomalias == 0:
    print("Tudo certo! Nenhuma violação lógica (números negativos ou estados impossíveis) foi detetada nos tensores de entrada.")

print("\nAuditoria concluída.")