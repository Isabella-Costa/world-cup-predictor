import os
import joblib
import pandas as pd
import numpy as np
import random
import streamlit as st

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'copa2026_dataset.csv')

@st.cache_resource
def load_models():
    try:
        modelo = joblib.load(os.path.join(MODELS_DIR, 'random_forest_copa.pkl'))
        scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler_copa.pkl'))
        imputer = joblib.load(os.path.join(MODELS_DIR, 'imputer_copa.pkl'))
        features = joblib.load(os.path.join(MODELS_DIR, 'features_esperadas.pkl'))
        return modelo, scaler, imputer, features
    except FileNotFoundError as e:
        st.error(f"⚠️ Erro ao carregar modelos pkl: {e}")
        return None, None, None, None

modelo, scaler, imputer, features_esperadas = load_models()

def _preparar_vetor_confronto(time_a_row, time_b_row, features_esperadas):
    vetor = {col: (False if "region_name" in col or "confederation" in col else 0) for col in features_esperadas}
    
    vetor['is_team_a_host'] = bool(time_a_row['host'])
    vetor['rank_a'] = float(time_a_row['rank_fifa'])
    vetor['valor_a'] = float(time_a_row['valor_eur_m'] * 1000000)
    vetor['vitorias_4a_a'] = float(time_a_row['vitorias_4a'])
    vetor['gols_elite_a'] = float(time_a_row['gols_elite'])
    vetor['ast_elite_a'] = float(time_a_row['ast_elite'])
    vetor['minutos_elite_a'] = float(time_a_row['min_elite'])
    
    vetor['is_team_b_host'] = bool(time_b_row['host'])
    vetor['rank_b'] = float(time_b_row['rank_fifa'])
    vetor['valor_b'] = float(time_b_row['valor_eur_m'] * 1000000)
    vetor['vitorias_4a_b'] = float(time_b_row['vitorias_4a'])
    vetor['gols_elite_b'] = float(time_b_row['gols_elite'])
    vetor['ast_elite_b'] = float(time_b_row['ast_elite'])
    vetor['minutos_elite_b'] = float(time_b_row['min_elite'])
    
    for col in features_esperadas:
        if "confederation_name_a" in col:
            if time_a_row['is_UEFA'] and "UEFA" in col: vetor[col] = True
            elif time_a_row['is_CONMEBOL'] and "South American" in col: vetor[col] = True
            elif time_a_row['is_CAF'] and "African" in col: vetor[col] = True
            elif time_a_row['is_CONCACAF'] and "North" in col: vetor[col] = True
            elif time_a_row['is_OFC'] and "Oceania" in col: vetor[col] = True
        elif "confederation_name_b" in col:
            if time_b_row['is_UEFA'] and "UEFA" in col: vetor[col] = True
            elif time_b_row['is_CONMEBOL'] and "South American" in col: vetor[col] = True
            elif time_b_row['is_CAF'] and "African" in col: vetor[col] = True
            elif time_b_row['is_CONCACAF'] and "North" in col: vetor[col] = True
            elif time_b_row['is_OFC'] and "Oceania" in col: vetor[col] = True
        elif "region_name_a" in col:
            if time_a_row['is_Europa'] and "Europe" in col: vetor[col] = True
            elif time_a_row['is_AmericaSul'] and "South America" in col: vetor[col] = True
            elif time_a_row['is_Africa'] and "Africa" in col: vetor[col] = True
            elif time_a_row['is_Asia'] and "Asia" in col: vetor[col] = True
        elif "region_name_b" in col:
            if time_b_row['is_Europa'] and "Europe" in col: vetor[col] = True
            elif time_b_row['is_AmericaSul'] and "South America" in col: vetor[col] = True
            elif time_b_row['is_Africa'] and "Africa" in col: vetor[col] = True
            elif time_b_row['is_Asia'] and "Asia" in col: vetor[col] = True

    return pd.DataFrame([vetor])[features_esperadas]

CACHE_PROBABILIDADES = {}

def prever_vencedor_partida(time_a_name, time_b_name, df_selecoes, permitir_empate=True):
    chave_jogo = f"{time_a_name}_vs_{time_b_name}"
    
    if chave_jogo in CACHE_PROBABILIDADES:
        probs = CACHE_PROBABILIDADES[chave_jogo]
    else:
        row_a = df_selecoes[df_selecoes['selecao'] == time_a_name].iloc[0]
        row_b = df_selecoes[df_selecoes['selecao'] == time_b_name].iloc[0]
        
        df_confronto = _preparar_vetor_confronto(row_a, row_b, features_esperadas)
        matriz_imputada = imputer.transform(df_confronto)
        matriz_escalada = scaler.transform(matriz_imputada)
        
        probs = modelo.predict_proba(matriz_escalada)[0]
        CACHE_PROBABILIDADES[chave_jogo] = probs 
    
    resultados_possiveis = [time_b_name, 'Empate', time_a_name]
    escolha = random.choices(resultados_possiveis, weights=probs, k=1)[0]
    
    if escolha == 'Empate' and not permitir_empate:
        return time_a_name if probs[2] >= probs[0] else time_b_name
    return escolha

def executar_monte_carlo(selecao_analise, lista_paises, num_simulacoes):
    df_selecoes = pd.read_csv(DATA_PATH)
    historico_fases = {pais: {"oitavas": 0, "quartas": 0, "semifinal": 0, "final": 0, "campeao": 0} for pais in lista_paises}
    
    for _ in range(num_simulacoes):
        paises_embaralhados = list(lista_paises)
        random.shuffle(paises_embaralhados)
        
        avancam_mata_mata = []
        
        for i in range(0, len(paises_embaralhados), 4):
            grupo = paises_embaralhados[i:i+4]
            if len(grupo) < 2: continue
            
            pontuacao = {p: 0 for p in grupo}
            for g1 in range(len(grupo)):
                for g2 in range(g1 + 1, len(grupo)):
                    res = prever_vencedor_partida(grupo[g1], grupo[g2], df_selecoes, permitir_empate=True)
                    if res == 'Empate':
                        pontuacao[grupo[g1]] += 1
                        pontuacao[grupo[g2]] += 1
                    else:
                        pontuacao[res] += 3
            
            classificados = sorted(pontuacao, key=pontuacao.get, reverse=True)[:2]
            avancam_mata_mata.extend(classificados)
        
        if len(avancam_mata_mata) > 32:
            avancam_mata_mata = avancam_mata_mata[:32]
        elif len(avancam_mata_mata) < 32:
            resto = list(set(lista_paises) - set(avancam_mata_mata))
            avancam_mata_mata.extend(resto[:(32 - len(avancam_mata_mata))])
            
        # --- FASE DOS 32 (Eliminatória direta até as Oitavas) ---
        vencedores_r32 = []
        for i in range(0, len(avancam_mata_mata), 2):
            vencedor = prever_vencedor_partida(avancam_mata_mata[i], avancam_mata_mata[i+1], df_selecoes, permitir_empate=False)
            vencedores_r32.append(vencedor)
            
        # --- OITAVAS DE FINAL (Top 16) ---
        vencedores_oitavas = []
        for pais in vencedores_r32:
            historico_fases[pais]["oitavas"] += 1
            
        for i in range(0, len(vencedores_r32), 2):
            if i+1 < len(vencedores_r32):
                vencedor = prever_vencedor_partida(vencedores_r32[i], vencedores_r32[i+1], df_selecoes, permitir_empate=False)
                vencedores_oitavas.append(vencedor)
                
        # --- QUARTAS DE FINAL (Top 8) ---
        vencedores_quartas = []
        for pais in vencedores_oitavas:
            historico_fases[pais]["quartas"] += 1
            
        for i in range(0, len(vencedores_oitavas), 2):
            if i+1 < len(vencedores_oitavas):
                vencedor = prever_vencedor_partida(vencedores_oitavas[i], vencedores_oitavas[i+1], df_selecoes, permitir_empate=False)
                vencedores_quartas.append(vencedor)
                
        # --- SEMIFINAL (Top 4) ---
        vencedores_semi = []
        for pais in vencedores_quartas:
            historico_fases[pais]["semifinal"] += 1
            
        for i in range(0, len(vencedores_quartas), 2):
            if i+1 < len(vencedores_quartas):
                vencedor = prever_vencedor_partida(vencedores_quartas[i], vencedores_quartas[i+1], df_selecoes, permitir_empate=False)
                vencedores_semi.append(vencedor)
                
        # --- FINAL ---
        if len(vencedores_semi) >= 2:
            historico_fases[vencedores_semi[0]]["final"] += 1
            historico_fases[vencedores_semi[1]]["final"] += 1
            
            campeao = prever_vencedor_partida(vencedores_semi[0], vencedores_semi[1], df_selecoes, permitir_empate=False)
            historico_fases[campeao]["campeao"] += 1

    resultados_proporcional = {}
    for pais, fases in historico_fases.items():
        resultados_proporcional[pais] = {fase: (fator / num_simulacoes) * 100 for fase, fator in fases.items()}
        
    st.session_state['matrix_monte_carlo'] = resultados_proporcional
    
    return resultados_proporcional[selecao_analise]

def obter_ranking_etapa(etapa_alvo, lista_paises):
    mapa_etapas = {
        "Oitavas de Final": "oitavas",
        "Quartas de Final": "quartas",
        "Semifinal": "semifinal",
        "Final": "final",
        "Campeão": "campeao"
    }
    
    chave_interna = mapa_etapas.get(etapa_alvo, "campeao")

    if 'matrix_monte_carlo' not in st.session_state:
        return pd.DataFrame({"Seleção": lista_paises, "Chance": [0.0] * len(lista_paises)}).sort_values(by="Seleção")
        
    dados_simulados = st.session_state['matrix_monte_carlo']
    
    dados_ranking = []
    for pais in lista_paises:
        chance_pais = dados_simulados.get(pais, {}).get(chave_interna, 0.0)
        dados_ranking.append({"Seleção": pais, "Chance": chance_pais})
        
    df_ranking = pd.DataFrame(dados_ranking)
    return df_ranking.sort_values(by="Chance", ascending=False)

def obter_probabilidades_confronto(time_a_name, time_b_name, df_selecoes):
    chave_jogo = f"{time_a_name}_vs_{time_b_name}"
    
    if chave_jogo in CACHE_PROBABILIDADES:
        probs = CACHE_PROBABILIDADES[chave_jogo]
    else:
        row_a = df_selecoes[df_selecoes['selecao'] == time_a_name].iloc[0]
        row_b = df_selecoes[df_selecoes['selecao'] == time_b_name].iloc[0]
        
        df_confronto = _preparar_vetor_confronto(row_a, row_b, features_esperadas)
        matriz_imputada = imputer.transform(df_confronto)
        matriz_escalada = scaler.transform(matriz_imputada)
        
        probs = modelo.predict_proba(matriz_escalada)[0]
        CACHE_PROBABILIDADES[chave_jogo] = probs 
        
    return {
        "vitoria_a": probs[2] * 100,
        "empate": probs[1] * 100,
        "vitoria_b": probs[0] * 100
    }