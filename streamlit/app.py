import streamlit as st
import pandas as pd
import backend_app as bk
import os

st.set_page_config(
    page_title="World Cup Predictor AI 2026",
    page_icon="⚽",
    layout="wide"
)

# Dados das Seleções 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'copa2026_dataset.csv')

@st.cache_data
def carregar_dados_copa():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    else:
        st.error(f"Não foi possível encontrar o arquivo em: {DATA_PATH}")
        return pd.DataFrame()

df_selecoes = carregar_dados_copa()

st.title("⚽ World Cup Predictor & Simulator - IA")
st.divider()

if df_selecoes.empty:
    st.warning("Por favor, garanta que o arquivo 'copa2026_dataset.csv' está na pasta 'data/'.")
    st.stop()

with st.sidebar:
    st.header("Configuração")
    st.write("### Seleções Participantes")
    
    selecionar_todas = st.checkbox("Selecionar Todas (48 Seleções)", value=True)
    
    lista_paises = sorted(df_selecoes['selecao'].unique())
    paises_participantes = []
    
    with st.expander("Ver lista completa", expanded=True):
        for pais in lista_paises:
            status = st.checkbox(pais, value=selecionar_todas, key=f"chk_{pais}")
            if status:
                paises_participantes.append(pais)
                
    st.markdown(f"Selecionadas: {len(paises_participantes)} de {len(lista_paises)}")
    st.divider()

if len(paises_participantes) == 0:
    st.warning("Selecione pelo menos 1 país na barra lateral para começar.")
    st.stop()

tab_confronto, tab_torneio = st.tabs(["Confronto Direto", "Simulador do Torneio"])

with tab_confronto:
    col_A, col_B = st.columns(2)
    
    with col_A:
        time_a = st.selectbox("Seleção A (Mandante):", options=paises_participantes, index=paises_participantes.index("Brasil") if "Brasil" in paises_participantes else 0)
        
    with col_B:
        idx_b = paises_participantes.index("França") if "França" in paises_participantes else (1 if len(paises_participantes) > 1 else 0)
        time_b = st.selectbox("Seleção B (Visitante):", options=paises_participantes, index=idx_b)

    st.write("") 
    
    if st.button("Analisar Confronto", type="primary", width="stretch"):
        if time_a == time_b:
            st.warning("Por favor, escolha duas seleções diferentes.")
        else:
            with st.spinner("A rede neural está a analisar o confronto..."):
                probs = bk.obter_probabilidades_confronto(time_a, time_b, df_selecoes)
                
                st.markdown("---")
                st.subheader("Saída da Rede Neural:")
                
                st.write(f"Vitória {time_a}")
                st.progress(probs['vitoria_a'] / 100.0)
                st.caption(f"{probs['vitoria_a']:.1f}%")
                
                st.write(f"Empate")
                st.progress(probs['empate'] / 100.0)
                st.caption(f"{probs['empate']:.1f}%")
                
                st.write(f"Vitória {time_b}")
                st.progress(probs['vitoria_b'] / 100.0)
                st.caption(f"{probs['vitoria_b']:.1f}%")

with tab_torneio:
    col_UI, col_ranking = st.columns([1.2, 1])

    with col_UI:
        st.header("Progresso no Torneio")
        selecao_analise = st.selectbox(
            "Escolha uma seleção para analisar a campanha:",
            options=paises_participantes,
            index=paises_participantes.index("Brasil") if "Brasil" in paises_participantes else 0
        )
            
        num_simulacoes = st.slider("Número de Simulações de Monte Carlo:", min_value=1000, max_value=10000, value=1000, step=1000)
        
        if st.button("Calcular Probabilidades da Copa", width="stretch"):
            with st.spinner(f"A rodar {num_simulacoes} Copas do Mundo via Monte Carlo..."):
                bk.executar_monte_carlo(selecao_analise, paises_participantes, num_simulacoes)
                st.session_state['simulacao_concluida'] = True

        st.markdown("---")

        if st.session_state.get('simulacao_concluida', False):
            chances = st.session_state['matrix_monte_carlo'].get(selecao_analise, {})
            
            st.subheader(f"Chances: {selecao_analise}")
            etapas = [
                ("Oitavas de Final", chances.get('oitavas', 0)),
                ("Quartas de Final", chances.get('quartas', 0)),
                ("Semifinal", chances.get('semifinal', 0)),
                ("Final", chances.get('final', 0)),
                ("Campeão", chances.get('campeao', 0))
            ]
            
            for nome_etapa, valor in etapas:
                st.write(f"{nome_etapa:<20}")
                st.progress(min(max(valor / 100.0, 0.0), 1.0))
                st.caption(f"{valor:.1f}%")

    with col_ranking:
        st.header("🏆 Ranking Geral")
        st.write("Filtre as seleções com mais chance de chegar em cada fase:")
        
        etapa_alvo = st.selectbox(
            "Etapa desejada:",
            ["Oitavas de Final", "Quartas de Final", "Semifinal", "Final", "Campeão"]
        )
        
        if st.session_state.get('simulacao_concluida', False):
            df_ranking_etapa = bk.obter_ranking_etapa(etapa_alvo, paises_participantes)
            st.success(f"Top Seleções: {etapa_alvo}")
            
            st.dataframe(
                df_ranking_etapa,
                column_config={
                    "Seleção": st.column_config.TextColumn("Seleção"),
                    "Chance": st.column_config.ProgressColumn("Chance", format="%.1f%%", min_value=0, max_value=100)
                },
                width="stretch",
                hide_index=True
            )
        else:
            st.info("👈 Rode a simulação ao lado primeiro.")