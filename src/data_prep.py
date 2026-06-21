import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

matches = pd.read_csv(os.path.join(RAW_DIR, 'matches_1930_2022.csv'))
teams = pd.read_csv(os.path.join(RAW_DIR, 'world_fifa_worldcup_teams.csv'))
rankings_stats = pd.read_csv(os.path.join(RAW_DIR, 'train.csv')) 
players_25_26 = pd.read_csv(os.path.join(RAW_DIR, 'players_data-2025_2026.csv'))
players_history = pd.read_csv(os.path.join(RAW_DIR, 'world_fifa_worldcup_players.csv'))
awards = pd.read_csv(os.path.join(RAW_DIR, 'world_fifa_worldcup_award_winners.csv')) 

matches.rename(columns={
    'Year': 'year',
    'home_team': 'team_a_name',
    'away_team': 'team_b_name',
    'home_score': 'team_a_score',
    'away_score': 'team_b_score',
    'Host': 'country_name'
}, inplace=True)

matches = matches[matches['year'] >= 2002].copy()

conditions = [
    (matches['team_a_score'] > matches['team_b_score']),
    (matches['team_a_score'] == matches['team_b_score']),
    (matches['team_a_score'] < matches['team_b_score'])
]
matches['resultado'] = np.select(conditions, [2, 1, 0], default=np.nan)

matches['is_team_a_host'] = (matches['team_a_name'] == matches['country_name']).astype(int)
matches['is_team_b_host'] = (matches['team_b_name'] == matches['country_name']).astype(int)

df_unified = matches.copy()
df_unified.drop(columns=['country_name'], inplace=True)

df_unified = df_unified.merge(
    teams[['team_name', 'team_code', 'region_name', 'confederation_name']], 
    left_on='team_a_name', right_on='team_name', how='left'
).drop(columns=['team_name'])
df_unified.rename(columns={'region_name': 'region_name_a', 'confederation_name': 'confederation_name_a', 'team_code': 'team_code_x'}, inplace=True)

df_unified = df_unified.merge(
    teams[['team_name', 'team_code', 'region_name', 'confederation_name']], 
    left_on='team_b_name', right_on='team_name', how='left'
).drop(columns=['team_name'])
df_unified.rename(columns={'region_name': 'region_name_b', 'confederation_name': 'confederation_name_b', 'team_code': 'team_code_y'}, inplace=True)

rank_cols = ['version', 'team', 'fifa_rank_pre_tournament', 'squad_total_market_value_eur', 'wins_last_4y']

df_unified = df_unified.merge(
    rankings_stats[rank_cols], 
    left_on=['year', 'team_a_name'], right_on=['version', 'team'], how='left'
).drop(columns=['version', 'team'])
df_unified.rename(columns={
    'fifa_rank_pre_tournament': 'rank_a', 
    'squad_total_market_value_eur': 'valor_a',
    'wins_last_4y': 'vitorias_4a_a'
}, inplace=True)

df_unified = df_unified.merge(
    rankings_stats[rank_cols], 
    left_on=['year', 'team_b_name'], right_on=['version', 'team'], how='left'
).drop(columns=['version', 'team'])
df_unified.rename(columns={
    'fifa_rank_pre_tournament': 'rank_b', 
    'squad_total_market_value_eur': 'valor_b',
    'wins_last_4y': 'vitorias_4a_b'
}, inplace=True)

players_25_26['Nation_Code'] = players_25_26['Nation'].dropna().apply(lambda x: str(x).split(' ')[-1])
squad_stats = players_25_26[['Nation_Code', 'Gls', 'Ast', 'Min']].groupby('Nation_Code').sum().reset_index()

df_unified = df_unified.merge(squad_stats, left_on='team_code_x', right_on='Nation_Code', how='left').drop(columns=['Nation_Code'])
df_unified.rename(columns={'Gls': 'gols_elite_a', 'Ast': 'ast_elite_a', 'Min': 'minutos_elite_a'}, inplace=True)

df_unified = df_unified.merge(squad_stats, left_on='team_code_y', right_on='Nation_Code', how='left').drop(columns=['Nation_Code'])
df_unified.rename(columns={'Gls': 'gols_elite_b', 'Ast': 'ast_elite_b', 'Min': 'minutos_elite_b'}, inplace=True)

exp_jogador = players_history[['player_id', 'count_tournaments']].copy()
premios_por_jogador = awards.groupby('player_id')['award_id'].count().reset_index()
premios_por_jogador.rename(columns={'award_id': 'total_premios'}, inplace=True)

perfil_jogador = exp_jogador.merge(premios_por_jogador, on='player_id', how='left').fillna(0)
jogador_time = awards[['player_id', 'tournament_id']].drop_duplicates(subset=['player_id'])

output_path = os.path.join(PROCESSED_DIR, 'dataset_final.csv')
df_unified.to_csv(output_path, index=False)
print(f"Pipeline concluído. Dataset salvo em: {output_path}")