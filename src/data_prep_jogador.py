import pandas as pd

df_players = pd.read_csv('src/players_data-2025_2026.csv')
df_players['Nation_Code'] = df_players['Nation'].dropna().apply(lambda x: str(x).split(' ')[-1])

mapa_selecoes = {
    'United States': 'USA', 'Canada': 'CAN', 'Mexico': 'MEX', 'Costa Rica': 'CRC',
    'Panama': 'PAN', 'Jamaica': 'JAM', 'Honduras': 'HON', 'Argentina': 'ARG',
    'Brazil': 'BRA', 'Uruguay': 'URU', 'Colombia': 'COL', 'Ecuador': 'ECU',
    'Venezuela': 'VEN', 'Peru': 'PER', 'France': 'FRA', 'Spain': 'ESP',
    'England': 'ENG', 'Portugal': 'POR', 'Netherlands': 'NED', 'Belgium': 'BEL',
    'Italy': 'ITA', 'Germany': 'GER', 'Croatia': 'CRO', 'Switzerland': 'SUI',
    'Denmark': 'DEN', 'Serbia': 'SRB', 'Poland': 'POL', 'Sweden': 'SWE',
    'Ukraine': 'UKR', 'Austria': 'AUT', 'Morocco': 'MAR', 'Senegal': 'SEN',
    'Egypt': 'EGY', 'Ivory Coast': 'CIV', 'Nigeria': 'NGA', 'Algeria': 'ALG',
    'Cameroon': 'CMR', 'Mali': 'MLI', 'Tunisia': 'TUN', 'Japan': 'JPN',
    'South Korea': 'KOR', 'Iran': 'IRN', 'Australia': 'AUS', 'Saudi Arabia': 'KSA',
    'Qatar': 'QAT', 'Uzbekistan': 'UZB', 'United Arab Emirates': 'UAE', 'New Zealand': 'NZL'
}

df_players['Nota_Convocacao'] = df_players['Min'].fillna(0) + (df_players['G+A'].fillna(0) * 90)
lista_convocados = []

print("Gerando convocações oficiais baseadas em desempenho...")

for selecao, sigla in mapa_selecoes.items():
    jogadores_pais = df_players[df_players['Nation_Code'] == sigla]
    top_26 = jogadores_pais.sort_values(by='Nota_Convocacao', ascending=False).head(26)
    for _, row in top_26.iterrows():
        lista_convocados.append({
            'Selecao': selecao,
            'Jogador': row['Player'],
            'Posicao': row['Pos'],
            'Clube': row['Squad']
        })

df_convocados_finais = pd.DataFrame(lista_convocados)
df_convocados_finais.to_csv('convocados_48_selecoes.csv', index=False)

print(f"\nSUCESSO! Arquivo 'convocados_48_selecoes.csv' gerado.")
print(f"Total de jogadores convocados: {len(df_convocados_finais)}")