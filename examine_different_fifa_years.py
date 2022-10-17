import pandas as pd

player_22_df = pd.read_csv('fifa_player_data/players_22.csv')
player_21_df = pd.read_csv('fifa_player_data/players_21.csv')
player_20_df = pd.read_csv('fifa_player_data/players_20.csv')
player_19_df = pd.read_csv('fifa_player_data/players_19.csv')
player_18_df = pd.read_csv('fifa_player_data/players_18.csv')
player_17_df = pd.read_csv('fifa_player_data/players_17.csv')
player_16_df = pd.read_csv('fifa_player_data/players_16.csv')
player_15_df = pd.read_csv('fifa_player_data/players_15.csv')

player_dfs = [
    player_22_df,
    player_21_df,
    player_20_df,
    player_19_df,
    player_18_df,
    player_17_df,
    player_16_df,
    player_15_df
]

keys_in_22 = player_22_df.columns

for player_df in player_dfs:
    print(f'Diff in columns {set(player_16_df.columns) - set(player_df.columns)}')
