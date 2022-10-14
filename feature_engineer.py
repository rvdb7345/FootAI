import pandas as pd

def load_players_per_team(team, player_list, national):
    """Load all players that could potentially have played."""

    if national == 0:
        team_player_list = player_list.loc[player_list['club_name'] == team]
    else:
        team_player_list = player_list.loc[(player_list['nationality_name'] == team)]

    return team_player_list

def generate_fictive_team_from_players(players):
    """We don't know how the teams were composed at the time of playing, therefore, we make a fictive team out of the
    potential players we have."""

    return


if __name__ == '__main__':
    fixture_overview_df = pd.read_csv('prepped_data_sources/prepped_fixture_overview.csv')
    player_22_df = pd.read_csv('fifa_player_data/players_22.csv')

    for idx, row in fixture_overview_df.iterrows():
        hometeam = row['HomeTeam']
        awayteam = row['AwayTeam']

        all_potential_players_in_home_team = load_players_per_team(hometeam, player_22_df, row['national_game'])
        all_potential_players_in_away_team = load_players_per_team(awayteam, player_22_df, row['national_game'])




