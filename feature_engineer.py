import numpy as np
import pandas as pd
from tqdm import tqdm

formations_dict = {'4-3-1-2': ['GK', 'RB|RWB', 'LCB|CB', 'RCB|CB', 'LB|LWB', 'CDM|CM', 'CDM|CM', 'CDM|CM', 'CAM|CF', 'CF|ST', 'CF|ST'],
                   '4-3-2-1': ['GK', 'RB|RWB', 'LCB|CB', 'RCB|CB', 'LB|LWB', 'CDM|CM', 'CDM|CM', 'CDM|CM', 'CAM|CF', 'CAM|CF', 'CF|ST'],
                   '4-3-3': ['GK', 'RB|RWB', 'LCB|CB', 'RCB|CB', 'LB|LWB', 'CDM|CM', 'CDM|CM', 'CDM|CM', 'RW|RF|ST', 'CF|ST', 'LW|LF|ST'],
                   '4-4-2': ['GK', 'RB|RWB', 'RCB|CB', 'LCB|CB', 'LB|LWB', 'RM|RW', 'CDM|CM', 'CDM|CM', 'LM|LW', 'CF|ST', 'CF|ST'],
                   '4-5-1': ['GK', 'RB|RWB', 'RCB|CB', 'LCB|CB', 'LB|LWB', 'RM|RW', 'CDM|CM', 'CDM|CM', 'LM|LW', 'CF|ST', 'CF|ST'],
                   '3-4-1-2': ['GK', 'RCB|CB', 'CB', 'LCB|CB', 'RM|RW', 'CDM|CM', 'CDM|CM', 'LM|LW', 'CAM|CF', 'CF|ST', 'CF|ST'],
                   '3-4-3': ['GK', 'RCB|CB', 'CB', 'LCB|CB', 'RWB|RM', 'CDM|CM', 'CDM|CM', 'LWB|LM', 'RW|RF|ST', 'CF|ST', 'LW|LF|ST'],
                   '3-5-2': ['GK', 'RCB|CB', 'CB', 'LCB|CB', 'RM|RWB|RB', 'CDM|CM', 'CDM|CM', 'CDM|CM', 'LM|LWB|LB', 'CF|ST', 'CF|ST']}

def load_players_per_team(team, player_list, national):
    """Load all players that could potentially have played."""

    if national == 0:
        team_player_list = player_list.loc[player_list['club_name'] == team]
    else:
        team_player_list = player_list.loc[(player_list['nationality_name'] == team)]

    return team_player_list

def obtain_best_formation(players, measurement='overall'):
    """Find the formation for which we have the best players."""

    # loop over the formations and check what we have players for
    formations_total_vals = {}
    for formation in formations_dict:
        copied_df = players.copy()
        pos_list = formations_dict[formation]
        total_vals = []
        for pos in pos_list:
            # get best record based on 'overall' or 'potential',
            # then drop that record from copied df, so that it cannot be selected again
            if not np.isnan(copied_df[copied_df['player_positions'].str.contains(pos)][measurement].max()):
                total_vals.append(copied_df[copied_df['player_positions'].str.contains(pos)][measurement].max())
                copied_df.drop(copied_df[copied_df['player_positions'].str.contains(pos)][measurement].idxmax(),
                               inplace=True)
        if len(total_vals) == 11:
            formations_total_vals[formation] = sum(total_vals)
        else:
            # some formations might not find 11 available players -
            # these ones need to be excluded from any possible calcuation
            formations_total_vals[formation] = 0

    # return none if no possible configuration could be found
    if sum(formations_total_vals.values()) == 0:
        return None
    else:
        # return best formation
        best_formation = max(formations_total_vals, key=formations_total_vals.get)
        return best_formation


def get_best_lineup(lineup_df, formation='', measurement=''):
    """Select the best players by our selected formation."""
    copy_df = lineup_df.copy()

    # if formation is not chosen, then the best one is calculated with a formula
    if formation == '':
        formation = obtain_best_formation(copy_df, measurement)
    squad_lineup = formations_dict[formation]

    # select the best player for the position
    composed_squad = pd.DataFrame()
    for pos in squad_lineup:
        best_player_record = copy_df.loc[[copy_df[copy_df['player_positions'].str.contains(pos)][measurement].idxmax()]]
        composed_squad = pd.concat([composed_squad, best_player_record])
        copy_df.drop(copy_df[copy_df['player_positions'].str.contains(pos)][measurement].idxmax(), inplace=True)
    return formation, composed_squad


def compose_best_squad(team):
    """Compose the best possible team based on the available players."""
    all_potential_players_in_team = load_players_per_team(team, player_22_df, row['national_game'])

    if len(all_potential_players_in_team) >= 11:
        best_formation = obtain_best_formation(all_potential_players_in_team)

        if best_formation is not None:
            formation, composed_team = get_best_lineup(all_potential_players_in_team,
                                            formation=best_formation, measurement='overall')

            assert len(composed_team) == 11, f'Composition {team} not correct, ' \
                                                f'{len(composed_team[1])} players'

            return composed_team
        else:
            return None
    else:
        return None

def change_club_name_to_match_fifa(club):
    parsed_name = club
    if club == 'China':
        parsed_name = 'China PR'
    if club == 'DR Congo':
        parsed_name = 'Congo DR'
    if club == 'USA':
        parsed_name = 'United States'
    if club == 'Guinea-Bissau':
        parsed_name = 'Guinea Bissau'
    return parsed_name

if __name__ == '__main__':
    fixture_overview_df = pd.read_csv('prepped_data_sources/prepped_fixture_overview.csv')
    player_22_df = pd.read_csv('fifa_player_data/players_22.csv')

    # player_22_df.iloc[:, 0:10].info()
    # player_22_df.iloc[:, 10:20].info()
    # player_22_df.iloc[:, 20:30].info()
    # player_22_df.iloc[:, 30:40].info()
    # player_22_df.iloc[:, 40:50].info()
    # player_22_df.iloc[:, 50:60].info()
    # player_22_df.iloc[:, 60:70].info()
    # player_22_df.iloc[:, 70:80].info()
    # player_22_df.iloc[:, 80:90].info()
    # player_22_df.iloc[:, 90:100].info()
    #
    # assert False

    # initiate feature columns
    fixture_overview_df['total_home_team_price'] = 0
    fixture_overview_df['total_away_team_price'] = 0
    fixture_overview_df['total_home_team_potential'] = 0
    fixture_overview_df['total_away_team_potential'] = 0

    composed_teams = {}
    uncomposed_teams = []
    for idx, row in tqdm(fixture_overview_df.iterrows()):
        hometeam = row['HomeTeam']
        awayteam = row['AwayTeam']

        # change names to match fifa club names
        hometeam = change_club_name_to_match_fifa(hometeam)
        awayteam = change_club_name_to_match_fifa(awayteam)

        # obtain best home and away teams
        if hometeam not in composed_teams.keys():
            best_home_team = compose_best_squad(hometeam)
            if best_home_team is not None:
                composed_teams[hometeam] = best_home_team
            else:
                uncomposed_teams.append(hometeam)
        else:
            best_home_team = composed_teams[hometeam]

        if awayteam not in composed_teams.keys():
            best_away_team = compose_best_squad(awayteam)
            if best_away_team is not None:
                composed_teams[awayteam] = best_away_team
            else:
                uncomposed_teams.append(awayteam)
        else:
            best_away_team = composed_teams[awayteam]

        if best_home_team is not None and best_away_team is not None:
            fixture_overview_df.loc[idx, 'total_home_team_price'] = best_home_team['value_eur'].sum()
            fixture_overview_df.loc[idx, 'total_away_team_price'] = best_away_team['value_eur'].sum()

            fixture_overview_df.loc[idx, 'total_home_team_potential'] = best_home_team['potential'].sum()
            fixture_overview_df.loc[idx, 'total_away_team_potential'] = best_away_team['potential'].sum()

            fixture_overview_df.loc[idx, 'total_home_team_overall'] = best_home_team['overall'].sum()
            fixture_overview_df.loc[idx, 'total_away_team_overall'] = best_away_team['overall'].sum()

            fixture_overview_df.loc[idx, 'total_home_team_work_rate'] = best_home_team['work_rate'].sum()
            fixture_overview_df.loc[idx, 'total_away_team_work_rate'] = best_away_team['work_rate'].sum()

            fixture_overview_df.loc[idx, 'total_home_team_international_reputation'] = best_home_team['international_reputation'].sum()
            fixture_overview_df.loc[idx, 'total_away_team_international_reputation'] = best_away_team['international_reputation'].sum()

            fixture_overview_df.loc[idx, 'total_home_team_age'] = best_home_team['age'].sum()
            fixture_overview_df.loc[idx, 'total_away_team_age'] = best_away_team['age'].sum()

            fixture_overview_df.loc[idx, 'total_home_team_height_cm'] = best_home_team['height_cm'].sum()
            fixture_overview_df.loc[idx, 'total_away_team_height_cm'] = best_away_team['height_cm'].sum()

            fixture_overview_df.loc[idx, 'total_home_team_weight_kg'] = best_home_team['weight_kg'].sum()
            fixture_overview_df.loc[idx, 'total_away_team_weight_kg'] = best_away_team['weight_kg'].sum()

    print(f'Team found for {len(composed_teams) / len(fixture_overview_df["HomeTeam"].unique())*100}% of the teams')
    print(f'Uncomposed teams: {set(uncomposed_teams)}')

    fixture_overview_df.dropna(inplace=True)

    fixture_overview_df.to_csv('prepped_data_set.csv')



