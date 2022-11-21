"""This file contains all the code used for engineering features from FIFA player data for a dataset of specified
matches.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

formations_dict = {
    '4-3-1-2': ['GK', 'RB|RWB', 'LCB|CB', 'RCB|CB', 'LB|LWB', 'CDM|CM', 'CDM|CM', 'CDM|CM', 'CAM|CF', 'CF|ST', 'CF|ST'],
    '4-3-2-1': ['GK', 'RB|RWB', 'LCB|CB', 'RCB|CB', 'LB|LWB', 'CDM|CM', 'CDM|CM', 'CDM|CM', 'CAM|CF', 'CAM|CF',
                'CF|ST'],
    '4-3-3': ['GK', 'RB|RWB', 'LCB|CB', 'RCB|CB', 'LB|LWB', 'CDM|CM', 'CDM|CM', 'CDM|CM', 'RW|RF|ST', 'CF|ST',
              'LW|LF|ST'],
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
        best_player_record['chosen_position'] = pos
        composed_squad = pd.concat([composed_squad, best_player_record])
        copy_df.drop(copy_df[copy_df['player_positions'].str.contains(pos)][measurement].idxmax(), inplace=True)
    return formation, composed_squad


def compose_best_squad(team, player_df, row):
    """Compose the best possible team based on the available players."""
    all_potential_players_in_team = load_players_per_team(team, player_df, row['national_game'])

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
    """Some countries are called differently in the FIFA dataset vs the historic fixtures."""
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


def parse_work_rate(work_rate_column):
    """Work rate is given in high, medium or low. We have to parse it to integers."""
    work_rate_column = work_rate_column.map({
        'High/High': 6,
        'High/Medium': 5,
        'High/Low': 4,
        'Medium/High': 5,
        'Medium/Medium': 4,
        'Medium/Low': 3,
        'Low/High': 4,
        'Low/Medium': 3,
        'Low/Low': 2
    })
    return work_rate_column


def load_all_different_player_data():
    """Load all player data."""
    player_data = {
        2022: pd.read_csv('fifa_player_data/players_22.csv'),
        2021: pd.read_csv('fifa_player_data/players_21.csv'),
        2020: pd.read_csv('fifa_player_data/players_20.csv'),
        2019: pd.read_csv('fifa_player_data/players_19.csv'),
        2018: pd.read_csv('fifa_player_data/players_18.csv'),
        2017: pd.read_csv('fifa_player_data/players_17.csv'),
        2016: pd.read_csv('fifa_player_data/players_16.csv'),
        2015: pd.read_csv('fifa_player_data/players_15.csv')
    }
    return player_data


def engineer_features(pred_df=None):
    """Create features for a set of given matches or our entire data set."""
    if pred_df is None:
        fixture_overview_df = pd.read_csv('prepped_data_sources/prepped_fixture_overview.csv')
        int_fixture_overview_df = pd.read_csv('prepped_data_sources/fixture_overview_international.csv')
        fixture_overview_df = fixture_overview_df.append(int_fixture_overview_df)
    else:
        fixture_overview_df = pred_df

    player_data = load_all_different_player_data()
    for key in player_data:
        player_data[key]['work_rate'] = parse_work_rate(player_data[key]['work_rate'])

    # define line definitions and features we want to extract (field is every position excl. goalkeeper)
    line_definitions = {"goal": ['GK'], "def": ['B'], "mid": ['M'], "att": ['CAM', 'CF', 'ST'],
                        "field": ['B', 'M', 'CAM', 'CF', 'ST']}

    features_to_extract = {
        'general': ['value_eur', 'potential', 'overall', 'work_rate', 'international_reputation', 'age', 'height_cm',
                    'weight_kg', 'shooting', 'passing', 'defending', 'physic',
                    'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots',
                    'mentality_interceptions', 'mentality_positioning', 'mentality_vision',
                    'mentality_penalties', 'mentality_composure', 'league_level'],
        'field': ['skill_dribbling', 'skill_curve', 'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control',
                  'mentality_aggression', 'pace', 'dribbling', 'movement_acceleration', 'movement_sprint_speed',
                  'movement_agility', 'movement_reactions', 'movement_balance', 'weak_foot', 'skill_moves'],
        'goal': ['goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking',
                 'goalkeeping_positioning', 'goalkeeping_reflexes', 'goalkeeping_speed'],
        'def': ['defending_marking_awareness', 'defending_standing_tackle', 'defending_sliding_tackle'],
        'att': ['attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy', 'attacking_short_passing',
                'attacking_volleys']
    }

    composed_teams = {2015: {}, 2016: {}, 2017: {}, 2018: {}, 2019: {}, 2020: {}, 2021: {}, 2022: {}}
    uncomposed_teams = {2015: [], 2016: [], 2017: [], 2018: [], 2019: [], 2020: [], 2021: [], 2022: []}
    for idx, row in tqdm(fixture_overview_df.iterrows()):
        hometeam = row['HomeTeam']
        awayteam = row['AwayTeam']
        year = row['Year']

        # change names to match fifa club names
        hometeam = change_club_name_to_match_fifa(hometeam)
        awayteam = change_club_name_to_match_fifa(awayteam)

        # obtain best home and away teams
        if hometeam not in composed_teams[year].keys():
            best_home_team = compose_best_squad(hometeam, player_df=player_data[year], row=row)
            if best_home_team is not None:
                composed_teams[year][hometeam] = best_home_team
            else:
                uncomposed_teams[year].append(hometeam)
        else:
            best_home_team = composed_teams[year][hometeam]

        if awayteam not in composed_teams[year].keys():
            best_away_team = compose_best_squad(awayteam, player_df=player_data[year], row=row)
            if best_away_team is not None:
                composed_teams[year][awayteam] = best_away_team
            else:
                uncomposed_teams[year].append(awayteam)
        else:
            best_away_team = composed_teams[year][awayteam]

        if best_home_team is not None and best_away_team is not None:

            # get features for different lines in the game
            for line_key, line in line_definitions.items():
                home_players_in_line = best_home_team[best_home_team['chosen_position'].str.contains('|'.join(line))]
                away_players_in_line = best_away_team[best_away_team['chosen_position'].str.contains('|'.join(line))]

                # loop over the different teams
                for team, players_in_line in {"home_team": home_players_in_line,
                                              "away_team": away_players_in_line}.items():

                    # loop over the general features
                    for general_feat in features_to_extract['general']:
                        fixture_overview_df.loc[idx, f'{team}_{general_feat}_{line_key}'] = \
                            players_in_line[general_feat].sum() / len(players_in_line)

                    # line specific features
                    if line_key == 'field':
                        for field_feat in features_to_extract['field']:
                            fixture_overview_df.loc[idx, f'{team}_{field_feat}_{line_key}'] = players_in_line[
                                                                                                  field_feat].sum() / len(
                                players_in_line)
                    if line_key == 'goal':
                        for goal_feat in features_to_extract['goal']:
                            fixture_overview_df.loc[idx, f'{team}_{goal_feat}_{line_key}'] = players_in_line[
                                                                                                 goal_feat].sum() / len(
                                players_in_line)
                    if line_key == 'def':
                        for def_feat in features_to_extract['def']:
                            fixture_overview_df.loc[idx, f'{team}_{def_feat}_{line_key}'] = players_in_line[
                                                                                                def_feat].sum() / len(
                                players_in_line)
                    if line_key == 'att':
                        for att_feat in features_to_extract['att']:
                            fixture_overview_df.loc[idx, f'{team}_{att_feat}_{line_key}'] = players_in_line[
                                                                                                att_feat].sum() / len(
                                players_in_line)

                for general_feat in features_to_extract['general']:
                    fixture_overview_df.loc[idx, f'rel_{general_feat}_{line_key}'] = \
                        fixture_overview_df.loc[idx, f'home_team_{general_feat}_{line_key}'] - \
                        fixture_overview_df.loc[idx, f'away_team_{general_feat}_{line_key}']

                if line_key == 'field':
                    for field_feat in features_to_extract['field']:
                        fixture_overview_df.loc[idx, f'rel_{field_feat}_{line_key}'] = \
                            fixture_overview_df.loc[idx, f'home_team_{field_feat}_{line_key}'] - \
                            fixture_overview_df.loc[idx, f'away_team_{field_feat}_{line_key}']
                if line_key == 'goal':
                    for goal_feat in features_to_extract['goal']:
                        fixture_overview_df.loc[idx, f'rel_{goal_feat}_{line_key}'] = \
                            fixture_overview_df.loc[idx, f'home_team_{goal_feat}_{line_key}'] - \
                            fixture_overview_df.loc[idx, f'away_team_{goal_feat}_{line_key}']
                if line_key == 'def':
                    for def_feat in features_to_extract['def']:
                        fixture_overview_df.loc[idx, f'rel_{def_feat}_{line_key}'] = \
                            fixture_overview_df.loc[idx, f'home_team_{def_feat}_{line_key}'] - \
                            fixture_overview_df.loc[idx, f'away_team_{def_feat}_{line_key}']
                if line_key == 'att':
                    for att_feat in features_to_extract['att']:
                        fixture_overview_df.loc[idx, f'rel_{att_feat}_{line_key}'] = \
                            fixture_overview_df.loc[idx, f'home_team_{att_feat}_{line_key}'] - \
                            fixture_overview_df.loc[idx, f'away_team_{att_feat}_{line_key}']

    fixture_overview_df.dropna(inplace=True)

    print(f'Uncomposed teams: {uncomposed_teams}')

    return fixture_overview_df


if __name__ == '__main__':
    fixture_overview_df = engineer_features()

    fixture_overview_df.to_csv('prepped_data_sources/int_prepped_data_set.csv')
