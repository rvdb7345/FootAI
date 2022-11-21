"""This file contains to code to map the team names in the fixtures to the fifa clubnames in order to make feature
engineering easier."""

import numpy as np
import pandas as pd
from difflib import SequenceMatcher

def similar(a, b):
    """Calculate fuzzy matching score."""
    return SequenceMatcher(None, a, b).ratio()


def create_fixture_to_fifa_club_mapping(fixture_df, players_22_df, all_country_names):
    """Use fuzzy matching to map team names of the fixtures to those in the fifa club overview."""

    unique_clubs_fifa_players = players_22_df['club_name'].unique().astype(str)
    unique_clubs_fixtures_overview = fixture_df['HomeTeam'].unique().astype(str)

    non_matched_club_names = [club_name for club_name in unique_clubs_fixtures_overview
                              if club_name not in unique_clubs_fifa_players]

    print(
        f'Percentage of non_matched_club_names {len(non_matched_club_names) / len(unique_clubs_fixtures_overview) * 100} %')

    # some preset mapping
    club_fixture_to_fifa_mapping = {"Man United": "Manchester United", "Man. City": "Manchester City",
                                    'Man. United': "Manchester United", 'Man City': "Manchester City",
                                    'Man Utd': "Manchester United", 'Bayern Munich': 'FC Bayern München',
                                    'UD Almería': 'Unión Deportiva Almería', 'Levante UD': 'Levante Unión Deportiva',
                                    'Toulouse FC': 'Toulouse Football Club', 'Kasimpasa': 'Kasimpaşa SK',
                                    'Besiktas': 'Beşiktaş JK', 'Wolfsberg': 'VfL Wolfsburg',
                                    'Istanbul Basaksehir': 'İstanbul Başakşehir FK',
                                    'Bayern Munich': 'FC Bayern München',
                                    'Spartak Moscow': 'Spartak Moskva', 'Alanyaspor': 'Antalyaspor',
                                    'Kasimpasa': 'Kasimpaşa SK', 'D. Alavés': 'Deportivo Alavés',
                                    'SM Caen': 'Stade Malherbe Caen', 'EA Guingamp': 'En Avant de Guingamp',
                                    'Inter Milan': 'Inter Miami CF',
                                    'Fortuna Düsseldorf 1895 e.V.': 'Fortuna Düsseldorf',
                                    'FC Famalicão': 'Futebol Clube de Famalicão',
                                    'FC P.Ferreira': 'FC Paços de Ferreira'}
    list_of_countries_in_fixtures = []

    # go over all unmatches club names in the fixtures
    for unmatched_club_name in non_matched_club_names:

        # check if the club name was a country
        if unmatched_club_name in all_country_names:
                list_of_countries_in_fixtures.append(unmatched_club_name)
        elif unmatched_club_name in club_fixture_to_fifa_mapping.keys():
            continue
        else:
            # look for correct club name using fuzzy matching
            word_match_score = \
                [(fifa_club, similar(unmatched_club_name, fifa_club)) for fifa_club
                 in unique_clubs_fifa_players]

            # select the highest matched word
            word_match_score = np.array(word_match_score)
            sorted_match_score = word_match_score[word_match_score[:, 1].argsort()]

            if float(sorted_match_score[-1][1]) > 0.8 or unmatched_club_name in sorted_match_score[-1][0]:
                club_fixture_to_fifa_mapping[str(unmatched_club_name)] = sorted_match_score[-1][0]
            else:
                print(
                    f"Fuzzy search unsure about match to following teams: "
                    f"{unmatched_club_name, sorted_match_score[-1]}")

    return club_fixture_to_fifa_mapping, list_of_countries_in_fixtures

if __name__ == '__main__':

    # load all necessary data
    all_country_names_df = pd.read_csv('data/all_countries.csv')
    all_country_names = all_country_names_df['Name'].to_list()

    players_22_df = pd.read_csv('fifa_player_data/players_22.csv')
    fixture_df = pd.read_csv('data/fixture_overview.csv')

    # find mapping for club names in fixture that are not in fifa player list
    mapping, list_of_countries_in_fixtures = \
        create_fixture_to_fifa_club_mapping(fixture_df, players_22_df, all_country_names)

    # map the team names so that fixtures match fifa clubs
    fixture_df['HomeTeam'] = fixture_df['HomeTeam'].map(mapping).fillna(fixture_df['HomeTeam'])
    fixture_df['AwayTeam'] = fixture_df['AwayTeam'].map(mapping).fillna(fixture_df['AwayTeam'])

    # evaluate how the preprocessing went
    unique_clubs_fifa_players = players_22_df['club_name'].unique().astype(str)
    unique_clubs_fixtures_overview = fixture_df['HomeTeam'].unique().astype(str)
    non_matched_club_names = \
        [club_name for club_name in unique_clubs_fixtures_overview
         if club_name not in unique_clubs_fifa_players and club_name not in list_of_countries_in_fixtures]

    print(
        f'Percentage of non_matched_club_names '
        f'{len(non_matched_club_names) / len(unique_clubs_fixtures_overview) * 100} %')

    # filter unmatched club names
    fixture_df = fixture_df[(fixture_df['HomeTeam'].isin(unique_clubs_fifa_players) |
                            fixture_df['HomeTeam'].isin(all_country_names))]
    fixture_df = fixture_df[(fixture_df['AwayTeam'].isin(unique_clubs_fifa_players) |
                            fixture_df['AwayTeam'].isin(all_country_names))]

    # add if the game was national or not
    fixture_df['national_game'] = 0
    fixture_df.loc[fixture_df['HomeTeam'].isin(all_country_names), 'national_game'] = 1

    fixture_df.to_csv('prepped_data_sources/prepped_fixture_overview.csv')





