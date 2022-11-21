"""This file contains code that converts the jsons into csv and removes the useless and double games."""

from tqdm import tqdm
import json
import os

import pandas as pd


def convert_json_to_csv(matches):
    """Convert json to csv and sanitise."""

    # load json as dataframe
    fixture_df = pd.DataFrame(matches)

    # drop games that have not yet been played
    fixture_df = fixture_df.dropna(subset=['HomeTeamScore', 'AwayTeamScore'])

    # drop duplicates and unnecessary columns
    fixture_df = fixture_df.drop_duplicates()
    fixture_df = fixture_df.drop(columns=['Location', 'RoundNumber', 'MatchNumber', 'Group'])

    # parse dates and get year features
    fixture_df['DateUtc'] = pd.to_datetime(fixture_df['DateUtc'], format='%Y-%m-%d %H:%M:%SZ')
    fixture_df['Year'] = fixture_df['DateUtc'].dt.year

    return fixture_df


def flatten(l):
    return [item for sublist in l for item in sublist]


def merge_same_league_year_games(competition, list_of_files, json_location):
    """Merge jsons from the same league and the same year so that they can be filtered out."""

    files_in_same_competition = [json.load(open(json_location + '/' + file_path)) for file_path in list_of_files if
                                 competition in file_path]
    files_in_same_competition = flatten(files_in_same_competition)

    return files_in_same_competition


def process_all_jsons_to_rdb(json_location, save_location):
    """
    Process all jsons in json_location and save them to save_location.

    :param json_location: path to jsons
    :param save_location: location to save rdb
    :return:
    """

    list_of_files = os.listdir(json_location)

    different_competitions = [file.split('_')[0] for file in list_of_files]
    different_competitions = list(set(different_competitions))

    # merge and preprocess games from same competitions
    processed_fixtures = []
    for competition in tqdm(different_competitions):
        all_matches_in_competitions = merge_same_league_year_games(competition, list_of_files, json_location)
        fixture_df = convert_json_to_csv(all_matches_in_competitions)
        processed_fixtures.append(fixture_df)

    # combine all information and save the rdb
    full_dataset_df = pd.concat(processed_fixtures)
    full_dataset_df.to_csv(save_location, index=False)


if __name__ == '__main__':
    process_all_jsons_to_rdb('fixtures', 'fixture_overview.csv')
