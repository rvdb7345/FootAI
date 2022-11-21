"""Formats the data to a column format with the other fixes.
Data fram: https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017/code?select=results.csv"""

import pandas as pd


if __name__ == '__main__':
    games_df = pd.read_csv('data/add_international_games.csv', parse_dates=True)

    # cut off years for which we can not make teams
    games_df['date'] = pd.to_datetime(games_df['date'], format='%Y-%m-%d')
    games_df['Year'] = games_df['date'].dt.year
    games_df = games_df[games_df['Year'] > 2015]

    # rename columns so that they match our other fixtures
    games_df.rename({'home_team': 'HomeTeam', 'away_team': 'AwayTeam',
                     'home_score': 'HomeTeamScore', 'away_score': "AwayTeamScore",
                     "date": "DateUtc"}, inplace=True, axis=1)
    games_df['national_game'] = 1

    # swap home and away team to balance out home player advance if the game was played on neutral grounds
    swap = games_df[games_df['neutral']].rename(columns={'HomeTeam': 'AwayTeam', 'AwayTeam': 'HomeTeam',
                                                                   'HomeTeamScore': 'AwayTeamScore',
                                                                   'AwayTeamScore': 'HomeTeamScore'})
    games_df = games_df.append(swap).sort_index(ignore_index=True)


    # drop unnecessary columns
    games_df.drop(columns=['tournament','city','country','neutral'], inplace=True)

    # save the
    games_df.to_csv('fixture_overview_international.csv')
    print(games_df)
