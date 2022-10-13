import csv
import itertools
import sys
import numpy as np
import pandas as pd
from sklearn import preprocessing
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.mixture import GaussianMixture
from sklearn.multioutput import ClassifierChain

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, ShuffleSplit
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, SVR

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
np.set_printoptions(threshold=sys.maxsize)

player_df = pd.read_csv('players_21.csv')
match_df = pd.read_csv('uefa-euro-2020-CentralEuropeStandardTime.csv')
qualification_rounds_2020 = pd.read_csv('international-uefa-euro-qualifiers-matches-2019-to-2020-stats.csv')
country_rankings = pd.read_csv('fifa_ranking-2021-05-27.csv')
qualification_rounds = pd.read_csv('international-uefa-euro-qualifiers-matches-2019-to-2020-stats.csv')
# qualification_rounds_2016 = pd.read_csv('international-uefa-euro-qualifiers-matches-2016-to-2016-stats.csv')
# qualification_rounds = pd.concat((qualification_rounds_2016, qualification_rounds_2020))

historic_player_df = pd.concat((pd.read_csv('players_20.csv'), pd.read_csv('players_19.csv'), pd.read_csv(
    'players_18.csv'), pd.read_csv('players_17.csv'), pd.read_csv('players_16.csv'), pd.read_csv(
    'players_15.csv'))).drop_duplicates(subset=['long_name'])

played_matches = match_df.dropna(subset=['Result'])

played_matches.rename({'Home Team': 'home_team_name', 'Away Team': 'away_team_name'}, axis=1, inplace=True)
played_matches['home_team_goal_count'] = played_matches['Result'].apply(lambda x: int(x.split('-')[0]))
played_matches['away_team_goal_count'] = played_matches['Result'].apply(lambda x: int(x.split('-')[1]))

# print(played_matches)
qualification_rounds = qualification_rounds.append(played_matches)
qualification_rounds = qualification_rounds.append(played_matches)
qualification_rounds = qualification_rounds.append(played_matches)
qualification_rounds = qualification_rounds.append(played_matches)

# print(qualification_rounds)

planned_matches = match_df[match_df['Round Number'] == 'Semi Finals']
match_df = match_df[match_df['Round Number'] == 'Semi Finals']

all_playing_countries = np.concatenate((np.array(planned_matches['Home Team'].unique()),
                                        np.array(planned_matches['Away Team'].unique())))

# print(all_playing_countries)

# useful_features =

num_of_top_players = 14

# numerical_columns = ['age', 'height_cm', 'weight_kg', 'league_rank', 'overall', 'potential', 'value_eur',
#                      'wage_eur', 'international_reputation', 'skill_moves', 'pace', 'shooting', 'passing',
#                      'dribbling', 'defending', 'physic','attacking_crossing', 'attacking_finishing',
#                      'attacking_heading_accuracy', 'attacking_short_passing', 'attacking_volleys', 'skill_dribbling',
#                      'skill_curve', 'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control',
#                      'movement_acceleration', 'movement_sprint_speed', 'movement_agility', 'movement_reactions',
#                      'movement_balance', 'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength',
#                      'power_long_shots', 'mentality_aggression', 'mentality_interceptions', 'mentality_positioning',
#                      'mentality_vision', 'mentality_penalties', 'mentality_composure', 'defending_standing_tackle',
#                      'defending_sliding_tackle', 'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking',
#                      'goalkeeping_positioning', 'goalkeeping_reflexes']

numerical_columns = ['overall', 'international_reputation', 'wage_eur']
categorical_columns = ['player_positions', 'preferred_foot', 'work_rate', 'body_type']

X = pd.DataFrame(columns=['average_overall_score_home', 'average_overall_score_away', 'average_international_home',
                          'average_international_away', 'compared_overall_score',
                          'compared_international', 'gk_compared', 'compared_dribbling', 'compared_passing',
                          'compared_shooting', 'compared_defending', 'compared_physic', 'compared_pace',
                          'ranking_compared', 'scoring_compared'])
# X = pd.DataFrame(columns = [['average_overall_score_home', 'average_overall_score_away'] +
#                            ['average_international_home', 'average_international_away', 'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking',
#                           'goalkeeping_positioning', 'goalkeeping_reflexes', 'pace_mean', 'shooting_mean', 'passing_mean',
#                       'dribbling_mean', 'defending_mean', 'physic_mean']])
y = []
y_goals = []
idx_number = 0

player_df['player_positions'] = player_df['player_positions'].apply(lambda x: x.split(',')[0])

for idx, row in qualification_rounds.iterrows():
    top_players_home_team = player_df[player_df['nationality'] == row['home_team_name']].sort_values(
        by=['international_reputation', 'overall'], ascending=False).head(num_of_top_players)
    top_players_away_team = player_df[player_df['nationality'] == row['away_team_name']].sort_values(
        by=['international_reputation', 'overall'], ascending=False).head(num_of_top_players)

    if len(top_players_home_team) < num_of_top_players:
        historic_players_home_team = historic_player_df[historic_player_df['nationality'] == row[
            'home_team_name']].sort_values(
            by=['international_reputation', 'overall'], ascending=False).head(
            num_of_top_players - len(top_players_home_team))

        top_players_home_team = pd.concat((top_players_home_team, historic_players_home_team))
    if len(top_players_away_team) < num_of_top_players:
        historic_players_away_team = historic_player_df[historic_player_df['nationality'] == row[
            'away_team_name']].sort_values(
            by=['international_reputation', 'overall'], ascending=False).head(
            num_of_top_players - len(top_players_away_team))
        top_players_away_team = pd.concat((top_players_away_team, historic_players_away_team))

    # # national_top_players = top_players.dropna(subset=["nation_jersey_number"])
    # if len(top_players_home_team) < num_of_top_players:
    #     print(row['home_team_name'], ' has only {} players'.format(len(top_players_home_team)))
    # if len(top_players_away_team) < num_of_top_players:
    #     print(row['away_team_name'], ' has only {} players'.format(len(top_players_away_team)))

    if len(top_players_home_team) == num_of_top_players and len(top_players_away_team) == num_of_top_players:
        X.loc[idx_number, ['average_overall_score_home', 'average_overall_score_away']] = \
            [top_players_home_team['overall'].mean(), top_players_away_team['overall'].mean()]
        X.loc[idx_number, ['average_international_home', 'average_international_away']] = \
            [top_players_home_team['international_reputation'].mean(),
             top_players_away_team['international_reputation'].mean()]

        X.loc[idx_number, ['compared_overall_score']] = top_players_home_team['overall'].mean() - top_players_away_team[
            'overall'].mean()
        X.loc[idx_number, ['compared_international']] = top_players_home_team['international_reputation'].mean() - \
                                                        top_players_away_team['international_reputation'].mean()
        X.loc[idx_number, ['compared_dribbling']] = top_players_home_team.nlargest(3, 'dribbling')['dribbling'].mean() - \
                                                    top_players_away_team.nlargest(3, 'dribbling')['dribbling'].mean()
        X.loc[idx_number, ['compared_passing']] = top_players_home_team.nlargest(5, 'passing')['passing'].mean() - \
                                                  top_players_away_team.nlargest(5, 'passing')['passing'].mean()
        X.loc[idx_number, ['compared_shooting']] = top_players_home_team.nlargest(2, 'shooting')['shooting'].mean() - \
                                                   top_players_away_team.nlargest(2, 'shooting')['shooting'].mean()
        X.loc[idx_number, ['compared_defending']] = top_players_home_team.nlargest(2, 'defending')['defending'].mean() - \
                                                    top_players_away_team.nlargest(2, 'defending')['defending'].mean()
        X.loc[idx_number, ['compared_physic']] = top_players_home_team.nlargest(11, 'physic')['physic'].mean() - \
                                                 top_players_away_team.nlargest(11, 'physic')['physic'].mean()
        X.loc[idx_number, ['compared_pace']] = top_players_home_team.nlargest(2, 'pace')['pace'].mean() - \
                                               top_players_away_team.nlargest(2, 'pace')['pace'].mean()

        X.loc[idx_number, ['gk_compared']] = top_players_home_team[
                                                 ['goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking',
                                                  'goalkeeping_positioning', 'goalkeeping_reflexes']].max(
            axis=0).mean() - top_players_away_team[['goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking',
                                                    'goalkeeping_positioning', 'goalkeeping_reflexes']].max(
            axis=0).mean()

        X.loc[idx_number, ['ranking_compared']] = country_rankings.loc[country_rankings['country_full'] == row[
            'home_team_name']].tail(1)['rank'] - country_rankings.loc[country_rankings['country_full'] == row[
            'away_team_name']].tail(1)['rank']
        X.loc[idx_number, ['scoring_compared']] = country_rankings.loc[country_rankings['country_full'] == row[
            'home_team_name']].tail(1)['total_points'] - country_rankings.loc[country_rankings['country_full'] == row[
            'away_team_name']].tail(1)['total_points']
        # X.loc[idx_number, ['pace_mean', 'shooting_mean', 'passing_mean',
        #               'dribbling_mean', 'defending_mean', 'physic_mean']] = top_players_home_team[
        #     ['pace', 'shooting', 'passing',
        #      'dribbling', 'defending', 'physic']].max(axis=0)

        # X.loc[idx_number, ['goalkeeping_diving',  'goalkeeping_handling', 'goalkeeping_kicking',
        #                    'goalkeeping_positioning',  'goalkeeping_reflexes'] =

        if row.loc['home_team_goal_count'] - row['away_team_goal_count'] > 0:
            score = -1
        elif row.loc['home_team_goal_count'] - row['away_team_goal_count'] == 0:
            score = 0
        elif row.loc['home_team_goal_count'] - row['away_team_goal_count'] < 0:
            score = 1
        else:
            assert False, print('somehting has gone wrong')

        y_goals.append(row.loc['home_team_goal_count'] - row['away_team_goal_count'])
        y.append(score)
        idx_number += 1

X.fillna(0, inplace=True)

# print(player_df.head(5))

print(np.array(player_df.columns))

print('shape of X', X.shape)
print('shape of y', len(y))

# clf = LGBMClassifier(max_depth=10, random_state=0, n_jobs=-1)

# params_lgbm = {
#     'num_leaves': [3, 7, 14, 21, 28, 31, 50],
#     'learning_rate': [0.1, 0.03, 0.003],
#     'max_depth': [1, 3, 5, 10, 15],
#     'n_estimators': [5, 10, 20, 35, 50, 100, 200]
# }

params_lgbm = {
    'learning_rate': [0.03],
    'n_estimators': [5, 8, 16, 24],
    'num_leaves': [3, 6, 8],  # large num_leaves helps improve accuracy but might lead to over-fitting
    'boosting_type': ['dart'],  # for better accuracy -> try dart
    'objective': ['multiclass'],
    'max_bin': [510, 700, 1000, 2000, 3000],  # large max_bin helps improve accuracy but might slow down
    # training
    # progress
    'random_state': [42],
    'colsample_bytree': [0.66, 0.8, 1],
    'subsample': [0.7, 0.75, 0.9, 1],
    'reg_alpha': [0., 1, 1.2],
    'reg_lambda': [0, 1, 1.2, 1.4],
}

# params_svc = [{'kernel': ['rbf'], 'gamma': [1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1,'scale', 'auto'],
#                      'C': [1, 10, 100, 200, 300, 600, 1000], 'shrinking': [True, False], 'decision_function_shape': [
#         'ovo', 'ovr'],
#                'cache_size': [2000]}]

''' compare basic classifiers '''
# for classifier in [KNeighborsClassifier, SVC, LGBMClassifier, LinearDiscriminantAnalysis, GaussianNB,
#                    DecisionTreeClassifier, GaussianMixture, AdaBoostClassifier,MLPClassifier]:
#     try:
#         model = classifier(n_jobs=-1)
#     except:
#         model = classifier()
#     scores_tunes = cross_val_score(model, X, y, cv=5)
#     print(str(classifier), ' get the following score: ', scores_tunes, np.mean(scores_tunes), np.std(scores_tunes))


''' SVM machine '''
#
# grid = GridSearchCV(SVC(random_state=0, probability=False), params_svc, scoring='accuracy', cv=3, n_jobs=-1,
#                           verbose=5)
# grid.fit(X, y)
#
# print('the best parameters available', grid.best_params_)
#
# means = grid.cv_results_['mean_test_score']
# stds = grid.cv_results_['std_test_score']
# for mean, std, params in zip(means, stds, grid.cv_results_['params']):
#     print("%0.3f (+/-%0.03f) for %r"
#           % (mean, std * 2, params))
#
# reg = SVC(random_state=0)
# reg.fit(X_train, y_train)
#
# svc_tuned = grid.best_estimator_
#
# # clf.fit(X_train, y_train)
# scores_tunes = cross_val_score(svc_tuned, X, y, cv=5)
# print('scores of the tunes algorithm',  scores_tunes, np.mean(scores_tunes), np.std(scores_tunes))
#
# scores_basic = cross_val_score(reg, X, y, cv=5)
# print('scores of the basic algorithm',  scores_basic)


# ''' LGBMClassifier '''
# grid = RandomizedSearchCV(LGBMClassifier(random_state=0), params_lgbm, scoring='accuracy', cv=3, n_jobs=-1,
#                           verbose=3, n_iter=3000)
# grid.fit(X, y)
#
# print('the best parameters available', grid.best_params_)
#
# lgbm_tuned = grid.best_estimator_
# scores_tuned_lgbm = cross_val_score(lgbm_tuned, X, y, cv=5)
# print('SCore of the tuned lgbm algorithm: ', scores_tuned_lgbm, np.mean(scores_tuned_lgbm), np.std(scores_tuned_lgbm))
#

# assert False, print('ended')


cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)

# classification
svc_classification = SVC()
scores_tuned_svc = cross_val_score(svc_classification, X, y, cv=cv)
print('Accuracy of the svc algorithm: {:.3f} +- {:.3f}'.format(np.mean(scores_tuned_svc),
                                                               np.std(scores_tuned_svc)))

lgbm_classification = LGBMClassifier(**{'learning_rate': 0.1, 'max_depth': 1, 'n_estimators': 10, 'num_leaves': 3,
                                        'objective': 'multiclass'})
scores_tuned_lgbm = cross_val_score(lgbm_classification, X, y, cv=cv)
print('scores of the lgbm algorithm: {:.3f} +- {:.3f}'.format(np.mean(scores_tuned_lgbm), np.std(scores_tuned_lgbm)))

lgbm_classification.fit(X, y)
print(lgbm_classification.feature_importances_)

print(X.columns[lgbm_classification.feature_importances_ > 0])

# regression
svr_regression = SVR()
scores_tuned_svr = cross_val_score(svr_regression, X, y, cv=cv)
print('Prediction accuracy of the svc algorithm {:.3f} +- {:.3f}'.format(np.mean(scores_tuned_svr),
                                                                         np.std(scores_tuned_svr)))

lgbm_regression = LGBMRegressor(**{'learning_rate': 0.1, 'max_depth': 1, 'n_estimators': 10, 'num_leaves': 3})
scores_regression_lgbm = cross_val_score(lgbm_regression, X, y, cv=cv)
print('Prediction accuracy of the lgbm algorithm {:.3f} +- {:.3f}'.format(np.mean(scores_regression_lgbm),
                                                                          np.std(scores_regression_lgbm)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
X_train_goals, X_test_goals, y_train_goals, y_test_goals = train_test_split(X, y_goals, test_size=0.1, random_state=0)

svr_regression.fit(X_train_goals, y_train_goals)
predictions_reg = svr_regression.predict(X_test_goals)

svc_classification.fit(X_train, y_train)
predictions_clas = svc_classification.predict(X_test)

print('-1 = home team wins, 1 = away team wins')
for idx, prediction in enumerate(predictions_reg):
    print('{}, \t {} '.format(y_test[idx], predictions_clas[idx]))
    print('{}, \t {} '.format(y_test_goals[idx], predictions_reg[idx]))
    print('')
    # print('{0.3f}, \t {0.3f}'.format(y_test[idx] , prediction))

# scores_basic = cross_val_score(reg, X, y, cv=5)
# print('scores of the basic algorithm',  scores_basic)
X_predict = pd.DataFrame(columns=['average_overall_score_home', 'average_overall_score_away',
                                  'average_international_home',
                                  'average_international_away', 'compared_overall_score',
                                  'compared_international', 'gk_compared', 'compared_dribbling', 'compared_passing',
                                  'compared_shooting', 'compared_defending', 'compared_physic', 'compared_pace',
                                  'ranking_compared', 'scoring_compared'])

all_possible_matches = np.array(list(itertools.combinations(all_playing_countries, 2)))
match_df = pd.DataFrame(all_possible_matches, columns=['Home Team', 'Away Team'])

for idx, row in match_df.iterrows():
    top_players_home_team = player_df[player_df['nationality'] == row['Home Team']].sort_values(
        by=['international_reputation', 'overall'], ascending=False).head(num_of_top_players)
    top_players_away_team = player_df[player_df['nationality'] == row['Away Team']].sort_values(
        by=['international_reputation', 'overall'], ascending=False).head(num_of_top_players)

    if len(top_players_home_team) < num_of_top_players:
        historic_players_home_team = historic_player_df[historic_player_df['nationality'] == row[
            'Home Team']].sort_values(
            by=['international_reputation', 'overall'], ascending=False).head(
            num_of_top_players - len(top_players_home_team))

        top_players_home_team = pd.concat((top_players_home_team, historic_players_home_team))
    if len(top_players_away_team) < num_of_top_players:
        historic_players_away_team = historic_player_df[historic_player_df['nationality'] == row[
            'Away Team']].sort_values(
            by=['international_reputation', 'overall'], ascending=False).head(
            num_of_top_players - len(top_players_away_team))
        top_players_away_team = pd.concat((top_players_away_team, historic_players_away_team))

    # # national_top_players = top_players.dropna(subset=["nation_jersey_number"])
    # if len(top_players_home_team) < num_of_top_players:
    #     print(row['home_team_name'], ' has only {} players'.format(len(top_players_home_team)))
    # if len(top_players_away_team) < num_of_top_players:
    #     print(row['away_team_name'], ' has only {} players'.format(len(top_players_away_team)))

    if len(top_players_home_team) == num_of_top_players and len(top_players_away_team) == num_of_top_players:
        X_predict.loc[idx_number, ['average_overall_score_home', 'average_overall_score_away']] = \
            [top_players_home_team['overall'].mean(), top_players_away_team['overall'].mean()]
        X_predict.loc[idx_number, ['average_international_home', 'average_international_away']] = \
            [top_players_home_team['international_reputation'].mean(),
             top_players_away_team['international_reputation'].mean()]

        X_predict.loc[idx_number, ['compared_overall_score']] = top_players_home_team['overall'].mean() - \
                                                                top_players_away_team['overall'].mean()
        X_predict.loc[idx_number, ['compared_international']] = top_players_home_team[
                                                                    'international_reputation'].mean() - \
                                                                top_players_away_team['international_reputation'].mean()
        X_predict.loc[idx_number, ['compared_dribbling']] = top_players_home_team.nlargest(3, 'dribbling')[
                                                                'dribbling'].mean() - \
                                                            top_players_away_team.nlargest(3, 'dribbling')[
                                                                'dribbling'].mean()
        X_predict.loc[idx_number, ['compared_passing']] = top_players_home_team.nlargest(5, 'passing')[
                                                              'passing'].mean() - \
                                                          top_players_away_team.nlargest(5, 'passing')['passing'].mean()
        X_predict.loc[idx_number, ['compared_shooting']] = top_players_home_team.nlargest(2, 'shooting')[
                                                               'shooting'].mean() - \
                                                           top_players_away_team.nlargest(2, 'shooting')[
                                                               'shooting'].mean()
        X_predict.loc[idx_number, ['compared_defending']] = top_players_home_team.nlargest(2, 'defending')[
                                                                'defending'].mean() - \
                                                            top_players_away_team.nlargest(2, 'defending')[
                                                                'defending'].mean()
        X_predict.loc[idx_number, ['compared_physic']] = top_players_home_team.nlargest(11, 'physic')['physic'].mean() - \
                                                         top_players_away_team.nlargest(11, 'physic')['physic'].mean()
        X_predict.loc[idx_number, ['compared_pace']] = top_players_home_team.nlargest(2, 'pace')['pace'].mean() - \
                                                       top_players_away_team.nlargest(2, 'pace')['pace'].mean()

        X_predict.loc[idx_number, ['gk_compared']] = top_players_home_team[
                                                         ['goalkeeping_diving', 'goalkeeping_handling',
                                                          'goalkeeping_kicking',
                                                          'goalkeeping_positioning', 'goalkeeping_reflexes']].max(
            axis=0).mean() - top_players_away_team[['goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking',
                                                    'goalkeeping_positioning', 'goalkeeping_reflexes']].max(
            axis=0).mean()

        X_predict.loc[idx_number, ['ranking_compared']] = country_rankings.loc[country_rankings['country_full'] == row[
            'Home Team']].tail(1)['rank'] - country_rankings.loc[country_rankings['country_full'] == row[
            'Away Team']].tail(1)['rank']
        X_predict.loc[idx_number, ['scoring_compared']] = country_rankings.loc[country_rankings['country_full'] == row[
            'Home Team']].tail(1)['total_points'] - country_rankings.loc[country_rankings['country_full'] == row[
            'Away Team']].tail(1)['total_points']
        X_predict.loc[idx_number, ['away_team']] = row['Away Team']
        X_predict.loc[idx_number, ['home_team']] = row['Home Team']

        # X.loc[idx_number, ['pace_mean', 'shooting_mean', 'passing_mean',
        #               'dribbling_mean', 'defending_mean', 'physic_mean']] = top_players_home_team[
        #     ['pace', 'shooting', 'passing',
        #      'dribbling', 'defending', 'physic']].max(axis=0)

        # X.loc[idx_number, ['goalkeeping_diving',  'goalkeeping_handling', 'goalkeeping_kicking',
        #                    'goalkeeping_positioning',  'goalkeeping_reflexes'] =

        idx_number += 1

X_predict.fillna(0, inplace=True)

# classification
svc_classification = SVC()
svc_classification.fit(X, y)
svcclass_predictions = svc_classification.predict(X_predict)

lgbm_classification = LGBMClassifier(**{'learning_rate': 0.1, 'max_depth': 1, 'n_estimators': 10, 'num_leaves': 3,
                                        'objective': 'multiclass'})
lgbm_classification.fit(X, y, categorical_feature=['away_team', 'home_team'])
lgbmclass_predictions = lgbm_classification.predict(X_predict)

# regression
svr_regression = SVR()
svr_regression.fit(X, y_goals)
svrgoals_predictions = svr_regression.predict(X_predict)

lgbm_regression = LGBMRegressor(**{'learning_rate': 0.1, 'max_depth': 1, 'n_estimators': 10, 'num_leaves': 3})
lgbm_regression.fit(X, y_goals, categorical_feature=['away_team', 'home_team'])
lgbmgoals_predictions = lgbm_regression.predict(X_predict)

points = pd.DataFrame({'country': all_playing_countries, 'points': np.zeros(len(all_playing_countries))})

print('-1 = home team wins, 1 = away team wins')
idx_count = 0
for idx, prediction in enumerate(svcclass_predictions):

    print('the predicted game:')
    print('home team: {}, away team: {}'.format(match_df.loc[idx_count, 'Home Team'], match_df.loc[idx_count,
                                                                                                   'Away Team']))

    if svcclass_predictions[idx_count] == -1:
        print(match_df.loc[idx_count, 'Home Team'],
              ' wins with {}/{} goals'.format(svrgoals_predictions[idx], lgbmgoals_predictions[idx]))
        points.loc[points['country'] == match_df.loc[idx_count, 'Home Team'], 'points'] += 2
    if svcclass_predictions[idx_count] == 0:
        print('draw')
        points.loc[points['country'] == match_df.loc[idx_count, 'Home Team'], 'points'] += 1
        points.loc[points['country'] == match_df.loc[idx_count, 'Away Team'], 'points'] += 1

    if svcclass_predictions[idx_count] == 1:
        print(match_df.loc[idx_count, 'Away Team'], ' wins with {}/{} goals'.format(-svrgoals_predictions[idx],
                                                                                    -lgbmgoals_predictions[idx]))
        points.loc[points['country'] == match_df.loc[idx_count, 'Away Team'], 'points'] += 1

    print('\n')

    # print('svc predicts: {} with goals {}'.format(svcclass_predictions[idx_count], -svrgoals_predictions[idx]))
    # print('lgbm predicts: {} with goals {}'.format(lgbmclass_predictions[idx_count], -lgbmgoals_predictions[idx]))

    idx_count += 1

print(points.sort_values(by=['points'], ascending=False).drop_duplicates())