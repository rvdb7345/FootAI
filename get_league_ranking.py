from feature_engineer import engineer_features
import pandas as pd
import tensorflow as tf
from train_evaluate_dnn import sign_penalty, create_feature_names
import tensorflow_decision_forests as tfdf
import pickle

if __name__ == '__main__':
    matches_to_predict = pd.read_csv('data/fifa-world-cup-2022-UTC.csv')
    matches_to_predict['national_game'] = 1
    matches_to_predict['Year'] = 2022
    matches_to_predict.rename({'Home Team': 'HomeTeam', 'Away Team': 'AwayTeam'}, axis=1, inplace=True)

    matches_to_predict.dropna(axis=1, inplace=True)
    matches_to_predict.dropna(axis=0, inplace=True)

    all_participants = matches_to_predict['HomeTeam'].unique().tolist()
    all_participants.append('Costa Rica')
    all_participants.append('Saudi Arabia')

    matches_to_predict = pd.DataFrame(columns=matches_to_predict.columns)
    for home in all_participants:
        for away in all_participants:
            if home != away:
                matches_to_predict = matches_to_predict.append({"HomeTeam": home, 'AwayTeam': away, "Year": 2022,
                                                                "national_game": 1}, ignore_index=True)

    matches_to_predict.dropna(axis=1, inplace=True)
    matches_to_predict.dropna(axis=0, inplace=True)

    print(matches_to_predict)

    engineered_df = engineer_features(matches_to_predict)
    print(engineered_df)

    tf.keras.losses.sign_penalty = sign_penalty
    model = tf.keras.models.load_model('trained_models/gbm_regression_model_0.5366107094799906')

    scaler = pickle.load(open('scalers/gbm_scaler.pkl', 'rb'))

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

    # the different positions and teams
    teams = ['home_team', 'away_team', 'rel']

    # list for the features and the base of features that are not player dependent
    features_to_use = ['national_game']
    features_to_use = create_feature_names(line_definitions, features_to_use, features_to_extract, teams)

    predict_data = scaler.transform(engineered_df.loc[:, features_to_use])

    test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(engineered_df.loc[:, features_to_use])

    goal_diffs = model.predict(test_ds)

    matches_to_predict['goal_difference'] = goal_diffs
    matches_to_predict['rounded_diff'] = matches_to_predict['goal_difference'].apply(lambda x: round(x))
    print(matches_to_predict[['HomeTeam', 'AwayTeam', 'goal_difference', 'rounded_diff']])

    total_ranking = []

    for participant in all_participants:
        outcomes_home = matches_to_predict[matches_to_predict['HomeTeam'] == participant]
        outcomes_away = matches_to_predict[matches_to_predict['AwayTeam'] == participant]

        total_score = 0
        total_score += len(outcomes_home[outcomes_home['rounded_diff'] >= 1]) * 3
        total_score += len(outcomes_home[outcomes_home['rounded_diff'] == 0]) * 1
        total_score += len(outcomes_away[outcomes_away['rounded_diff'] <= -1]) * 3
        total_score += len(outcomes_away[outcomes_away['rounded_diff'] == 0]) * 1

        total_ranking.append((participant, total_score))

    sorted_outcome = sorted(
        total_ranking,
        key=lambda t: t[1],
        reverse=True
    )

    print(sorted_outcome)


