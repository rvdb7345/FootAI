import shap
import tensorflow as tf
from sklearn import preprocessing
import pandas as pd
from train_evaluate_dnn import create_predictable, create_feature_names, sign_penalty
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    fixture_overview_df = pd.read_csv('prepped_data_sources/prepped_data_set.csv')
    fixture_overview_df = create_predictable(fixture_overview_df)

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

    # preprocess the data
    scaled_X = preprocessing.RobustScaler().fit_transform(fixture_overview_df[features_to_use].values)

    # split data in to train, validation and test
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, fixture_overview_df['team_victory'].values,
        test_size=0.2, random_state=42, shuffle=True)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.2, random_state=42, shuffle=True)

    tf.keras.losses.sign_penalty = sign_penalty
    new_model = tf.keras.models.load_model('regression_model_0.818306543109031.csv',
                                           custom_objects={'sign_penalty': sign_penalty})

    # explain predictions of the model on four images
    shap.explainers._deep.deep_tf.op_handlers[
        "AddV2"] = shap.explainers._deep.deep_tf.passthrough  # this solves the "shap_ADDV2" problem but another one will appear
    shap.explainers._deep.deep_tf.op_handlers[
        "FusedBatchNormV3"] = shap.explainers._deep.deep_tf.passthrough  # this solves the next problem which allows you to run the DeepExplainer.
    explainer = shap.DeepExplainer(new_model, X_train[0:1000])
    # ...or pass tensors directly
    # e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
    shap_values = explainer.shap_values( X_test[:1000, :])

    print(explainer.expected_value)
    print(shap_values)
    shap.force_plot(explainer.expected_value[1], shap_values[1][:1000, :], X_test[:1000, :])

