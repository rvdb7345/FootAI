import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from sklearn import preprocessing
import pickle
import tensorflow_decision_forests as tfdf
from sklearn.metrics import accuracy_score, confusion_matrix

MAX_EPOCHS = 4000


def sign_penalty(y_true, y_pred):
    """Function that assigns a heavier weight to errors that fall into the wrong quandrant.
    For match outcome prediction it is most important that we predict the right sign, because this gets the most points.
    """
    penalty = 5.
    loss = tf.where(tf.less(y_true * y_pred, 0),
                    penalty * tf.abs(y_true - y_pred),
                    tf.abs(y_true - y_pred))

    return tf.reduce_mean(loss, axis=-1)


def fit(model, X_train, y_train, X_val, y_val):
    """Fit the model"""
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=200,
                                                      mode='min',
                                                      restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=MAX_EPOCHS,
                        validation_data=(X_val, y_val),
                        callbacks=[early_stopping],
                        batch_size=32)
    return history



def create_tensorflow_model_regressor(num_features):
    """Compose tensorflow model."""
    inputs = tf.keras.Input(shape=(len(num_features),))
    # x = tf.keras.layers.Dropout(0.2)(inputs)
    x = tf.keras.layers.Dense(512, activation=tf.nn.relu)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(256, activation=tf.nn.relu)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(64, activation=tf.nn.relu)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(32, activation=tf.nn.relu)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(16, activation=tf.nn.relu)(x)
    outputs = tf.keras.layers.Dense(1, activation="linear")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=20000,
        decay_rate=0.9)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)

    tf.keras.losses.sign_penalty = sign_penalty

    model.compile(loss=sign_penalty,
                  optimizer=optimizer,
                  metrics=['mae'])

    return model


def plot_loss(history):
    """Plot the loss over training."""
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_predictions(true, preds, label=''):
    """Plot the predicted value versus the real values."""
    fix, ax = plt.subplots()
    plt.title(f'{label}')
    plt.xlabel('True value')
    plt.ylabel('Predicted value')
    plt.scatter(true, preds)

    # create quadrant
    roc_t = 0.0
    roc_v = 0.0
    ax.fill_between([min(np.append(true, preds)), roc_t], min(np.append(true, preds)), roc_v,
                    alpha=0.3, color='#1F98D0')
    ax.fill_between([roc_t, max(np.append(true, preds))], min(np.append(true, preds)), roc_v,
                    alpha=0.3, color='#DA383D')
    ax.fill_between([min(np.append(true, preds)), roc_t], roc_v, max(np.append(true, preds)),
                    alpha=0.3, color='#DA383D')
    ax.fill_between([roc_t, max(np.append(true, preds))], roc_v, max(np.append(true, preds)),
                    alpha=0.3, color='#1F98D0')

    plt.grid(True)
    plt.show()

def create_feature_names(line_definitions, features_to_use, features_to_extract, teams):
    # generate list of features to load from the prepped dataset
    for line_key, item in line_definitions.items():
        for team in teams:
            # loop over the general features
            for general_feat in features_to_extract['general']:
                features_to_use.append(f'{team}_{general_feat}_{line_key}')

            # line specific features
            if line_key == 'goal':
                for goal_feat in features_to_extract['goal']:
                    features_to_use.append(f'{team}_{goal_feat}_{line_key}')
            if line_key == 'field':
                for field_feat in features_to_extract['field']:
                    features_to_use.append(f'{team}_{field_feat}_{line_key}')
            if line_key == 'def':
                for def_feat in features_to_extract['def']:
                    features_to_use.append(f'{team}_{def_feat}_{line_key}')
            if line_key == 'att':
                for att_feat in features_to_extract['att']:
                    features_to_use.append(f'{team}_{att_feat}_{line_key}')

    return features_to_use

def create_predictable(fixture_overview_df):
    # add predictable
    fixture_overview_df['team_victory'] = fixture_overview_df['HomeTeamScore'] - fixture_overview_df['AwayTeamScore']

    # cap large differences between matches to prevent overfitting on outliers
    fixture_overview_df.loc[fixture_overview_df['team_victory'] > 4, 'team_victory'] = 4
    fixture_overview_df.loc[fixture_overview_df['team_victory'] < -4, 'team_victory'] = -4

    return fixture_overview_df



if __name__ == '__main__':
    experiment_name = 'gbm_medium'

    fixture_overview_df = pd.read_csv('prepped_data_sources/prepped_data_set.csv')
    fixture_overview_df = create_predictable(fixture_overview_df)

    fixture_overview_df.dropna(axis=0, inplace=True)

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
    # teams = ['rel']


    # list for the features and the base of features that are not player dependent
    features_to_use = ['national_game']
    features_to_use = create_feature_names(line_definitions, features_to_use, features_to_extract, teams)

    # print(fixture_overview_df[features_to_use + ['team_victory']].corr()['team_victory'].sort_values(ascending=False)[0:30])

    # preprocess the data
    fitted_scaler = preprocessing.RobustScaler()
    fitted_scaler.fit(fixture_overview_df[features_to_use].values)
    scaled_X = fitted_scaler.transform(fixture_overview_df[features_to_use].values)

    pickle.dump(fitted_scaler, open(f'{experiment_name}_scaler.pkl', 'wb'))

    # split data in to train, validation and test
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, fixture_overview_df['team_victory'].values,
        test_size=0.2, random_state=42, shuffle=True)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.2, random_state=42, shuffle=True)

    print(f'train length: {len(X_train)}')
    print(f'val length: {len(X_val)}')
    print(f'test length: {len(X_test)}')

    # create the model
    # tf.keras.losses.sign_penalty = sign_penalty
    # tuner = tfdf.tuner.RandomSearch(num_trials=30)
    # tuner.choice("num_candidate_attributes_ratio", [1.0, 0.9, 0.8, 0.7, 0.6])
    #
    # local_search_space = tuner.choice("growing_strategy", ["LOCAL"])
    # local_search_space.choice("max_depth", [4, 12, 15, 25, 30, 40, 50, 70 ])
    # local_search_space.choice("num_trees", [100, 200, 300, 1500, 2000, 5000, 7500, 10000])
    # local_search_space.choice("l2_regularization", [0.0, 0.1, 0.2])
    # local_search_space.choice("l1_regularization", [0.0, 0.1, 0.2])
    #
    # global_search_space = tuner.choice(
    #     "growing_strategy", ["BEST_FIRST_GLOBAL"], merge=True)
    # global_search_space.choice("max_num_nodes", [16, 32, 64, 128, 256])

    model = tfdf.keras.GradientBoostedTreesModel(task=tfdf.keras.Task.REGRESSION, num_trees=8000, max_depth=70,
                                                 verbose=2, l2_regularization=0.3, l1_regularization=0.3,
                                                 shrinkage=0.02, min_examples=100,
                                                 compute_permutation_variable_importance=True,
                                                 early_stopping='LOSS_INCREASE',
                                                 early_stopping_num_trees_look_ahead=100)
    # model = create_tensorflow_model_regressor(features_to_use)

    # fit the model and visualise the results
    train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(fixture_overview_df[features_to_use + ['team_victory']].iloc[:20000], label='team_victory', task=tfdf.keras.Task.REGRESSION)
    test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(fixture_overview_df[features_to_use + ['team_victory']].iloc[20000:], label='team_victory', task=tfdf.keras.Task.REGRESSION)

    print(np.isnan(y_train).sum())
    history = model.fit(train_ds, verbose=2)

    inspector = model.make_inspector()
    logs = inspector.training_logs()
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot([log.num_trees for log in logs], [log.evaluation.rmse for log in logs])
    plt.xlabel("Number of trees")
    plt.ylabel("Accuracy (out-of-bag)")

    plt.subplot(1, 2, 2)
    plt.plot([log.num_trees for log in logs], [log.evaluation.loss for log in logs])
    plt.xlabel("Number of trees")
    plt.ylabel("Logloss (out-of-bag)")

    plt.show()

    print(inspector.variable_importances())

    # plot_loss(history)

    # predict for the training and the test data and visualise the results
    test_pred = model.predict(test_ds, verbose=0)
    train_pred = model.predict(train_ds, verbose=0)

    plot_predictions(fixture_overview_df['team_victory'].iloc[20000:], test_pred, label='test')
    plot_predictions(fixture_overview_df['team_victory'].iloc[:20000], train_pred, label='train')

    # calculate the test and training scores
    print(f"Test score: {mean_absolute_error(fixture_overview_df['team_victory'].iloc[20000:], test_pred)}")
    print(f"Train score: {mean_absolute_error(fixture_overview_df['team_victory'].iloc[:20000], train_pred)}")

    test_win_true = fixture_overview_df['team_victory'].iloc[20000:].copy()
    test_win_true[(test_win_true < 0.5) & (test_win_true > -0.5)] = 0
    test_win_true[(test_win_true > 0.5)] = 1
    test_win_true[(test_win_true < -0.5)] = -1

    test_pred_classification = test_pred.copy()
    test_pred_classification[(test_pred_classification < 0.5) & (test_pred_classification > -0.5)] = 0
    test_pred_classification[(test_pred_classification > 0.5)] = 1
    test_pred_classification[(test_pred_classification < -0.5)] = -1

    print("Test confusion matrix \n", confusion_matrix(test_win_true, test_pred_classification))
    print(f'Test score: {accuracy_score(test_win_true, test_pred_classification)}')

    model.save(f"trained_models/{experiment_name}_regression_model_{mean_absolute_error(fixture_overview_df['team_victory'].iloc[20000:], test_pred)}")
