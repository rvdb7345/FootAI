from pyexpat import features

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix


MAX_EPOCHS = 1000


def sign_penalty(y_true, y_pred):
    """Function that assigns a heavier weight to errors that fall into the wrong quandrant.
    For match outcome prediction it is most important that we predict the right sign, because this gets the most points.
    """
    penalty = 5.
    loss = tf.where(tf.less(y_true * y_pred, 0),
                    penalty * tf.abs(y_true - y_pred),
                    tf.abs(y_true - y_pred))

    return tf.reduce_mean(loss, axis=-1)


def fit(model, X_train_home, X_train_away, y_train_reg, y_train_class, X_val_home, X_val_away, y_val_reg, y_val_class):
    """Fit the model"""
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=200,
                                                      mode='min')

    history = model.fit([X_train_home, X_train_away], [y_train_reg, y_train_class], epochs=MAX_EPOCHS,
                        validation_data=([X_val_home, X_val_away], [y_val_reg, y_val_class]),
                        callbacks=[early_stopping],
                        batch_size=32)
    return history



def create_tensorflow_model_regressor(num_features):
    """Compose tensorflow model."""
    inputs_1 = tf.keras.Input(shape=(len(num_features),))
    inputs_2 = tf.keras.Input(shape=(len(num_features),))

    # home team
    x_home = tf.keras.layers.Dense(512, activation=tf.nn.relu)(inputs_1)
    x_home = tf.keras.layers.BatchNormalization()(x_home)
    x_home = tf.keras.layers.Dense(256, activation=tf.nn.relu)(x_home)
    x_home = tf.keras.layers.BatchNormalization()(x_home)
    x_home = tf.keras.layers.Dense(128, activation=tf.nn.relu)(x_home)
    x_home = tf.keras.layers.BatchNormalization()(x_home)

    x_away = tf.keras.layers.Dense(512, activation=tf.nn.relu)(inputs_2)
    x_away = tf.keras.layers.BatchNormalization()(x_away)
    x_away = tf.keras.layers.Dense(256, activation=tf.nn.relu)(x_away)
    x_away = tf.keras.layers.BatchNormalization()(x_away)
    x_away = tf.keras.layers.Dense(128, activation=tf.nn.relu)(x_away)
    x_away = tf.keras.layers.BatchNormalization()(x_away)

    x = tf.keras.layers.Concatenate()([x_home, x_away])

    x = tf.keras.layers.Dense(256, activation=tf.nn.relu)(x)
    x = tf.keras.layers.Dense(64, activation=tf.nn.relu)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(32, activation=tf.nn.relu)(x)
    outputs_reg = tf.keras.layers.Dense(1, activation="linear", name='diff')(x)
    outputs_class = tf.keras.layers.Dense(3, activation='softmax', name='win')(outputs_reg)

    model = tf.keras.Model(inputs=[inputs_1, inputs_2], outputs=[outputs_reg, outputs_class])
    # print(model.summary())

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=20000,
        decay_rate=0.9)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)

    tf.keras.losses.sign_penalty = sign_penalty

    model.compile(loss={'diff': sign_penalty, 'win': tf.keras.losses.CategoricalCrossentropy()},
                  optimizer=optimizer,
                  metrics={'diff': 'mae', "win": 'categorical_crossentropy'})

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

def create_feature_names(line_definitions, features_to_use, features_to_extract):
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

            if line_key == 'def':
                for def_feat in features_to_extract['def']:
                    features_to_use.append(f'{team}_{def_feat}_{line_key}')

            if line_key == 'att':
                for att_feat in features_to_extract['att']:
                    features_to_use.append(f'{team}_{att_feat}_{line_key}')

    return features_to_use


if __name__ == '__main__':
    fixture_overview_df = pd.read_csv('prepped_data_set.csv')

    # add predictable
    fixture_overview_df['team_point_diff'] = fixture_overview_df['HomeTeamScore'] - fixture_overview_df['AwayTeamScore']
    fixture_overview_df.loc[fixture_overview_df['team_point_diff'] > 0, 'team_victory'] = 2
    fixture_overview_df.loc[fixture_overview_df['team_point_diff'] == 0, 'team_victory'] = 1
    fixture_overview_df.loc[fixture_overview_df['team_point_diff'] < 0, 'team_victory'] = 0
    fixture_overview_df['team_victory'] = fixture_overview_df['team_victory'].astype(int)

    # cap large differences between matches to prevent overfitting on outliers
    fixture_overview_df.loc[fixture_overview_df['team_point_diff'] > 4, 'team_point_diff'] = 4
    fixture_overview_df.loc[fixture_overview_df['team_point_diff'] < -4, 'team_point_diff'] = -4

    # add home team advantage feature
    fixture_overview_df['home_team_advantage'] = 1
    fixture_overview_df.loc[fixture_overview_df['national_game'] == 1, 'home_team_advantage'] = 0
    fixture_overview_df['away_team_advantage'] = 0

    # the features we want to use for out model
    features_to_extract = {
        'general': ['value_eur', 'potential', 'overall', 'work_rate', 'international_reputation', 'age', 'height_cm',
                    'weight_kg', 'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic',
                    'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots',
                    'movement_acceleration', 'movement_sprint_speed', 'movement_agility', 'movement_reactions',
                    'movement_balance',
                    'skill_dribbling', 'skill_curve', 'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control',
                    'mentality_aggression', 'mentality_interceptions', 'mentality_positioning', 'mentality_vision',
                    'mentality_penalties', 'mentality_composure'],
        'goal': ['goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking',
                 'goalkeeping_positioning', 'goalkeeping_reflexes', 'goalkeeping_speed'],
        'def': ['defending_marking_awareness', 'defending_standing_tackle', 'defending_sliding_tackle'],
        'att': ['attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy', 'attacking_short_passing',
                'attacking_volleys']
    }

    # the different positions and teams
    line_definitions = {"goal": ['GK'], "def": ['B'], "mid": ['M'], "att": ['CAM', 'CF', 'ST']}
    teams = ['home_team', 'away_team']

    # list for the features and the base of features that are not player dependent
    features_to_use = ['national_game', 'home_team_advantage', 'away_team_advantage']
    features_to_use = create_feature_names(line_definitions, features_to_use, features_to_extract)
    features_to_use_home = [feature for feature in features_to_use if 'home_team' in feature]
    features_to_use_away = [feature for feature in features_to_use if 'away_team' in feature]

    # preprocess the data
    scaled_X_home = preprocessing.RobustScaler().fit_transform(fixture_overview_df[features_to_use_home].values)
    scaled_X_away = preprocessing.RobustScaler().fit_transform(fixture_overview_df[features_to_use_away].values)

    one_hot_y_train = np.zeros((fixture_overview_df['team_victory'].values.size, fixture_overview_df['team_victory'].values.max() + 1))
    one_hot_y_train[np.arange(fixture_overview_df['team_victory'].values.size), fixture_overview_df['team_victory'].values] = 1

    # split data in to train, validation and test
    X_train_home, X_test_home, X_train_away, X_test_away, y_train_reg, y_test_reg, y_train_class, y_test_class = train_test_split(
        scaled_X_home, scaled_X_away, fixture_overview_df['team_point_diff'].values, one_hot_y_train,
        test_size=0.2, random_state=0, shuffle=True)

    X_train_home, X_val_home, X_train_away, X_val_away, y_train_reg, y_val_reg, y_train_class, y_val_class = \
        train_test_split(X_train_home, X_train_away, y_train_reg, y_train_class, test_size=0.2, random_state=0, shuffle=True)

    print(f'train length: {len(X_train_home)}')
    print(f'val length: {len(X_val_home)}')
    print(f'test length: {len(X_test_home)}')

    # create the model
    model = create_tensorflow_model_regressor(features_to_use_home)

    # fit the model and visualise the results
    history = fit(model, X_train_home, X_train_away, y_train_reg, y_train_class, X_val_home, X_val_away, y_val_reg, y_val_class)
    plot_loss(history)

    # predict for the training and the test data and visualise the results
    test_pred = model.predict([X_test_home, X_test_away], verbose=0)
    train_pred = model.predict([X_train_home, X_train_away], verbose=0)

    test_pred_reg = test_pred[0].reshape((len(test_pred[0]),))
    test_pred_class = np.argmax(test_pred[1], axis=1)

    train_pred_reg = train_pred[0].reshape((len(train_pred[0]),))
    train_pred_class = np.argmax(train_pred[1], axis=1)

    plot_predictions(y_test_reg, test_pred_reg, label='test')
    plot_predictions(y_train_reg, train_pred_reg, label='train')

    # calculate the test and training scores
    print(f'Test regression score: {mean_absolute_error(y_test_reg, test_pred_reg)}')
    print(f'Train regression score: {mean_absolute_error(y_train_reg, train_pred_reg)}')

    print("Test confusion matrix \n", confusion_matrix(np.argmax(y_test_class, axis=1), test_pred_class))
    print("Train confusion matrix \n", confusion_matrix(np.argmax(y_train_class, axis=1), train_pred_class))

    print(f'Test score: {accuracy_score(np.argmax(y_test_class, axis=1), test_pred_class)}')
    print(f'Train score: {accuracy_score(np.argmax(y_train_class, axis=1), train_pred_class)}')
