import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from sklearn import preprocessing

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


def fit(model, X_train, y_train, X_val, y_val):
    """Fit the model"""
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=200,
                                                      mode='min')

    history = model.fit(X_train, y_train, epochs=MAX_EPOCHS,
                        validation_data=(X_val, y_val),
                        callbacks=[early_stopping],
                        batch_size=32)
    return history



def create_tensorflow_model_regressor(num_features):
    """Compose tensorflow model."""
    inputs = tf.keras.Input(shape=(len(num_features),))
    x = tf.keras.layers.Dense(1024, activation=tf.nn.relu)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x_skip = tf.keras.layers.Dropout(0.2)(x)

    # blcok with skip connection
    x = tf.keras.layers.Dense(512, activation=tf.nn.relu)(x_skip)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(1024, activation=tf.nn.relu)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.Activation('relu')(x)

    # block with skip connection
    x_skip = tf.keras.layers.Dense(256, activation=tf.nn.relu)(x)
    x = tf.keras.layers.Dense(512, activation=tf.nn.relu)(x_skip)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(256, activation=tf.nn.relu)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.Activation('relu')(x)

    # block with skip connection
    x_skip = tf.keras.layers.Dense(256, activation=tf.nn.relu)(x)
    x = tf.keras.layers.Dense(512, activation=tf.nn.relu)(x_skip)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(256, activation=tf.nn.relu)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Dense(64, activation=tf.nn.relu)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(32, activation=tf.nn.relu)(x)
    outputs = tf.keras.layers.Dense(1, activation="linear")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    # print(model.summary())

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


if __name__ == '__main__':
    fixture_overview_df = pd.read_csv('../prepped_data_sources/prepped_data_set.csv')

    # add predictable
    fixture_overview_df['team_victory'] = fixture_overview_df['HomeTeamScore'] - fixture_overview_df['AwayTeamScore']

    # cap large differences between matches to prevent overfitting on outliers
    fixture_overview_df.loc[fixture_overview_df['team_victory'] > 4, 'team_victory'] = 4
    fixture_overview_df.loc[fixture_overview_df['team_victory'] < -4, 'team_victory'] = -4

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
    features_to_use = create_feature_names(line_definitions, features_to_use, features_to_extract)

    # preprocess the data
    scaled_X = preprocessing.RobustScaler().fit_transform(fixture_overview_df[features_to_use].values)

    # split data in to train, validation and test
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, fixture_overview_df['team_victory'].values,
        test_size=0.2, random_state=0, shuffle=True)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.2, random_state=0, shuffle=True)

    print(f'train length: {len(X_train)}')
    print(f'val length: {len(X_val)}')
    print(f'test length: {len(X_test)}')

    # create the model
    model = create_tensorflow_model_regressor(features_to_use)

    # fit the model and visualise the results
    history = fit(model, X_train, y_train, X_val, y_val)
    plot_loss(history)

    # predict for the training and the test data and visualise the results
    test_pred = model.predict(X_test, verbose=0)
    train_pred = model.predict(X_train, verbose=0)
    plot_predictions(y_test, test_pred, label='test')
    plot_predictions(y_train, train_pred, label='train')

    # calculate the test and training scores
    print(f'Test score: {mean_absolute_error(y_test, test_pred)}')
    print(f'Train score: {mean_absolute_error(y_train, train_pred)}')

    model.save(f'ResNet_model_{mean_absolute_error(y_test, test_pred)}.csv')
