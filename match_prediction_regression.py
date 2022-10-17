import numpy as np
import xgboost
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, ShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error
import tensorflow as tf
from sklearn import preprocessing

MAX_EPOCHS = 1000


def sign_penalty(y_true, y_pred):
    penalty = 3.
    loss = tf.where(tf.less(y_true * y_pred, 0),
                    penalty * tf.square(y_true - y_pred),
                    tf.square(y_true - y_pred))

    return tf.reduce_mean(loss, axis=-1)


def compile_and_fit(model, X_train, y_train, X_val, y_val):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=200,
                                                      mode='min')

    history = model.fit(X_train, y_train, epochs=MAX_EPOCHS,
                        validation_data=(X_val, y_val),
                        callbacks=[early_stopping],
                        batch_size=32)
    return history



def create_tensorflow_model_regressor(num_features):
    inputs = tf.keras.Input(shape=(len(num_features),))
    x = tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(16, activation=tf.nn.leaky_relu)(x)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    # print(model.summary())

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=10000,
        decay_rate=0.9)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    tf.keras.losses.sign_penalty = sign_penalty

    model.compile(loss=sign_penalty,
                  optimizer=optimizer,
                  metrics=['mean_absolute_error'])

    return model


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_predictions(true, preds, label=''):


    fix, ax = plt.subplots()
    plt.title(f'{label}')
    plt.scatter(true, preds)

    # create quadrant
    roc_t = 0.0
    roc_v = 0.0
    ax.fill_between([min(np.append(true, preds)), roc_t], min(np.append(true, preds)), roc_v, alpha=0.3, color='#1F98D0')  # blue
    ax.fill_between([roc_t, max(np.append(true, preds))], min(np.append(true, preds)), roc_v, alpha=0.3, color='#DA383D')  # yellow
    ax.fill_between([min(np.append(true, preds)), roc_t], roc_v, max(np.append(true, preds)), alpha=0.3, color='#DA383D')  # orange
    ax.fill_between([roc_t, max(np.append(true, preds))], roc_v, max(np.append(true, preds)), alpha=0.3, color='#1F98D0')  # red

    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    fixture_overview_df = pd.read_csv('prepped_data_set.csv')

    # prep prediction column
    # fixture_overview_df['team_victory'] = 0
    # fixture_overview_df.loc[fixture_overview_df['HomeTeamScore'] > fixture_overview_df['AwayTeamScore'],
    #                         'team_victory'] = 0
    # fixture_overview_df.loc[fixture_overview_df['HomeTeamScore'] == fixture_overview_df['AwayTeamScore'],
    #                         'team_victory'] = 1
    # fixture_overview_df.loc[fixture_overview_df['HomeTeamScore'] < fixture_overview_df['AwayTeamScore'],
    #                         'team_victory'] = 2

    fixture_overview_df['team_victory'] = fixture_overview_df['HomeTeamScore'] - fixture_overview_df['AwayTeamScore']



    features_to_use = ['total_home_team_price', 'total_away_team_price',
                       'total_home_team_potential', 'total_away_team_potential',
                       'total_home_team_overall', 'total_away_team_overall',
                       # 'total_home_team_work_rate', 'total_away_team_work_rate',
                       'total_home_team_international_reputation', 'total_away_team_international_reputation',
                       'total_home_team_age', 'total_away_team_age',
                       'total_home_team_height_cm', 'total_away_team_height_cm',
                       'total_home_team_weight_kg', 'total_away_team_weight_kg',
                       'national_game']

    # print(fixture_overview_df[features_to_use].info())

    # split data in to train, validation and test
    X_train, X_test, y_train, y_test = train_test_split(
        preprocessing.StandardScaler().fit_transform(fixture_overview_df[features_to_use].values),
        fixture_overview_df['team_victory'].values,
        test_size=0.2, random_state=0, shuffle=True)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.2, random_state=0, shuffle=True)

    print(f'train length: {len(X_train)}')
    print(f'val length: {len(X_val)}')
    print(f'test length: {len(X_test)}')

    model = create_tensorflow_model_regressor(features_to_use)

    history = compile_and_fit(model, X_train, y_train, X_val, y_val)

    plot_loss(history)

    print(model.evaluate(X_test, y_test))

    # predict for the training and the test data
    test_pred = model.predict(X_test, verbose=0)
    train_pred = model.predict(X_train, verbose=0)

    plot_predictions(y_test, test_pred, label='test')
    plot_predictions(y_train, train_pred, label='train')

    print(test_pred)
    print(train_pred)

    # param = {
    #     'gamma': 1,
    #     'learning_rate': 0.1,
    #     'max_depth': 1000,
    #     'n_jobs': -1,
    #     'verbosity': 1,
    #     'max_leaves': 2000
    # }
    #
    # print('start to train the model')
    # xgb_cl = xgboost.XGBClassifier(**param)
    # model = xgb_cl.fit(X_train, y_train)
    #
    # test_pred = model.predict(X_test)
    # train_pred = model.predict(X_train)
    #
    # print(test_pred)

    # print("Test confusion matrix \n", confusion_matrix(y_test, test_pred))
    # print("Train confusion matrix \n", confusion_matrix(y_train, train_pred))

    print(f'Test score: {mean_absolute_error(y_test, test_pred)}')
    print(f'Train score: {mean_absolute_error(y_train, train_pred)}')
