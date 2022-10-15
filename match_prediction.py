import numpy as np
import xgboost
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, ShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from sklearn import preprocessing

MAX_EPOCHS = 100


def compile_and_fit(model, X_train, y_train, X_val, y_val):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=10,
                                                      mode='min')

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=MAX_EPOCHS,
                        validation_data=(X_val, y_val),
                        callbacks=[early_stopping],
                        batch_size=32)
    return history


def create_tensorflow_model(num_features):
    inputs = tf.keras.Input(shape=(len(num_features),))
    x = tf.keras.layers.Dense(100, activation=tf.nn.relu)(inputs)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(100, activation=tf.nn.relu)(x)
    outputs = tf.keras.layers.Dense(3, activation=tf.nn.softmax)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    # print(model.summary())

    return model


if __name__ == '__main__':
    fixture_overview_df = pd.read_csv('prepped_data_set.csv')

    # prep prediction column
    fixture_overview_df['team_victory'] = 0
    fixture_overview_df.loc[fixture_overview_df['HomeTeamScore'] > fixture_overview_df['AwayTeamScore'],
                            'team_victory'] = 0
    fixture_overview_df.loc[fixture_overview_df['HomeTeamScore'] == fixture_overview_df['AwayTeamScore'],
                            'team_victory'] = 1
    fixture_overview_df.loc[fixture_overview_df['HomeTeamScore'] < fixture_overview_df['AwayTeamScore'],
                            'team_victory'] = 2

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
        preprocessing.normalize(fixture_overview_df[features_to_use].values),
        fixture_overview_df['team_victory'].astype(int).values,
        test_size=0.2, random_state=0, shuffle=False)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.2, random_state=0, shuffle=False)

    print(f'train length: {len(X_train)}')
    print(f'val length: {len(X_val)}')
    print(f'test length: {len(X_test)}')

    model = create_tensorflow_model(features_to_use)

    one_hot_y_train = np.zeros((y_train.size, y_train.max() + 1))
    one_hot_y_train[np.arange(y_train.size), y_train] = 1

    one_hot_y_val = np.zeros((y_val.size, y_val.max() + 1))
    one_hot_y_val[np.arange(y_val.size), y_val] = 1

    one_hot_y_test = np.zeros((y_test.size, y_test.max() + 1))
    one_hot_y_test[np.arange(y_test.size), y_test] = 1

    history = compile_and_fit(model, X_train, one_hot_y_train, X_val, one_hot_y_val)

    print(model.evaluate(X_test, y_test))

    # predict for the training and the test data
    test_pred = model.predict(X_test, verbose=0)
    train_pred = model.predict(X_train, verbose=0)

    # convert predictions to labels
    test_pred = np.argmax(test_pred, axis=1)
    train_pred = np.argmax(train_pred, axis=1)

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

    print(confusion_matrix(y_test, test_pred))

    print(f'Test score: {accuracy_score(y_test, test_pred)}')
    print(f'Train score: {accuracy_score(y_train, train_pred)}')
