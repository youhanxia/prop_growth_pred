import os
import pickle
import numpy as np
from types import SimpleNamespace

from sklearn.svm import SVR, SVC
from sklearn.metrics import *
import xgboost as xgb

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def trainer(fn, par='xgb', pred=False):
    # read data
    with open(fn, 'rb') as f:
        d = pickle.load(f)
    # keys: 'X_train', 'X_test', 'y_train', 'y_test', 'X_latest', 'Z_latest'
    data = SimpleNamespace(**d)

    # establish model
    if par == 'xgb':
        if 'clf' in fn:
            model = xgb.XGBClassifier(objective='reg:squarederror')
        else:
            model = xgb.XGBRegressor(objective='reg:squarederror')
    elif par == 'svm':
        if 'clf' in fn:
            model = SVC(kernel='linear', verbose=True)
        else:
            model = SVR(kernel='linear', verbose=True)
    elif par == 'dl':
        model = struct_nn(data.X_train.shape[1], 'clf' in fn)
        if 'clf' in fn:
            model.compile(loss=keras.losses.binary_crossentropy, metrics=['accuracy'])
        else:
            model.compile(loss=keras.losses.mse)
    else:
        return

    # train
    if par != 'dl':
        model.fit(data.X_train, data.y_train)
    else:
        history = model.fit(data.X_train, data.y_train, batch_size=64, epochs=2, validation_split=0.2)

    # eval
    if pred:
        y_pred = model.predict(data.X_latest)

        with open(fn[:-4] + '_pred.pkl', 'wb') as f:
            pickle.dump(y_pred, f)
    else:
        y_pred = model.predict(data.X_test)

        if 'clf' in fn:
            y_pred = np.array(y_pred >= 0.5, dtype=int)
            print('accuracy_score', accuracy_score(data.y_test, y_pred))
            print('balanced_accuracy_score', balanced_accuracy_score(data.y_test, y_pred))
            print('precision_score', precision_score(data.y_test, y_pred))
            print('recall_score', recall_score(data.y_test, y_pred))
            print('f1_score', f1_score(data.y_test, y_pred))
        else:
            print('MSE', mean_squared_error(data.y_test, y_pred))

        with open(fn[:-4] + '_vali.pkl', 'wb') as f:
            pickle.dump(y_pred, f)

    pass


def struct_nn(d, clf=True):
    inputs = keras.Input(shape=(d,))
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    if clf:
        outputs = layers.Dense(1, activation='sigmoid')(x)
    else:
        outputs = layers.Dense(1)(x)

    return keras.Model(inputs=inputs, outputs=outputs, name='model')


if __name__ == '__main__':
    # trainer('apm_sa2_rgs.pkl', par='svm')
    # trainer('apm_sa2_clf.pkl', par='svm')

    # trainer('apm_sa2_rgs.pkl', par='xgb')
    # trainer('apm_sa2_clf.pkl', par='xgb')

    # trainer('apm_sa2_rgs.pkl', par='dl', pred=True)
    # trainer('apm_sa2_clf.pkl', par='dl', pred=True)

    trainer('apm_sa2_rgs.pkl', par='dl')
    trainer('apm_sa2_clf.pkl', par='dl')

    os.system('say "Mission complete."')
