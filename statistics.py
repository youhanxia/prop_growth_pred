import os
import pickle
from collections import Counter
import pandas as pd
import numpy as np
from types import SimpleNamespace

from sklearn.metrics import *


def class_analysis(exp, k=10000):
    fns = [exp + '_rgs.pkl', exp + '_clf.pkl']
    top_k = [0] * k
    for fn in fns:
        print(fn)
        # read data
        with open(fn, 'rb') as f:
            d = pickle.load(f)
        # keys: 'X_train', 'X_test', 'y_train', 'y_test', 'X_latest', 'Z_latest'
        data = SimpleNamespace(**d)

        with open(fn[:-4] + '_vali.pkl', 'rb') as f:
            data.y_pred = pickle.load(f)

        val = data.y_train

        if 'clf' in fn:
            counts = Counter(val)
            print('class label counts', counts)
            dummy_acc = counts[0] / (counts[0] + counts[1])
            if dummy_acc < 0.5:
                dummy_acc = 1 - dummy_acc
            print('dummy accuracy', dummy_acc)

            # repeat DSR's rank based metric
            print('success rate at', k, sum(data.y_test[top_k]) / k)
        else:
            v_mean = np.mean(val)
            v_std = np.std(val)
            print('mean', v_mean, 'std', v_std)
            print('dummy MSE', mean_squared_error(val, [v_mean] * len(val)))

            # repeat DSR's rank based metric
            top_k = np.argsort(data.y_pred.T[0])[::-1][:k]

    pass


def pred_interp(exp):
    fn = exp + '_rgs.pkl'
    fnl = exp + '_short_labeled_latest.csv'

    print(fn)
    # read data
    with open(fn, 'rb') as f:
        d = pickle.load(f)
    # keys: 'X_train', 'X_test', 'y_train', 'y_test', 'X_latest', 'Z_latest'
    data = SimpleNamespace(**d)

    with open(fn[:-4] + '_pred.pkl', 'rb') as f:
        data.y_pred = pickle.load(f)

    df = pd.read_csv(fnl, index_col=['DateYear',
                                     'DateMonth',
                                     'SA22016ID',
                                     'House'])

    # compose predicted values for output
    df = df[[
        'For_Sale_Both_Auction_Private_Treaty_MedianPrice',
        'Sold_PrivateTreaty_AverageDaysOnMarket',
        'Sold_Private_Treaty_AverageDiscount',
        'Auction_Activity_AuctionClearanceRate',
        'Auction_Activity_AuctionTotalReported',
        'Auction_Activity_AuctionTotalSold',
        'For_Rent_Home_Lease_MedianPrice'
    ]]

    pred_growth = pd.DataFrame(data.Z_latest.astype(int), columns=['DateYear',
                                                                   'DateMonth',
                                                                   'SA22016ID',
                                                                   'House'])
    pred_growth['Predicted_MedianPrice_Growth'] = data.y_pred.T[0]
    pred_growth.set_index(['DateYear',
                           'DateMonth',
                           'SA22016ID',
                           'House'], inplace=True)
    df = df.join(pred_growth)
    df.sort_values('Predicted_MedianPrice_Growth', ascending=False, inplace=True)

    df.to_csv(exp + '_final_pred.csv')


if __name__ == '__main__':
    # class_analysis('apm_sa2')
    pred_interp('apm_sa2')

    os.system('say "Mission complete."')
