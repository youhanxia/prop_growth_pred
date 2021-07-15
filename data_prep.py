import os
import json
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import pickle


used_keys = [
    'DateYear',
    'DateMonth',
    'SA22016ID',
    'Auction_Activity_AuctionClearanceRate',
    'Auction_Activity_AuctionListedCount',
    'Auction_Activity_AuctionNotSoldCount',
    'Auction_Activity_AuctionPostponedCount',
    'Auction_Activity_AuctionSoldAfterCount',
    'Auction_Activity_AuctionSoldPriorCount',
    'Auction_Activity_AuctionSoldUnderHammerCount',
    'Auction_Activity_AuctionTotalReported',
    'Auction_Activity_AuctionTotalSold',
    'Auction_Activity_AuctionWithdrawnCount',
    'For_Rent_Home_Lease_AveragePrice',
    'For_Rent_Home_Lease_DetailedPosition05Price',
    'For_Rent_Home_Lease_DetailedPosition10Price',
    'For_Rent_Home_Lease_DetailedPosition15Price',
    'For_Rent_Home_Lease_DetailedPosition20Price',
    'For_Rent_Home_Lease_DetailedPosition25Price',
    'For_Rent_Home_Lease_DetailedPosition30Price',
    'For_Rent_Home_Lease_DetailedPosition35Price',
    'For_Rent_Home_Lease_DetailedPosition40Price',
    'For_Rent_Home_Lease_DetailedPosition45Price',
    'For_Rent_Home_Lease_DetailedPosition50Price',
    'For_Rent_Home_Lease_DetailedPosition55Price',
    'For_Rent_Home_Lease_DetailedPosition60Price',
    'For_Rent_Home_Lease_DetailedPosition65Price',
    'For_Rent_Home_Lease_DetailedPosition70Price',
    'For_Rent_Home_Lease_DetailedPosition75Price',
    'For_Rent_Home_Lease_DetailedPosition80Price',
    'For_Rent_Home_Lease_DetailedPosition85Price',
    'For_Rent_Home_Lease_DetailedPosition90Price',
    'For_Rent_Home_Lease_DetailedPosition95Price',
    'For_Rent_Home_Lease_DetailedPriceCalculationRecordCount',
    'For_Rent_Home_Lease_EventCount',
    'For_Rent_Home_Lease_GeometricMeanPrice',
    'For_Rent_Home_Lease_MaximumPrice',
    'For_Rent_Home_Lease_MedianPrice',
    'For_Rent_Home_Lease_MinimumPrice',
    'For_Rent_Home_Lease_Position25Price',
    'For_Rent_Home_Lease_Position75Price',
    'For_Rent_Home_Lease_PriceCalculationRecordCount',
    'For_Rent_Home_Lease_StandardDeviationPrice',
    'For_Rent_Home_Lease_TotalPrice',
    'For_Sale_Both_Auction_Private_Treaty_AveragePrice',
    'For_Sale_Both_Auction_Private_Treaty_DetailedPosition05Price',
    'For_Sale_Both_Auction_Private_Treaty_DetailedPosition10Price',
    'For_Sale_Both_Auction_Private_Treaty_DetailedPosition15Price',
    'For_Sale_Both_Auction_Private_Treaty_DetailedPosition20Price',
    'For_Sale_Both_Auction_Private_Treaty_DetailedPosition25Price',
    'For_Sale_Both_Auction_Private_Treaty_DetailedPosition30Price',
    'For_Sale_Both_Auction_Private_Treaty_DetailedPosition35Price',
    'For_Sale_Both_Auction_Private_Treaty_DetailedPosition40Price',
    'For_Sale_Both_Auction_Private_Treaty_DetailedPosition45Price',
    'For_Sale_Both_Auction_Private_Treaty_DetailedPosition50Price',
    'For_Sale_Both_Auction_Private_Treaty_DetailedPosition55Price',
    'For_Sale_Both_Auction_Private_Treaty_DetailedPosition60Price',
    'For_Sale_Both_Auction_Private_Treaty_DetailedPosition65Price',
    'For_Sale_Both_Auction_Private_Treaty_DetailedPosition70Price',
    'For_Sale_Both_Auction_Private_Treaty_DetailedPosition75Price',
    'For_Sale_Both_Auction_Private_Treaty_DetailedPosition80Price',
    'For_Sale_Both_Auction_Private_Treaty_DetailedPosition85Price',
    'For_Sale_Both_Auction_Private_Treaty_DetailedPosition90Price',
    'For_Sale_Both_Auction_Private_Treaty_DetailedPosition95Price',
    'For_Sale_Both_Auction_Private_Treaty_DetailedPriceCalcRecCount',
    'For_Sale_Both_Auction_Private_Treaty_EventCount',
    'For_Sale_Both_Auction_Private_Treaty_GeometricMeanPrice',
    'For_Sale_Both_Auction_Private_Treaty_MaximumPrice',
    'For_Sale_Both_Auction_Private_Treaty_MedianPrice',
    'For_Sale_Both_Auction_Private_Treaty_MinimumPrice',
    'For_Sale_Both_Auction_Private_Treaty_Position25Price',
    'For_Sale_Both_Auction_Private_Treaty_Position75Price',
    'For_Sale_Both_Auction_Private_Treaty_PriceCalcRecordCount',
    'For_Sale_Both_Auction_Private_Treaty_StandardDeviationPrice',
    'For_Sale_Both_Auction_Private_Treaty_TotalPrice',
    'Gross_Rental_Yield',
    'PropertyCategorisation',
    'Sold_Both_Auction_Private_Treaty_AveragePrice',
    'Sold_Both_Auction_Private_Treaty_DetailedPosition05Price',
    'Sold_Both_Auction_Private_Treaty_DetailedPosition10Price',
    'Sold_Both_Auction_Private_Treaty_DetailedPosition15Price',
    'Sold_Both_Auction_Private_Treaty_DetailedPosition20Price',
    'Sold_Both_Auction_Private_Treaty_DetailedPosition25Price',
    'Sold_Both_Auction_Private_Treaty_DetailedPosition30Price',
    'Sold_Both_Auction_Private_Treaty_DetailedPosition35Price',
    'Sold_Both_Auction_Private_Treaty_DetailedPosition40Price',
    'Sold_Both_Auction_Private_Treaty_DetailedPosition45Price',
    'Sold_Both_Auction_Private_Treaty_DetailedPosition50Price',
    'Sold_Both_Auction_Private_Treaty_DetailedPosition55Price',
    'Sold_Both_Auction_Private_Treaty_DetailedPosition60Price',
    'Sold_Both_Auction_Private_Treaty_DetailedPosition65Price',
    'Sold_Both_Auction_Private_Treaty_DetailedPosition70Price',
    'Sold_Both_Auction_Private_Treaty_DetailedPosition75Price',
    'Sold_Both_Auction_Private_Treaty_DetailedPosition80Price',
    'Sold_Both_Auction_Private_Treaty_DetailedPosition85Price',
    'Sold_Both_Auction_Private_Treaty_DetailedPosition90Price',
    'Sold_Both_Auction_Private_Treaty_DetailedPosition95Price',
    'Sold_Both_Auction_Private_Treaty_DetailedPriceCalcRecordCount',
    'Sold_Both_Auction_Private_Treaty_EventCount',
    'Sold_Both_Auction_Private_Treaty_GeometricMeanPrice',
    'Sold_Both_Auction_Private_Treaty_MaximumPrice',
    'Sold_Both_Auction_Private_Treaty_MedianPrice',
    'Sold_Both_Auction_Private_Treaty_MinimumPrice',
    'Sold_Both_Auction_Private_Treaty_Position25Price',
    'Sold_Both_Auction_Private_Treaty_Position75Price',
    'Sold_Both_Auction_Private_Treaty_PriceCalculationRecordCount',
    'Sold_Both_Auction_Private_Treaty_StandardDeviationPrice',
    'Sold_Both_Auction_Private_Treaty_TotalPrice',
    'Sold_PrivateTreaty_AverageDaysOnMarket',
    'Sold_Private_Treaty_AverageDiscount',
    'Sold_Private_Treaty_DaysOnMarketRecordCount',
    'Sold_Private_Treaty_DiscountRecordCount',
    'State'
]


def consolidate(data_dir):
    # get all monthly files
    fns = os.listdir(data_dir)
    fns = list(filter(lambda x: x.endswith('.json') and not x.startswith('meta'), fns))

    count = 0
    for fn in fns:
        print('\r', count, end='')
        with open(os.path.join(data_dir, fn)) as f:
            data = json.load(f)
        data_body = data['features']
        if not count:
            with open('_'.join(data_dir.lower().split()) + '.csv', 'w') as f:
                print(','.join(map(str, data_body[0]['properties'].keys())), file=f)
        for item in data['features']:
            with open('_'.join(data_dir.lower().split()) + '.csv', 'a') as f:
                print(','.join(map(str, item['properties'].values())), file=f)
        count += 1


def data_cleaning(fn):
    # Current trade data predicts growth of median in 3 years
    df = pd.read_csv(fn)
    short_keys = filter(lambda x: 'Detailed' not in x and 'Count' not in x and 'TotalPrice' not in x, used_keys)
    df = df[short_keys]

    # labeling
    df.set_index(keys=['DateYear',
                       'DateMonth',
                       'SA22016ID',
                       'PropertyCategorisation'], inplace=True)

    mpg_lst = []
    historical_features = []
    i = 0
    for idx, row in df.iterrows():
        print('\r', i, end='')
        i += 1
        idx_in_3y = (idx[0] + 3, idx[1], idx[2], idx[3])
        try:
            mp = float(row.Sold_Both_Auction_Private_Treaty_MedianPrice)
            growth = float(df.loc[idx_in_3y].Sold_Both_Auction_Private_Treaty_MedianPrice) / mp - 1
        except:
            growth = np.nan
        mpg_lst.append(growth)
        temp_lst = []
        for j in [1, 2, 3]:
            if idx[1] - j < 1:
                idx__past_jm = (idx[0] - 1, idx[1] + 12 - j, idx[2], idx[3])
            else:
                idx__past_jm = (idx[0], idx[1] - j, idx[2], idx[3])
            idx__past_jy = (idx[0] - j, idx[1], idx[2], idx[3])
            try:
                temp_lst.append(df.loc[idx__past_jm].Sold_Both_Auction_Private_Treaty_MedianPrice)
            except:
                temp_lst.append(np.nan)
            try:
                temp_lst.append(df.loc[idx__past_jy].Sold_Both_Auction_Private_Treaty_MedianPrice)
            except:
                temp_lst.append(np.nan)
        historical_features.append(temp_lst)
    df['MedianPrice_Growth'] = mpg_lst


    df.reset_index(inplace=True)

    '''
    edit|add features here
    '''

    # one_hot encoding
    prop_cat_1h = pd.get_dummies(df['PropertyCategorisation'])
    state_1h = pd.get_dummies(df['State'])

    df.drop(columns=['PropertyCategorisation', 'State'], inplace=True)

    df['House'] = prop_cat_1h['House']
    df = df.join(state_1h)

    # add historical median prices
    col_names = [
        'MedianPrice_1_month_ago',
        'MedianPrice_1_year_ago',
        'MedianPrice_2_month_ago',
        'MedianPrice_2_year_ago',
        'MedianPrice_3_month_ago',
        'MedianPrice_3_year_ago',
    ]
    df.join(pd.DataFrame(np.array(historical_features), columns=col_names))

    '''
    finish feature engineering
    '''

    # put labels back to the last column
    lbl_cols = ['MedianPrice_Growth', 'Median_exceed_Avg']
    df = df[[c for c in df if c not in lbl_cols] + lbl_cols]

    # casting
    df = df.apply(lambda x: pd.to_numeric(x, errors='coerce'))

    # separate latest unlabeled data
    df_latest = df[df['DateYear'] > 2017]
    df.dropna(subset=['MedianPrice_Growth'], inplace=True)

    df.to_csv('apm_sa2_short_labeled.csv', index=False)
    df_latest.to_csv('apm_sa2_short_labeled_latest.csv', index=False)


def clf_lbl_patch(fn):
    df = pd.read_csv(fn)

    if 'latest' in fn:
        df['Median_exceed_Avg'] = np.nan
    else:
        mean_mpg = df.groupby(['DateYear',
                               'DateMonth',
                               'House']).mean()['MedianPrice_Growth']

        df['Median_exceed_Avg'] = df.apply(
            lambda x: x['MedianPrice_Growth'] is not np.nan and x['MedianPrice_Growth'] > mean_mpg.loc[
                x['DateYear'], x['DateMonth'], x['House']], axis=1).astype(int)

    df.to_csv(fn, index=False)


def data_prep(fn, fn_l):
    df = pd.read_csv(fn)
    df_l = pd.read_csv(fn_l)

    X = df.values
    X_l = df_l.values

    # split labels
    Y = X[:, -2:]
    X = X[:, 3:-2]

    Z_l = X_l[:, :3]
    Z__ = X_l[:, -11]
    Z_l = np.concatenate((Z_l, np.reshape(Z__, (-1, 1))), axis=1)
    X_l = X_l[:, 3:-2]

    # estimate missing values
    imp = SimpleImputer()
    X = imp.fit_transform(X)
    X_l = imp.transform(X_l)

    # scale data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_l = scaler.transform(X_l)

    # split data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    obj_c = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': Y_train[:, -1],
        'y_test': Y_test[:, -1],
        'X_latest': X_l,
        'Z_latest': Z_l
    }
    with open('apm_sa2_clf.pkl', 'wb') as f:
        pickle.dump(obj_c, f)

    obj_r = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': Y_train[:, -2],
        'y_test': Y_test[:, -2],
        'X_latest': X_l,
        'Z_latest': Z_l
    }
    with open('apm_sa2_rgs.pkl', 'wb') as f:
        pickle.dump(obj_r, f)


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    # consolidate('APM SA2')
    # data_cleaning('apm_sa2.csv')
    # clf_lbl_patch('apm_sa2_short_labeled.csv')
    # clf_lbl_patch('apm_sa2_short_labeled_latest.csv')
    data_prep('apm_sa2_short_labeled.csv', 'apm_sa2_short_labeled_latest.csv')

    os.system('say "Mission complete."')
