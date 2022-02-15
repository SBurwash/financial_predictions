import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle

KEY = 'EEADPN3Z20W94ARM'
functions = {
    'daily' : 'TIME_SERIES_DAILY',
    'intraday' : 'TIME_SERIES_INTRADAY',
    'daily_adjusted' : 'TIME_SERIES_DAILY_ADJUSTED',
    'weekly' : 'TIME_SERIES_WEEKLY',
    'weekly_adjusted' : 'TIME_SERIES_WEEKLY_ADJUSTED'


}

CATEGORY = '4. close'
TRAIN_PERCENTAGE = 0.8

def get_data(function='TIME_SERIES_INTRADAY', symbol='IBM', interval='5min', outputsize='full'):
    url = f'https://www.alphavantage.co/query?function={function}&symbol={symbol}&outputsize={outputsize}&interval={interval}&apikey={KEY}'
    r = requests.get(url)
    d = r.json()
    keys = list(d.keys())
    metadata = d[keys[0]]

    data = d[keys[1]]
    data = pd.DataFrame(data).T
    data.index = pd.to_datetime(data.index)
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.sort_index(ascending=True)

    return metadata, data



def prepare_data(data, train_perc=0.8, freq='D', drop=None, category=None):

    d = data
    d = d.asfreq(freq)
    d = d.interpolate()
    if drop is not None:
        d = d.drop(labels=[drop], axis=1)

    fh = d[int(d.shape[0]*train_perc) : ]
    train = d[: int(d.shape[0]*train_perc)]

    if category is not None:
        fh = fh[category]
        train = train[category]

    return train, fh





def train_arma(train, fh, model_name='arma.pkl'):

    model = SARIMAX(train, order = (1, 0, 1))
    model = model.fit()

    y_pred = model.get_forecast(len(fh.index))
    y_pred_df = y_pred.conf_int(alpha = 0.05)
    y_pred_df["Predictions"] = model.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
    y_pred_df.index = fh.index
    y_pred_out = y_pred_df["Predictions"]

    pickle.dump(model, open(model_name, 'wb'))

    return y_pred_out, model


metadata, data = get_data(function = functions['daily'])
train, fh = prepare_data(data=data, train_perc=TRAIN_PERCENTAGE, freq='D', drop='5. volume', category=CATEGORY)

y_pred, model = train_arma(train, fh, 'arma.pkl')