# Explainable Horizon-Adaptive Stacked Learning for U.S. Inflation Forecasting with SHAP
# Baker, M.R., Buyrukoğlu, S., & Jihad, K.H.

import pandas as pd
import numpy as np
import os

CACHE_DIR = 'x:/abc/Data_Cache'  # <-- Update this path to your local data folder


def load_series(name):
    path = os.path.join(CACHE_DIR, f"{name}.csv")
    if "_vintage" in name:
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'])
        df['realtime_start'] = pd.to_datetime(df['realtime_start'])
        s = df.sort_values('realtime_start').groupby('date')['value'].last()
    else:
        df = pd.read_csv(path, header=None, index_col=0)
        df.index = pd.to_datetime(df.index)
        s = df.iloc[:, 0]
    return s


def load_macro_dataset(use_realtime=True, publication_delays=True, **kwargs):
    series = {
        'PCE':        load_series('PCE_vintage'),
        'UNRATE':     load_series('UNRATE_vintage'),
        'INDPRO':     load_series('INDPRO_vintage'),
        'M2SL':       load_series('M2SL_vintage'),
        'PolicyRate': load_series('PolicyRate_market'),
        'WTI':        load_series('WTI_market'),
        'GS10':       load_series('GS10_market'),
        'TB3MS':      load_series('TB3MS_market'),
    }
    df = pd.DataFrame(series)
    df = df.resample('ME').last()
    df = df.ffill().dropna()
    df['Inflation_Rate'] = df['PCE'].pct_change(12) * 100
    df['TermSpread']     = df['GS10'] - df['TB3MS']
    df['UNRATE_Gap']     = df['UNRATE'] - df['UNRATE'].rolling(120).mean()
    df = df.dropna()
    print(f"Data loaded: {df.shape[0]} observations ({df.index[0].date()} to {df.index[-1].date()})")
    return df
