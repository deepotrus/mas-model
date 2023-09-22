import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

#---------------Returns------------------------------------

def get_returns(price_history): # pass prices as df
    return np.log(price_history / price_history.shift(1)).dropna()

    
#---------------Augmented Dickey-Fuller Test---------------
from statsmodels.tsa.stattools import adfuller

def get_aDF_test(price_history): # pass prices as df
    þ = adfuller(price_history)
    statistic = þ[0]
    p_value = þ[1]
    
    return statistic, p_value

#--------------Volatility Clustering-------------------------

def calc_ACF(price_history, lags):
    # Compute log returns and absolute returns
    log_returns = np.log(price_history) - np.log(price_history.shift(1))
    absolute_returns = np.abs(log_returns)

    # Compute autocorrelation functions for log returns and absolute returns
    log_returns_acf = [log_returns.autocorr(lag=lag) for lag in lags]
    absolute_returns_acf = [absolute_returns.autocorr(lag=lag) for lag in lags]
    
    return log_returns_acf, absolute_returns_acf

def plot_ACF(price_history, lags, title): # pass prices as df
    log_returns_acf, absolute_returns_acf = calc_ACF(price_history, lags)
    
    fig, axs = plt.subplots(ncols = 2, figsize = (12, 5))
    fig.suptitle(f'{title}')
    axs[0].stem(lags, log_returns_acf)
    axs[1].stem(lags, absolute_returns_acf)

    axs[0].set(xlabel = 'Lag', ylabel = 'ACF of Log Returns');
    axs[1].set(xlabel = 'Lag', ylabel = 'ACF of Absolute Returns')

    plt.tight_layout()
    plt.show();