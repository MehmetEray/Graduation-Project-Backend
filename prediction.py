import numpy as np
import pandas as pd
import json

# %matplotlib inline
# from plotly.graph_objs import *
import statsmodels.api as sm
import warnings
import yfinance as yf

warnings.filterwarnings('ignore')
import seaborn as sns
import itertools


def forecastwithoption(compName, day):
    comp = yf.Ticker(compName)
    # get historical market data
    df = comp.history(period="max")
    # Exploratory Data Analysis:
    df.isnull().sum()
    print(df.shape)


    # transform to datetime object here..
    df.index = pd.to_datetime(df.index)

    df_groupby = df.groupby(['Date'])['Close'].mean()
    df_groupby.sort_index(inplace=True)

    y = df_groupby
    y = y.tail(110)
    print(y)
    lastDayOfDf = y.index.max().strftime("%m/%d/%Y")
    firstDayOfDf = y.index.min().strftime("%m/%d/%Y")
    print("first day: " + firstDayOfDf + ", Last day: " + lastDayOfDf)

    """
    onemonthlater = pd.date_range(y.index.max(), periods=30, freq='1D')
    threemonthlater = pd.date_range(y.index.max(), periods=90, freq='1D')
    sixmonthlater = pd.date_range(y.index.max(), periods=180, freq='1D')
    """


    # ARIMA stands for Auto Regression Integrated Moving Average.
    # It is specified by three ordered parameters (p,d,q). Where:
    # p is the order of the autoregressive model(number of time lags)
    # d is the degree of differencing (number of times the data have had past values subtracted)
    # q is the order of moving average model. Before building an ARIMA model,
    # we have to make sure our data is stationary.

    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    print('Examples of parameter for SARIMA...')
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y,
                                                order=param,
                                                seasonal_order=param_seasonal)
                results = mod.fit(max_iter=50, method='powell')
                print('SARIMA{},{} - AIC:{}'.format(param, param_seasonal, results.aic))
            except Exception as ex:
                print('Exception: ', ex)

    """mod = sm.tsa.statespace.SARIMAX(y, order=param, seasonal_order=param_seasonal,
                                                enforce_stationarity=False, enforce_invertibility=False)"""
    print(results.summary().tables[1])


    pred = results.get_prediction(start=pd.to_datetime(firstDayOfDf), end=pd.to_datetime(lastDayOfDf), dynamic=False)
    pred_ci = pred.conf_int()


    print(pred_ci)

    pred_uc = results.get_forecast(steps=90)
    pred_ci = pred_uc.conf_int()




    forecast = pred_uc.predicted_mean
    print(forecast.head(day))

    jsonfiles = json.loads(forecast.to_json(orient='records'))
    return jsonfiles


def forecastwithuploadcsv(csv, day):
    df = csv
    df.index = pd.to_datetime(df.index)

    df_groupby = df.groupby(['Date'])['Close'].mean()


    df_groupby.sort_index(inplace=True)

    y = df_groupby
    y = y.tail(110)
    print(y)
    lastDayOfDf = y.index.max().strftime("%m/%d/%Y")
    firstDayOfDf = y.index.min().strftime("%m/%d/%Y")
    print("first day: " + firstDayOfDf + ", Last day: " + lastDayOfDf)

    """
    onemonthlater = pd.date_range(y.index.max(), periods=30, freq='1D')
    threemonthlater = pd.date_range(y.index.max(), periods=90, freq='1D')
    sixmonthlater = pd.date_range(y.index.max(), periods=180, freq='1D')
    """



    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    # print('Examples of parameter for SARIMA...')
    # print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
    # print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
    # print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
    # print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y, order=param, seasonal_order=param_seasonal,
                                                enforce_stationarity=False, enforce_invertibility=False)
                results = mod.fit()
                # print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            except:
                continue

    mod = sm.tsa.statespace.SARIMAX(y,
                                    order=(0, 0, 1),
                                    seasonal_order=(1, 1, 1, 12),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit()


    pred = results.get_prediction(start=pd.to_datetime(firstDayOfDf), end=pd.to_datetime(lastDayOfDf), dynamic=False)
    pred_ci = pred.conf_int()



    y_forecasted = pred.predicted_mean
    y_truth = y['2018-06-01':]
    mse = ((y_forecasted - y_truth) ** 2).mean()
    print('The Mean Squared Error is {}'.format(round(mse, 2)))
    print('The Root Mean Squared Error is {}'.format(round(np.sqrt(mse), 2)))

    pred_uc = results.get_forecast(steps=90)
    pred_ci = pred_uc.conf_int()


    y_forecasted = pred.predicted_mean
    y_forecasted.head(day)

    forecast = pred_uc.predicted_mean
    print(forecast.head(day))

    print(pred)
    return y_forecasted.head(day)

# forecastwithoption("MSFT",30)
