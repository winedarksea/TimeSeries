"""
Comparing Time Series Models' Point Forecast Accuracies

We only use Daily data for this example, much of it business day only, 
but this could easily be used for monthly or other frequency data

Incoming data format is a 'long' data format:
    Just three columns expected: date, series_id, value

Missing data here is handled with a fill-forward when necessary. 
For intermittent data, filling with zero may be better

pip install fredapi
"""
forecast_length = 12
frequency = 'MS'  # 'Offset aliases' from https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html
no_negatives = True # True if all forecasts should be positive

import traceback
import numpy as np
import pandas as pd
import datetime
import sys


def add_years(d, years):
    """Return a date that's `years` years after the date (or datetime)
    object `d`. Return the same calendar date (month and day) in the
    destination year, if it exists, otherwise use the following day
    from https://stackoverflow.com/questions/15741618/add-one-year-in-current-date-python 
    """
    try:
        return d.replace(year = d.year + years)
    except Exception:
        return d + (datetime.date(d.year + years, 1, 1) - datetime.date(d.year, 1, 1))

"""
Get sample data from the Federal Reserve API
"""
from fredapi import Fred
fredkey = 'XXXXXXXXXXXXXXXXXXXXXX' # get from https://research.stlouisfed.org/docs/api/fred/
fred = Fred(api_key=fredkey)

seriesNameDict = {'T10Y2Y':'10 Year Treasury Constant Maturity Minus 2 Year Treasury Constant Maturity', 
                  'DGS10': '10 Year Treasury Constant Maturity Rate',
                  'DCOILWTICO':'Crude Oil West Texas Intermediate Cushing Oklahoma', 
                  'SP500': 'S&P 500', 
                  'DEXUSEU': 'US Euro Foreign Exchange Rate',
                  'DEXCHUS': 'China US Foreign Exchange Rate',
                  'DEXCAUS' : 'Canadian to US Dollar Exchange Rate Daily',
                  'VIXCLS': 'CBOE Volatility Index: VIX',  # this is a more irregular series
                  'T10YIE' : '10 Year Breakeven Inflation Rate',
                  'USEPUINDXD': 'Economic Policy Uncertainty Index for United States' # also very irregular
                  }

series_desired = list(seriesNameDict.keys())

fred_timeseries = pd.DataFrame(columns = ['date', 'value', 'series_id','series_name'])

for series in series_desired:
    data = fred.get_series(series)
    try:
        series_name = seriesNameDict[series]
    except Exception:
        series_name = series
    data_df = pd.DataFrame({'date':data.index, 'value':data, 'series_id':series, 'series_name':series_name})
    data_df.reset_index(drop=True, inplace = True)
    fred_timeseries = pd.concat([fred_timeseries, data_df], axis = 0, ignore_index = True)


timeseries_long = fred_timeseries
# timeseries_long = pd.read_csv("timeseriestestsample.csv")
timeseries_long = pd.read_csv("MonthlyWarehouseProductSales.csv")
timeseries_long['series_id'] = timeseries_long['WarehouseID'] + '|' + timeseries_long['ProductID']
timeseries_long['value'] = timeseries_long['UnitSales']
timeseries_long['date'] = pd.to_datetime(timeseries_long['MonthDateKey'], format = '%Y%m%d')
timeseries_long = timeseries_long[['date','series_id','value']]

"""
Process Long Data
"""
# Attempt to convert to datetime format if not already
timeseries_long['date'] = pd.to_datetime(timeseries_long['date'], infer_datetime_format = True)
# drop older data, because too much of a good thing...
time_cutoff = add_years(datetime.datetime.now(), -5)
timeseries_long = timeseries_long[timeseries_long['date'] >= time_cutoff]
# pivot to different shape for some methods
timeseries_seriescols = timeseries_long.pivot_table(values='value', index='date', columns='series_id')
timeseries_seriescols = timeseries_seriescols.sort_index()
# fill missing dates in index
timeseries_seriescols = timeseries_seriescols.asfreq(frequency, fill_value=np.nan)
# remove series with way too many NaNs - probably those of a different frequency, or brand new
timeseries_seriescols = timeseries_seriescols.dropna(axis = 1, thresh=int(len(timeseries_seriescols.index) * 0.05))
# transpose to another possible shape
# timeseries_datecols = timeseries_seriescols.transpose()

timeseries_seriescols = timeseries_seriescols.fillna(0)
print("FILLED NA WITH ZERO which you don't usually want to do!")

""" 
Plot Time Series
"""
figures = False
if figures:
    try:
        # plot one and save
        # series = 'SP500'
        series = timeseries_seriescols.columns[0]
        ax = timeseries_seriescols[series].fillna(method = 'ffill').plot()
        fig = ax.get_figure()
        # fig.savefig((series + '.png'), dpi=300)
        
        # plot multiple time series all on same scale
        from sklearn.preprocessing import MinMaxScaler
        ax = pd.DataFrame(MinMaxScaler().fit_transform(timeseries_seriescols)).sample(5, axis = 1).plot()
        ax.get_legend().remove()
        fig = ax.get_figure()
        # fig.savefig('MultipleTimeSeries.png', dpi=300)
    except Exception:
        pass

"""
Some basics
"""
import math
def smape(actual, forecast):
    # Expects a 2-D numpy array of forecast_length * n series
    out_array = np.zeros(actual.shape[1])
    for r in range(actual.shape[1]):
        try:
            y_true = actual[:,r]
            y_pred = forecast[:,r]
            y_pred = y_pred[~np.isnan(y_true)]
            y_true = y_true[~np.isnan(y_true)]

            out = 0
            for i in range(y_true.shape[0]):
                a = y_true[i]
                b = math.fabs(y_pred[i])
                c = a+b
                if c == 0:
                    continue
                out += math.fabs(a - b) / c
            out *= (200.0 / y_true.shape[0])
        except Exception:
            out = np.nan
        out_array[r] = out
    return out_array

def mae(A, F):
    try:
        mae_result = abs(A - F)
    except Exception:
        mae_result = np.nan
    return mae_result

train = timeseries_seriescols.head(len(timeseries_seriescols.index) - forecast_length)
test = timeseries_seriescols.tail(forecast_length)



class ModelResult(object):
    def __init__(self, name=None, forecast=None, mae=None, overall_mae=-1, smape=None, overall_smape=-1, runtime=-1):
        self.name = name
        self.forecast = forecast
        self.mae = mae
        self.overall_mae = overall_mae
        self.smape = smape
        self.overall_smape = overall_smape
        self.runtime = runtime

    def __repr__(self):
        return "Time Series Model Result: " + str(self.name)
    def __str__(self):
        return "Time Series Model Result: " + str(self.name)
    def result_message(self):
        return "TS Method: " + str(self.name) + " of Avg SMAPE: " + str(self.overall_smape)

class EvaluationReturn(object):
    def __init__(self, model_performance = np.nan, per_series_mae = np.nan, per_series_smape = np.nan, errors = np.nan):
        self.model_performance = model_performance
        self.per_series_mae = per_series_mae
        self.per_series_smape = per_series_smape
        self.errors = errors
subset = 'All'

def model_evaluator(train, test, subset = 'All'):
    model_results = pd.DataFrame(columns = ['method', 'runtime', 'overall_smape', 'overall_mae', 'object_name'])
    error_list = []
    
    # take a subset of the availabe series to speed up
    if isinstance(subset, (int, float, complex)) and not isinstance(subset, bool):
        if subset > len(train.columns):
            subset = len(train.columns) 
        train = train.sample(subset, axis = 1, random_state = 425, replace = False)
        test = test[train.columns]
    
    trainStartDate = min(train.index)
    trainEndDate = max(train.index)
    testStartDate = min(test.index)
    testEndDate = max(test.index)
    
    """
    Linear Regression
    """
    GLM = ModelResult("GLM")
    try:
        from statsmodels.regression.linear_model import GLS
        startTime = datetime.datetime.now()
        glm_model = GLS(train.values, (train.index.astype(int).values), missing = 'drop').fit()
        GLM.forecast = glm_model.predict(test.index.astype(int).values)
        GLM.runtime = datetime.datetime.now() - startTime
        
        GLM.mae = pd.DataFrame(mae(test.values, GLM.forecast)).mean(axis=0, skipna = True)
        GLM.overall_mae = np.nanmean(GLM.mae)
        GLM.smape = smape(test.values, GLM.forecast)
        GLM.overall_smape = np.nanmean(GLM.smape)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    
    currentResult = pd.DataFrame({
            'method': GLM.name, 
            'runtime': GLM.runtime, 
            'overall_smape': GLM.overall_smape, 
            'overall_mae': GLM.overall_mae,
            'object_name': 'GLM'
            }, index = [0])
    model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
    GLM.result_message()
    """
    ETS Forecast
    
    Here I copy the comb benchmark of the M4 competition:
        the simple arithmetic average of Single, Holt and Damped exponential smoothing
        
    http://www.statsmodels.org/stable/tsa.html
    """
    
    sETS = ModelResult("SimpleETS")
    try:
        from statsmodels.tsa.holtwinters import SimpleExpSmoothing
        startTime = datetime.datetime.now()
        forecast = pd.DataFrame()
        for series in train.columns:
            current_series = train[series].copy()
            current_series = current_series.fillna(method='ffill').fillna(method='bfill')
            sesModel = SimpleExpSmoothing(current_series).fit()
            sesPred = sesModel.predict(start=testStartDate, end=testEndDate)
            if no_negatives:
                sesPred = sesPred.where(sesPred > 0, 0)   # replace all negatives with zeroes, remove if you want negatives!
            forecast = pd.concat([forecast, sesPred], axis = 1)
        sETS.forecast = forecast.values
        sETS.runtime = datetime.datetime.now() - startTime
        
        sETS.mae = pd.DataFrame(mae(test.values, sETS.forecast)).mean(axis=0, skipna = True)
        sETS.overall_mae = np.nanmean(sETS.mae)
        sETS.smape = smape(test.values, sETS.forecast)
        sETS.overall_smape = np.nanmean(sETS.smape)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    
    currentResult = pd.DataFrame({
            'method': sETS.name, 
            'runtime': sETS.runtime, 
            'overall_smape': sETS.overall_smape, 
            'overall_mae': sETS.overall_mae,
            'object_name': 'sETS'
            }, index = [0])
    model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
    
    sETS.result_message()
    
    
    ETS = ModelResult("ETS")
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        startTime = datetime.datetime.now()
        forecast = pd.DataFrame()
        for series in train.columns:
            current_series = train[series].copy()
            current_series = current_series.fillna(method='ffill').fillna(method='bfill')
            esModel = ExponentialSmoothing(current_series, damped = False).fit()
            esPred = esModel.predict(start=testStartDate, end=testEndDate)
            if no_negatives:
                esPred = esPred.where(esPred > 0, 0)   # replace all negatives with zeroes, remove if you want negatives!
            forecast = pd.concat([forecast, esPred], axis = 1)
        ETS.forecast = forecast.values
        ETS.runtime = datetime.datetime.now() - startTime
        
        ETS.mae = pd.DataFrame(mae(test.values, ETS.forecast)).mean(axis=0, skipna = True)
        ETS.overall_mae = np.nanmean(ETS.mae)
        ETS.smape = smape(test.values, ETS.forecast)
        ETS.overall_smape = np.nanmean(ETS.smape)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    
    currentResult = pd.DataFrame({
            'method': ETS.name, 
            'runtime': ETS.runtime, 
            'overall_smape': ETS.overall_smape, 
            'overall_mae': ETS.overall_mae,
            'object_name': 'ETS'
            }, index = [0])
    model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
    ETS.result_message()
    """
    Damped ETS
    """
    dETS = ModelResult("dampedETS")
    try:
        startTime = datetime.datetime.now()
        forecast = pd.DataFrame()
        for series in train.columns:
            current_series = train[series].copy()
            current_series = current_series.fillna(method='ffill').fillna(method='bfill')
            esModel = ExponentialSmoothing(current_series, damped = True, trend = 'add').fit()
            esPred = esModel.predict(start=testStartDate, end=testEndDate)
            if no_negatives:
                esPred = esPred.where(esPred > 0, 0)   # replace all negatives with zeroes, remove if you want negatives!
            forecast = pd.concat([forecast, esPred], axis = 1)
        dETS.forecast = forecast.values
        dETS.runtime = datetime.datetime.now() - startTime
        
        dETS.mae = pd.DataFrame(mae(test.values, dETS.forecast)).mean(axis=0, skipna = True)
        dETS.overall_mae = np.nanmean(dETS.mae)
        dETS.smape = smape(test.values, dETS.forecast)
        dETS.overall_smape = np.nanmean(dETS.smape)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    
    
    currentResult = pd.DataFrame({
            'method': dETS.name, 
            'runtime': dETS.runtime, 
            'overall_smape': dETS.overall_smape, 
            'overall_mae': dETS.overall_mae,
            'object_name': 'dETS'
            }, index = [0])
    model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
    dETS.result_message()
    """
    Markov AutoRegression - not yet stable release
    
    from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    MarkovAuto = ModelResult("MarkovAutoregression")
    
    startTime = datetime.datetime.now()
    forecast = pd.DataFrame()
    for series in train.columns:
        current_series = train[series].copy()
        current_series = current_series.fillna(method='ffill').fillna(method='bfill')
        model = MarkovRegression(current_series, k_regimes=3, trend='nc', switching_variance=True).fit()
        maPred = model.predict(start=testStartDate, end=testEndDate)
        if no_negatives:
            maPred = maPred.where(maPred > 0, 0)   # replace all negatives with zeroes, remove if you want negatives!
        forecast = pd.concat([forecast, maPred], axis = 1)
    MarkovAuto.forecast = forecast.values
    MarkovAuto.runtime = datetime.datetime.now() - startTime
    
    MarkovAuto.mae = pd.DataFrame(mae(test.values, MarkovAuto.forecast)).mean(axis=0, skipna = True)
    MarkovAuto.overall_mae = np.nanmean(MarkovAuto.mae)
    MarkovAuto.smape = smape(test.values, MarkovAuto.forecast)
    MarkovAuto.overall_smape = np.nanmean(MarkovAuto.smape)
    
    currentResult = pd.DataFrame({
            'method': MarkovAuto.name, 
            'runtime': MarkovAuto.runtime, 
            'overall_smape': MarkovAuto.overall_smape, 
            'overall_mae': MarkovAuto.overall_mae
            }, index = [0])
    model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
    
    MarkovRegression(dta_kns, k_regimes=3, trend='nc', switching_variance=True)
    MarkovAutoregression(dta_hamilton, k_regimes=2, order=4, switching_ar=False)
    """
    UnComp = ModelResult("UnobservedComponents")
    
    try:
        from statsmodels.tsa.statespace.structural import UnobservedComponents
        startTime = datetime.datetime.now()
        forecast = pd.DataFrame()
        unrestricted_model = {
            'level': 'local linear trend', 'cycle': True, 'damped_cycle': True, 'stochastic_cycle': True
        }
        for series in train.columns:
            try:
                current_series = train[series].copy()
                current_series = current_series.fillna(method='ffill').fillna(method='bfill')
                model = UnobservedComponents(current_series, **unrestricted_model).fit(method='powell')
                ucPred = model.predict(start=testStartDate, end=testEndDate)
            except Exception:
                ucPred = (np.zeros((forecast_length,)))
            if no_negatives:
                try:
                    ucPred = pd.Series(np.where(ucPred > 0, ucPred, 0))
                except Exception:
                    ucPred = ucPred.where(ucPred > 0, 0)   # replace all negatives with zeroes, remove if you want negatives!
            forecast = pd.concat([forecast, ucPred], axis = 1)
        UnComp.forecast = forecast.values
        UnComp.runtime = datetime.datetime.now() - startTime
        
        UnComp.mae = pd.DataFrame(mae(test.values, UnComp.forecast)).mean(axis=0, skipna = True)
        UnComp.overall_mae = np.nanmean(UnComp.mae)
        UnComp.smape = smape(test.values, UnComp.forecast)
        UnComp.overall_smape = np.nanmean(UnComp.smape)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    
    
    currentResult = pd.DataFrame({
            'method': UnComp.name, 
            'runtime': UnComp.runtime, 
            'overall_smape': UnComp.overall_smape, 
            'overall_mae': UnComp.overall_mae,
            'object_name': 'UnComp'
            }, index = [0])
    model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
    UnComp.result_message()
    """
    Simple ARIMA
    """
    SARIMA = ModelResult("ARIMA 101")
    try:
        from statsmodels.tsa.arima_model import ARIMA
        startTime = datetime.datetime.now()
        forecast = pd.DataFrame()
        for series in train.columns:
            try:
                current_series = train[series].copy()
                current_series = current_series.fillna(method='ffill').fillna(method='bfill')
                model = ARIMA(current_series, order=(1,0,1)).fit()
                saPred = model.predict(start=testStartDate, end=testEndDate)
            except Exception:
                saPred = (np.zeros((forecast_length,)))
            if no_negatives:
                try:
                    saPred = pd.Series(np.where(saPred > 0, saPred, 0))
                except Exception:
                    saPred = saPred.where(saPred > 0, 0)   # replace all negatives with zeroes, remove if you want negatives!
            forecast = pd.concat([forecast, saPred], axis = 1)
        SARIMA.forecast = forecast.values
        SARIMA.runtime = datetime.datetime.now() - startTime
        
        SARIMA.mae = pd.DataFrame(mae(test.values, SARIMA.forecast)).mean(axis=0, skipna = True)
        SARIMA.overall_mae = np.nanmean(SARIMA.mae)
        SARIMA.smape = smape(test.values, SARIMA.forecast)
        SARIMA.overall_smape = np.nanmean(SARIMA.smape)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    
    currentResult = pd.DataFrame({
            'method': SARIMA.name, 
            'runtime': SARIMA.runtime, 
            'overall_smape': SARIMA.overall_smape, 
            'overall_mae': SARIMA.overall_mae,
            'object_name': 'SARIMA'
            }, index = [0])
    model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
    SARIMA.result_message()
    """
    Prophet
    expects data in a specific format: a 'ds' column for dates and a 'y' column for values
    can handle missing data
    https://facebook.github.io/prophet/
	conda install -c conda-forge fbprophet
    """
    
    ProphetResult = ModelResult("Prophet")
    
    try:
        from fbprophet import Prophet
        startTime = datetime.datetime.now()
        forecast = pd.DataFrame()
        for series in train.columns:
            current_series = train.copy()
            current_series['y'] = current_series[series]
            current_series['ds'] = current_series.index
            
            m = Prophet().fit(current_series)
            future = m.make_future_dataframe(periods=forecast_length)
            fcst = m.predict(future)
            if no_negatives:
                fcst.loc[~(fcst['yhat'] > 0), 'yhat'] = 0   
            fcst = fcst.tail(forecast_length) # remove the backcast
            forecast = pd.concat([forecast, fcst['yhat']], axis = 1)
        ProphetResult.forecast = forecast.values
        ProphetResult.runtime = datetime.datetime.now() - startTime
        
        ProphetResult.mae = pd.DataFrame(mae(test.values, ProphetResult.forecast)).mean(axis=0, skipna = True)
        ProphetResult.overall_mae = np.nanmean(ProphetResult.mae)
        ProphetResult.smape = smape(test.values, ProphetResult.forecast)
        ProphetResult.overall_smape = np.nanmean(ProphetResult.smape)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    
    currentResult = pd.DataFrame({
            'method': ProphetResult.name, 
            'runtime': ProphetResult.runtime, 
            'overall_smape': ProphetResult.overall_smape, 
            'overall_mae': ProphetResult.overall_mae,
            'object_name': 'ProphetResult'
            }, index = [0])
    model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
    ProphetResult.result_message()
    
    ProphetResultHoliday = ModelResult("Prophet w Holidays")
    try:
        startTime = datetime.datetime.now()
        forecast = pd.DataFrame()
        for series in train.columns:
            current_series = train.copy()
            current_series['y'] = current_series[series]
            current_series['ds'] = current_series.index
            
            m = Prophet()
            m.add_country_holidays(country_name='US')
            m.fit(current_series)
            future = m.make_future_dataframe(periods=forecast_length)
            fcst = m.predict(future)
            if no_negatives:
                fcst.loc[~(fcst['yhat'] > 0), 'yhat'] = 0   
            fcst = fcst.tail(forecast_length) # remove the backcast
            forecast = pd.concat([forecast, fcst['yhat']], axis = 1)
        ProphetResultHoliday.forecast = forecast.values
        ProphetResultHoliday.runtime = datetime.datetime.now() - startTime
        
        ProphetResultHoliday.mae = pd.DataFrame(mae(test.values, ProphetResultHoliday.forecast)).mean(axis=0, skipna = True)
        ProphetResultHoliday.overall_mae = np.nanmean(ProphetResultHoliday.mae)
        ProphetResultHoliday.smape = smape(test.values, ProphetResultHoliday.forecast)
        ProphetResultHoliday.overall_smape = np.nanmean(ProphetResultHoliday.smape)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    
    
    currentResult = pd.DataFrame({
            'method': ProphetResultHoliday.name, 
            'runtime': ProphetResultHoliday.runtime, 
            'overall_smape': ProphetResultHoliday.overall_smape, 
            'overall_mae': ProphetResultHoliday.overall_mae,
            'object_name': 'ProphetResultHoliday'
            }, index = [0])
    model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
    ProphetResultHoliday.result_message()
    
    """
    AutoARIMA
    pip install pmdarima  (==1.4.0 to play with GluonTS==0.4.0)
    Install pmdarima after installing GluonTS to prevent numpy issues
    """
    AutoArima = ModelResult("Auto ARIMA S7")
    try:
        from pmdarima.arima import auto_arima
        startTime = datetime.datetime.now()
        forecast = pd.DataFrame()
        model_orders = []
        for series in train.columns:
            try:
                current_series = train[series].copy()
                current_series = current_series.fillna(method='ffill').fillna(method='bfill')
                current_series = current_series.reset_index(drop = True)
                model = auto_arima(current_series, error_action='ignore', seasonal=True, m=7, suppress_warnings = True)
                saPred = model.predict(n_periods=forecast_length)
                model_orders.extend([model.order])
            except Exception:
                saPred = (np.zeros((forecast_length,)))
                model_orders.extend([(0,0,0)])
            if no_negatives:
                try:
                    saPred = pd.Series(np.where(saPred > 0, saPred, 0))
                except Exception:
                    saPred = np.where(saPred > 0, saPred, 0)
            forecast = pd.concat([forecast, pd.Series(saPred)], axis = 1)
        AutoArima.forecast = forecast.values
        AutoArima.runtime = datetime.datetime.now() - startTime
        
        AutoArima.mae = pd.DataFrame(mae(test.values, AutoArima.forecast)).mean(axis=0, skipna = True)
        AutoArima.overall_mae = np.nanmean(AutoArima.mae)
        AutoArima.smape = smape(test.values, AutoArima.forecast)
        AutoArima.overall_smape = np.nanmean(AutoArima.smape)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    
    
    currentResult = pd.DataFrame({
            'method': AutoArima.name, 
            'runtime': AutoArima.runtime, 
            'overall_smape': AutoArima.overall_smape, 
            'overall_mae': AutoArima.overall_mae,
            'object_name': 'AutoArima'
            }, index = [0])
    model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
    AutoArima.result_message()
    
    """
    GluonTS
    https://gluon-ts.mxnet.io/
    
    Gluon has a nice built in evaluator, but that is not used here
    First install mxnet if you haven't already (in my case, pip install mxnet-cu90mkl)
    pip install gluonts==0.4.0
    pip install git+https://github.com/awslabs/gluon-ts.git
    pip install numpy==1.17.4 after gluon, because gluon seems okay with new version,
    but most things aren't okay with 1.14 numpy
    """
    try:
        try:
            from gluonts.transform import FieldName # old way (0.3.3 and older)
        except Exception:
            from gluonts.dataset.field_names import FieldName # new way
        
        gluon_train = train.fillna(method='ffill').fillna(method='bfill').transpose()
        
        ts_metadata = {'num_series': len(gluon_train.index),
                              'forecast_length': forecast_length,
                              'freq': frequency,
                              'gluon_start': [gluon_train.columns[0] for _ in range(len(gluon_train.index))],
                              'context_length': 2 * forecast_length
                             }
        from gluonts.dataset.common import ListDataset
        
        test_ds = ListDataset([{FieldName.TARGET: target, 
                                 FieldName.START: start
                                 # FieldName.FEAT_DYNAMIC_REAL: custDayOfWeek
                                 } 
                                for (target, start) in zip( # , custDayOfWeek
                                        gluon_train.values, 
                                        ts_metadata['gluon_start']
                                        # custDayOfWeek
                                        )],
                                freq=ts_metadata['freq']
                                )
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
        
    GluonNPTS = ModelResult("Gluon NPTS")
    try:
        startTime = datetime.datetime.now()
        from gluonts.model.npts import NPTSEstimator
        estimator = NPTSEstimator(freq=ts_metadata['freq'],
                                    context_length=ts_metadata['context_length'],
                                    prediction_length=ts_metadata['forecast_length'])
        
        forecast = pd.DataFrame()
        GluonPredictor = estimator.train(test_ds)
        gluon_results = GluonPredictor.predict(test_ds)
        i = 0
        for result in gluon_results:
            currentCust = gluon_train.index[i]
            rowForecast = pd.DataFrame({
                    "ForecastDate": pd.date_range(start = result.start_date, periods = ts_metadata['forecast_length'], freq = ts_metadata['freq']),
                    "series_id": currentCust,
                    # "Quantile10thForecast": (result.quantile(0.1)).astype(int),
                    "MedianForecast": (result.quantile(0.5)).astype(int),
                    # "Quantile90thForecast": (result.quantile(0.9)).astype(int)
                    })
            forecast = pd.concat([forecast, rowForecast], ignore_index = True).reset_index(drop = True)
            i += 1
        forecast = forecast.pivot_table(values='MedianForecast', index='ForecastDate', columns='series_id')
        
        GluonNPTS.forecast = forecast.values
        GluonNPTS.runtime = datetime.datetime.now() - startTime
        
        GluonNPTS.mae = pd.DataFrame(mae(test.values, GluonNPTS.forecast)).mean(axis=0, skipna = True)
        GluonNPTS.overall_mae = np.nanmean(GluonNPTS.mae)
        GluonNPTS.smape = smape(test.values, GluonNPTS.forecast)
        GluonNPTS.overall_smape = np.nanmean(GluonNPTS.smape)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    
    currentResult = pd.DataFrame({
            'method': GluonNPTS.name, 
            'runtime': GluonNPTS.runtime, 
            'overall_smape': GluonNPTS.overall_smape, 
            'overall_mae': GluonNPTS.overall_mae,
            'object_name': 'GluonNPTS'
            }, index = [0])
    model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
    GluonNPTS.result_message()
    
    
    GluonDeepAR = ModelResult("Gluon DeepARE")
    try:
        startTime = datetime.datetime.now()
        from gluonts.model.deepar import DeepAREstimator
        from gluonts.trainer import Trainer
        estimator = DeepAREstimator(freq=ts_metadata['freq'],
                                    context_length=ts_metadata['context_length'],
                                    prediction_length=ts_metadata['forecast_length'] 
                                    ,trainer=Trainer(epochs=20)
                                    )
        
        forecast = pd.DataFrame()
        GluonPredictor = estimator.train(test_ds)
        gluon_results = GluonPredictor.predict(test_ds)
        i = 0
        for result in gluon_results:
            currentCust = gluon_train.index[i]
            rowForecast = pd.DataFrame({
                    "ForecastDate": pd.date_range(start = result.start_date, periods = ts_metadata['forecast_length'], freq = ts_metadata['freq']),
                    "series_id": currentCust,
                    # "Quantile10thForecast": (result.quantile(0.1)).astype(int),
                    "MedianForecast": (result.quantile(0.5)).astype(int),
                    # "Quantile90thForecast": (result.quantile(0.9)).astype(int)
                    })
            forecast = pd.concat([forecast, rowForecast], ignore_index = True).reset_index(drop = True)
            i += 1
        forecast = forecast.pivot_table(values='MedianForecast', index='ForecastDate', columns='series_id')
        
        GluonDeepAR.forecast = forecast.values
        GluonDeepAR.runtime = datetime.datetime.now() - startTime
        
        GluonDeepAR.mae = pd.DataFrame(mae(test.values, GluonDeepAR.forecast)).mean(axis=0, skipna = True)
        GluonDeepAR.overall_mae = np.nanmean(GluonDeepAR.mae)
        GluonDeepAR.smape = smape(test.values, GluonDeepAR.forecast)
        GluonDeepAR.overall_smape = np.nanmean(GluonDeepAR.smape)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    
    currentResult = pd.DataFrame({
            'method': GluonDeepAR.name, 
            'runtime': GluonDeepAR.runtime, 
            'overall_smape': GluonDeepAR.overall_smape, 
            'overall_mae': GluonDeepAR.overall_mae,
            'object_name': 'GluonDeepAR'
            }, index = [0])
    model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
    GluonDeepAR.result_message()
    
        
    GluonMQCNN = ModelResult("Gluon MQCNN")
    try:
        startTime = datetime.datetime.now()
        # from gluonts.trainer import Trainer
        from gluonts.model.seq2seq import MQCNNEstimator
        estimator = MQCNNEstimator(freq=ts_metadata['freq'],
                                    context_length=ts_metadata['context_length'],
                                    prediction_length=ts_metadata['forecast_length'] 
                                    ,trainer=Trainer(epochs=20)
                                    )
        
        forecast = pd.DataFrame()
        GluonPredictor = estimator.train(test_ds)
        gluon_results = GluonPredictor.predict(test_ds)
        i = 0
        for result in gluon_results:
            currentCust = gluon_train.index[i]
            rowForecast = pd.DataFrame({
                    "ForecastDate": pd.date_range(start = result.start_date, periods = ts_metadata['forecast_length'], freq = ts_metadata['freq']),
                    "series_id": currentCust,
                    # "Quantile10thForecast": (result.quantile(0.1)).astype(int),
                    "MedianForecast": (result.quantile(0.5)).astype(int),
                    # "Quantile90thForecast": (result.quantile(0.9)).astype(int)
                    })
            forecast = pd.concat([forecast, rowForecast], ignore_index = True).reset_index(drop = True)
            i += 1
        forecast = forecast.pivot_table(values='MedianForecast', index='ForecastDate', columns='series_id')
        
        GluonMQCNN.forecast = forecast.values
        GluonMQCNN.runtime = datetime.datetime.now() - startTime
        
        GluonMQCNN.mae = pd.DataFrame(mae(test.values, GluonMQCNN.forecast)).mean(axis=0, skipna = True)
        GluonMQCNN.overall_mae = np.nanmean(GluonMQCNN.mae)
        GluonMQCNN.smape = smape(test.values, GluonMQCNN.forecast)
        GluonMQCNN.overall_smape = np.nanmean(GluonMQCNN.smape)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    
    currentResult = pd.DataFrame({
            'method': GluonMQCNN.name, 
            'runtime': GluonMQCNN.runtime, 
            'overall_smape': GluonMQCNN.overall_smape, 
            'overall_mae': GluonMQCNN.overall_mae,
            'object_name': 'GluonMQCNN'
            }, index = [0])
    model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
    GluonMQCNN.result_message()
    
    GluonSFF = ModelResult("Gluon SFF")
    try:
        startTime = datetime.datetime.now()
        from gluonts.trainer import Trainer
        from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
        estimator = SimpleFeedForwardEstimator(
            prediction_length=ts_metadata['forecast_length'],
            context_length=ts_metadata['context_length'],
            freq=ts_metadata['freq'],
            trainer=Trainer(epochs=10, 
                            learning_rate=1e-3, 
                            hybridize=False, 
                            num_batches_per_epoch=100
                           ))
        forecast = pd.DataFrame()
        GluonPredictor = estimator.train(test_ds)
        gluon_results = GluonPredictor.predict(test_ds)
        i = 0
        for result in gluon_results:
            currentCust = gluon_train.index[i]
            rowForecast = pd.DataFrame({
                    "ForecastDate": pd.date_range(start = result.start_date, periods = ts_metadata['forecast_length'], freq = ts_metadata['freq']),
                    "series_id": currentCust,
                    # "Quantile10thForecast": (result.quantile(0.1)).astype(int),
                    "MedianForecast": (result.quantile(0.5)).astype(int),
                    # "Quantile90thForecast": (result.quantile(0.9)).astype(int)
                    })
            forecast = pd.concat([forecast, rowForecast], ignore_index = True).reset_index(drop = True)
            i += 1
        forecast = forecast.pivot_table(values='MedianForecast', index='ForecastDate', columns='series_id')
        
        GluonSFF.forecast = forecast.values
        GluonSFF.runtime = datetime.datetime.now() - startTime
        
        GluonSFF.mae = pd.DataFrame(mae(test.values, GluonSFF.forecast)).mean(axis=0, skipna = True)
        GluonSFF.overall_mae = np.nanmean(GluonSFF.mae)
        GluonSFF.smape = smape(test.values, GluonSFF.forecast)
        GluonSFF.overall_smape = np.nanmean(GluonSFF.smape)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    
    currentResult = pd.DataFrame({
            'method': GluonSFF.name, 
            'runtime': GluonSFF.runtime, 
            'overall_smape': GluonSFF.overall_smape, 
            'overall_mae': GluonSFF.overall_mae,
            'object_name': 'GluonSFF'
            }, index = [0])
    model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
    GluonSFF.result_message()
    
    """
    Simple Ensembles
    """
    AllModelEnsemble = ModelResult("All Model Ensemble")
    try:
        master_array = np.zeros((test.shape[0], test.shape[1]))
        n = 0
        all_models = [GLM, sETS, ETS, dETS, ProphetResult, ProphetResultHoliday, 
     AutoArima, SARIMA, UnComp, GluonNPTS, GluonDeepAR, GluonMQCNN, GluonSFF]
        for modelmethod in all_models:
            if modelmethod.overall_smape != -1:
                master_array = master_array + modelmethod.forecast
                n += 1
    
        AllModelEnsemble.forecast = master_array/n
        AllModelEnsemble.runtime = datetime.datetime.now() - startTime
        
        AllModelEnsemble.mae = pd.DataFrame(mae(test.values, AllModelEnsemble.forecast)).mean(axis=0, skipna = True)
        AllModelEnsemble.overall_mae = np.nanmean(AllModelEnsemble.mae)
        AllModelEnsemble.smape = smape(test.values, AllModelEnsemble.forecast)
        AllModelEnsemble.overall_smape = np.nanmean(AllModelEnsemble.smape)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    
    currentResult = pd.DataFrame({
            'method': AllModelEnsemble.name, 
            'runtime': AllModelEnsemble.runtime, 
            'overall_smape': AllModelEnsemble.overall_smape, 
            'overall_mae': AllModelEnsemble.overall_mae,
            'object_name': 'AllModelEnsemble'
            }, index = [0])
    model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
    
    M4 = ModelResult("ETS M4 Comb")
    try:
        master_array = np.zeros((test.shape[0], test.shape[1]))
        n = 0
        ensemble_models = [sETS, ETS, dETS]
        for modelmethod in ensemble_models:
            if modelmethod.overall_smape != -1:
                master_array = master_array + modelmethod.forecast
                n += 1
    
        M4.forecast = master_array/n
        M4.runtime = datetime.datetime.now() - startTime
        
        M4.mae = pd.DataFrame(mae(test.values, M4.forecast)).mean(axis=0, skipna = True)
        M4.overall_mae = np.nanmean(M4.mae)
        M4.smape = smape(test.values, M4.forecast)
        M4.overall_smape = np.nanmean(M4.smape)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    
    currentResult = pd.DataFrame({
            'method': M4.name, 
            'runtime': M4.runtime, 
            'overall_smape': M4.overall_smape, 
            'overall_mae': M4.overall_mae,
            'object_name': 'M4'
            }, index = [0])
    model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
    
    bestN = ModelResult("Best N Ensemble")
    try:
        master_array = np.zeros((test.shape[0], test.shape[1]))
        
        bestNames = model_results[model_results['overall_smape'] > 0].sort_values('overall_smape', ascending = True).head(3)['object_name'].values
        n = 0
        for modelmethod in bestNames:
            modelmethod_obj = eval(modelmethod) # globals()[modelmethod] # getattr(sys.modules[__name__], modelmethod)
            if modelmethod_obj.overall_smape != -1:
                master_array = master_array + modelmethod_obj.forecast
                n += 1
    
        bestN.forecast = master_array/n
        bestN.runtime = datetime.datetime.now() - startTime
        
        bestN.mae = pd.DataFrame(mae(test.values, bestN.forecast)).mean(axis=0, skipna = True)
        bestN.overall_mae = np.nanmean(bestN.mae)
        bestN.smape = smape(test.values, bestN.forecast)
        bestN.overall_smape = np.nanmean(bestN.smape)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    
    currentResult = pd.DataFrame({
            'method': bestN.name, 
            'runtime': bestN.runtime, 
            'overall_smape': bestN.overall_smape, 
            'overall_mae': bestN.overall_mae,
            'object_name': 'bestN'
            }, index = [0])
    model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
    
    EPA = ModelResult("ETS + Prophet + ARIMA")
    try:
        master_array = np.zeros((test.shape[0], test.shape[1]))
        n = 0
        ensemble_models = [AutoArima, ETS, ProphetResult]
        for modelmethod in ensemble_models:
            if modelmethod.overall_smape != -1:
                master_array = master_array + modelmethod.forecast
                n += 1
    
        EPA.forecast = master_array/n
        EPA.runtime = datetime.datetime.now() - startTime
        
        EPA.mae = pd.DataFrame(mae(test.values, EPA.forecast)).mean(axis=0, skipna = True)
        EPA.overall_mae = np.nanmean(EPA.mae)
        EPA.smape = smape(test.values, EPA.forecast)
        EPA.overall_smape = np.nanmean(EPA.smape)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    
    currentResult = pd.DataFrame({
            'method': EPA.name, 
            'runtime': EPA.runtime, 
            'overall_smape': EPA.overall_smape, 
            'overall_mae': EPA.overall_mae,
            'object_name': 'EPA'
            }, index = [0])
    model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
    
    
    GE = ModelResult("GluonDeepAR + ETS")
    try:
        master_array = np.zeros((test.shape[0], test.shape[1]))
        n = 0
        ensemble_models = [GluonDeepAR, ETS]
        for modelmethod in ensemble_models:
            if modelmethod.overall_smape != -1:
                master_array = master_array + modelmethod.forecast
                n += 1
    
        GE.forecast = master_array/n
        GE.runtime = datetime.datetime.now() - startTime
        
        GE.mae = pd.DataFrame(mae(test.values, GE.forecast)).mean(axis=0, skipna = True)
        GE.overall_mae = np.nanmean(GE.mae)
        GE.smape = smape(test.values, GE.forecast)
        GE.overall_smape = np.nanmean(GE.smape)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    
    currentResult = pd.DataFrame({
            'method': GE.name, 
            'runtime': GE.runtime, 
            'overall_smape': GE.overall_smape, 
            'overall_mae': GE.overall_mae,
            'object_name': 'GE'
            }, index = [0])
    model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
    
    per_series_smape = np.zeros((len(test.columns),))
    try:
        finishedNames = model_results[model_results['overall_smape'] > 0].sort_values('overall_smape', ascending = True)['object_name'].values
        for modelmethod in finishedNames:
            modelmethod_obj = eval(modelmethod) # getattr(sys.modules[__name__], modelmethod) # or eval()
            if modelmethod_obj.overall_smape != -1:
                per_series_smape = np.vstack((per_series_smape, modelmethod_obj.smape))
        per_series_smape = np.delete(per_series_smape, (0), axis=0) # remove the zeros
        per_series_smape = pd.DataFrame(per_series_smape, columns = test.columns)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    
    per_series_mae = np.zeros((len(test.columns),))
    try:
        finishedNames = model_results[model_results['overall_mae'] > 0].sort_values('overall_mae', ascending = True)['object_name'].values
        for modelmethod in finishedNames:
            modelmethod_obj =  eval(modelmethod) #getattr(sys.modules[__name__], modelmethod)
            if modelmethod_obj.overall_smape != -1:
                per_series_mae = np.vstack((per_series_mae, modelmethod_obj.smape))
        per_series_mae = np.delete(per_series_mae, (0), axis=0) # remove the zeros
        per_series_mae = pd.DataFrame(per_series_mae, columns = test.columns)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    
    final_result = EvaluationReturn("run_result")
    
    final_result.model_performance = model_results
    final_result.per_series_mae = per_series_mae
    final_result.per_series_smape = per_series_smape
    final_result.errors = error_list
    return final_result

evaluator_result = model_evaluator(train, test, subset = 1000)
eval_table = evaluator_result.model_performance.sort_values('overall_smape', ascending = True)
print("Complete at: " + str(datetime.datetime.now()))

"""
Handle missing with better options
    Slice out missing days and pretend it's continuous (make up dates, remove week day effects)
    Fill forward and backwards
    Fill 0
Handle different length time series (time series that don't start until later)
Run multiple test segments
Limit the context length to a standard length?

Profile metadata of series (number NaN imputed, context length, etc)

Capture volatility of results - basically get methods that do great on many even if they absymally fail on others

Handle failures on series with Try/Except

Forecast distance blending (mix those more accurate in short term and long term) ensemble

For monthly, account for the last, incomplete, month
"""
