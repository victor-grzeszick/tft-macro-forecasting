import pandas as pd
import numpy as np
import os

def evaluate(y_history, y_val):
    return (abs(y_history - y_val).mean(), ((y_history - y_val)**2).mean()**0.5)

def align(name, h, df_val, predictions_df):
    rows = predictions_df[(predictions_df["target"] == name) & (predictions_df["horizon"] == h)]
    by_model = rows.set_index('model')["prediction"]
    rw = by_model['RW']
    arima = by_model['ARIMA']
    varr = by_model['VAR']
    tft = by_model['TFT']

    common = rw.index.intersection(arima.index).intersection(varr.index)

    rw = rw.loc[common]
    arima = arima.loc[common]
    varr  = varr.loc[common]    

    actual = df_val[name].loc[common]

    return actual, rw, arima, tft, varr