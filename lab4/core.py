import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
from pylab import rcParams


def vwap(df: DataFrame):
    volume = df['Volume']
    total_price = (df['Close'] + df['High'] + df['Low']) / 3
    return df.assign(VWAP=(total_price * volume).cumsum() / volume.cumsum())


def split_data_by_steps(data: DataFrame, steps=1):
    x_part, y_part = [], []
    for i in range(len(data) - steps):
        x = data.iloc[i: (i + steps), 0].values
        x_part.append(x)
        y = data.iloc[i + steps].values
        y_part.append(y)
    return np.array(x_part), np.array(y_part)


def prepare_prediction_sample(raw):
    result = []
    sub_result = []
    for element in raw:
        sub_result.append(element[0])
    result.append(sub_result)
    return result


def predict_values(model, source, predict_steps, window_size):
    result = source.copy()
    for i in range(predict_steps):
        prediction = model.predict(prepare_prediction_sample(result.values[-window_size:]), verbose=0)
        new_date = result.index[-1] + pd.DateOffset(months=1)
        result.loc[new_date] = {'VWAP': prediction[0][0]}
    return result


def forecast(model, source, months_to_forecast, window_size):
    forecasted_result = predict_values(model, source=source, predict_steps=months_to_forecast, window_size=window_size)
    return forecasted_result


def forecast_and_plot(model, source, months_to_forecast, window_size):
    forecasted_result = forecast(model=model, source=source, months_to_forecast=months_to_forecast,
                                 window_size=window_size)
    actual_stocks = forecasted_result[:'2022-03']
    prediction = forecasted_result['2022-03':]
    rcParams['figure.figsize'] = 18, 9
    plt.plot(actual_stocks, color="black")
    plt.plot(prediction, color="green")
    plt.title('Разделение данных об изменении показателя VWAP для акций Tesla на реальные и спрогнозированные')
    plt.ylabel('Коэффициент VWAP')
    plt.xlabel('Месяцы')
    plt.grid()
    plt.show()
    return forecasted_result


def plot_prediction(actual, prediction):
    plt.plot(actual, marker='.', label="Реальные значения")
    plt.plot(prediction, 'r', marker='.', label="Прогноз")
    plt.ylabel('Показатель VWAP')
    plt.xlabel('Время')
    plt.legend()
    plt.show()
