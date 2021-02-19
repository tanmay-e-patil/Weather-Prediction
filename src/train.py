"""
The main executable.

Written by Tanmay Patil
"""
from pandas.core.algorithms import mode
from model import create_model, train_model
from preprocess import object_to_date_time, get_daily_weather_data, normalize_data, read_csv, get_training_data
import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd



if __name__ == "__main__":
    df = read_csv("../data/processed/delhi_weather_data_processed.csv")
    df = object_to_date_time(df)
    daily_weather = get_daily_weather_data(df)
    daily_weather = normalize_data(daily_weather)
    X,y = get_training_data(daily_weather)
    X_train = X[:7300,::]
    X_test = X[7300:,::]
    y_train = y[:7300]
    y_test = y[7300:]
    model = create_model()
    history = train_model(model, X_train, y_train)
    train_loss = history.history['loss']
    x_axis = [i for i in range(1, len(train_loss + 1))]
    plt.title('Training Loss')
    plt.plot(x_axis, train_loss)
    plt.show()


