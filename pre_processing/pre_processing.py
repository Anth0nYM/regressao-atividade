import pandas as pd
from sklearn.preprocessing import StandardScaler

def pre_processing():
    weatherHistory = pd.read_csv('pre_processing/data.csv')

    # Remover linhas com valores não numéricos
    weatherHistory = weatherHistory.dropna()

    # Normalizar os dados
    scaler = StandardScaler()
    x = scaler.fit_transform(weatherHistory['Temperature (C)'].values.reshape(-1,1))
    y = scaler.fit_transform(weatherHistory['Humidity'].values.reshape(-1,1))

    # Dividir os dados em conjunto de treinamento e teste
    X_train = x[:int(len(x)*0.8)] 
    X_test = x[int(len(x)*0.8):] 
    y_train = y[:int(len(y)*0.8)] 
    y_test = y[int(len(y)*0.8):]

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = pre_processing()
