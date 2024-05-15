import pandas as pd

def pre_processing_dataset ():
    weatherHistory = pd.read_csv('pre_processing/weatherHistory.csv')
    x = weatherHistory['Temperature (C)'].values.reshape(-1,1)
    y = weatherHistory['Humidity'].values.reshape(-1,1)
    X_train = x[:int(len(x)*0.8)] 
    X_test = x[int(len(x)*0.8):] 
    y_train = y[:int(len(y)*0.8)] 
    y_test = y[int(len(y)*0.8):]
    return X_train, X_test, y_train, y_test




  
