from regression.simple import SimpleLinearRegression
from regression.multi import MultipleLinearRegressionGD 
from visualization.plot import simple_linear_regression_plot, multiple_linear_regression_plot
from metrics import rmse
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data.csv")

#REGRESSÃO LINEAR SIMPLES
# X = df['Humidity').values
# y = df['Temperature (C)'].values
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# model = SimpleLinearRegression()
# model.fit(X_train, y_train)
# pred = model.predict(X_test)
# simple_linear_regression_plot(y_test, X_test, pred , model.b0, model.b1)

#REGRESSÃO LINEAR MÚLTIPLA
# X = df.drop(columns=['Temperature (C)']).values
# y = df['Temperature (C)'].values
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# model = MultipleLinearRegressionGD()
# model.fit(X_train, y_train)
# pred = model.predict(X_test)
# print(f"Random Mean Square Error: {rmse.rmse(y_test, pred)}")
# multiple_linear_regression_plot(y_test, X_test, pred, model.beta)

#print(f"Random Mean Square Error: {rmse.rmse(y_test, pred)}")

#OBS NA REGRESSÃO LINEAR SIMPLES, O DATASET NÃO FOI TRATADO 
# PORQUE AS DUAS COLUNAS UTILIZADAS (TEMPERATURA E HUMIDADE) NÃO POSSUEM VALORES NULOS OU NÃO NUMÉRICOS