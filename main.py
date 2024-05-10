from regression.simple import SimpleLinearRegression
from visualization.plot import linear_regression_plot
from metrics import rmse
x_train = [1, 2, 3, 4, 5]
y_train = [2, 3, 4, 5, 6]

x_test = [6, 7, 8, 9, 10]
y_test = [7, 8, 9, 10, 11]

model = SimpleLinearRegression()
model.fit(x_train, y_train)
pred = model.predict(x_test)
print(f"Random Mean Square Error: {rmse.rmse(y_test, pred)}")
linear_regression_plot(y_test, x_test, pred , model.b0, model.b1)

