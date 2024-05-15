from pre_processing.pre_processing import pre_processing
from regression.simple import SimpleLinearRegression
from visualization.plot import linear_regression_plot
from metrics import rmse


X_train_processed, X_test_processed, y_train_processed, y_test_processed = pre_processing()

x_train = [elemento for sublista in X_train_processed for elemento in sublista]
y_train = [elemento for sublista in y_train_processed for elemento in sublista]
x_test = [elemento for sublista in X_test_processed for elemento in sublista]
y_test = [elemento for sublista in y_test_processed for elemento in sublista]

model = SimpleLinearRegression()
model.fit(x_train, y_train)
pred = model.predict(x_test)
print(f"Random Mean Square Error: {rmse.rmse(y_test, pred)}")
linear_regression_plot(y_test, x_test, pred , model.b0, model.b1)
