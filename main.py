from pre_processing.pre_processing import pre_processing_dataset
from regression.simple import SimpleLinearRegression
from visualization.plot import linear_regression_plot
from metrics import rmse

X_train_processed, X_test_processed, y_train_processed, y_test_processed = pre_processing_dataset()
x_train = X_train_processed
y_train = y_train_processed
x_test = X_test_processed
y_test = y_test_processed

model = SimpleLinearRegression()
model.fit(x_train, y_train)
pred = model.predict(x_test)
print(f"Random Mean Square Error: {rmse.rmse(y_test, pred)}")
linear_regression_plot(y_test, x_test, pred , model.b0, model.b1)
