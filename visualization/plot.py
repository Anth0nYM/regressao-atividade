import matplotlib.pyplot as plt
import numpy as np

def simple_linear_regression_plot(true_target, pred_feature, pred_target, b0, b1):
    # True values
    plt.scatter(pred_feature, true_target, color='green', label='True values')
    
    # Predicted values
    # plt.scatter(pred_feature, pred_target, color='blue', label='Predicted values')

    # Regression line
    y_values = [b1 * x + b0 for x in pred_feature]
    plt.plot(pred_feature, y_values, color='red', label='Line')

    # Labels and title
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Linear Regression')

    plt.show()
    
def multiple_linear_regression_plot(true_target, X, pred_target, betas):
    # Coletando os valores das variáveis independentes
    x1 = X[:, 0]
    x2 = X[:, 1]

    # Plotando os valores reais
    plt.scatter(x1, true_target, color='green', label='True values')

    # Plotando a reta de regressão
    plt.plot(x1, pred_target, color='red', label='Regression line')

    # Labels e título
    plt.xlabel('X1')
    plt.ylabel('Y')
    plt.title('Multiple Linear Regression')
    plt.legend()

    plt.show()