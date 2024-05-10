import matplotlib.pyplot as plt

def linear_regression_plot(true_target, pred_feature, pred_target, b0, b1):
    # True values
    plt.scatter(pred_feature, true_target, color='green', label='True values')
    
    # Predicted values
    plt.scatter(pred_feature, pred_target, color='blue', label='Predicted values')

    # Regression line
    y_values = [b1 * x + b0 for x in pred_feature]
    plt.plot(pred_feature, y_values, color='red', label='Line')

    # Labels and title
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Linear Regression')

    plt.show()