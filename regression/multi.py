import numpy as np

class MultipleLinearRegressionGD:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.beta = None
    
    def fit(self, X, y):
        # Converte X e y para arrays numpy
        X = np.array(X)
        y = np.array(y)
        
        # Adiciona uma coluna de 1s à matriz X para o intercepto
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        m, n = X_b.shape
        self.beta = np.zeros(n)
         
        # Gradiente descendente
        for _ in range(self.n_iterations):
            print('X_b.shape:',X_b.shape)
            print('self.beta:',self.beta.shape)
            gradients = -2/m * X_b.T.dot(y - X_b.dot(self.beta))

            self.beta -= self.learning_rate * gradients
    
    def predict(self, X):
        # Converte X para um array numpy
        X = np.array(X)
        
        # Adiciona uma coluna de 1s à matriz X para o intercepto
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Calcula as previsões
        return X_b.dot(self.beta)
    
    def get_coefficients(self):
        return self.beta
