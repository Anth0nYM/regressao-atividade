import statistics
class SimpleLinearRegression:
    def __init__(self, b0=None, b1=None):
        self.b0 = b0
        self.b1 = b1

    def fit(self, feature, target):
        if len(feature) != len(target):
            raise ValueError("feature and target must have the same length")

        n = len(feature)
        feature_mean = statistics.mean(feature)
        target_mean = statistics.mean(target)
        numerator = sum([(feature[i] - feature_mean) *
                        (target[i] - target_mean) for i in range(n)])
        denominator = sum([(feature[i] - feature_mean)**2 for i in range(n)])
        b1 = numerator/denominator
        b0 = target_mean - b1 * feature_mean
        self.b0 = b0
        self.b1 = b1
        return True if b0 and b1 else False

    def predict(self, feature_to_predict):
        if self.b0 is None or self.b1 is None:
            raise ValueError("Model not fitted")
        targets_predicted = []
        for f in feature_to_predict:
            target_predicted = f * self.b1 + self.b0
            targets_predicted.append(target_predicted)
            
        return targets_predicted
    
    def get_coefficients(self): return self.b0, self.b1