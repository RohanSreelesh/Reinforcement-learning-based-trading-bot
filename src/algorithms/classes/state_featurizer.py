from sklearn.preprocessing import StandardScaler


class StateFeaturizer:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, data):
        self.scaler.fit(data)

    def transform(self, state):
        return self.scaler.transform(state)
