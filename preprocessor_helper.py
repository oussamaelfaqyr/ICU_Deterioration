import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, keep_idx=None):
        self.keep_idx = keep_idx

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.keep_idx is None:
            return X
        # Handle pandas dataframe or numpy array
        if hasattr(X, "iloc"):
            return X.iloc[:, self.keep_idx]
        return X[:, self.keep_idx]

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        return np.array(input_features)[self.keep_idx]
