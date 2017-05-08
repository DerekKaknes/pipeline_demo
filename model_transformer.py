import pandas as pd

from sklearn.base import TransformerMixin, BaseEstimator


class ModelTransformer(BaseEstimator, TransformerMixin):
    """This class is used to wrap estimators or pipelines so that they can be
    added to other pipelines"""

    def __init__(self, model):
        self.model = model

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
        return self

    def transform(self, X, **transform_params):
        return pd.DataFrame(self.model.predict(X))
