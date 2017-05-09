from __future__ import print_function, division


import numpy as np
import pandas as pd

from sklearn.datasets import load_iris, load_digits
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


random_seed = 43
X, y = load_iris(True)
X, y = shuffle(X, y, random_state=random_seed)
X_train, X_test, y_train, y_test = train_test_split(X, y,
        random_state=random_seed)

iris = load_iris()
digits = load_digits()

df = pd.DataFrame(data = np.c_[iris.data, iris.target], columns =
        iris.feature_names + ["species"])

df2 = pd.DataFrame(data = np.ndarray([digits.data, digits.target]),
        columns = ["features"] + ["label"])
