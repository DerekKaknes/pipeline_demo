from __future__ import print_function, division


import numpy as np
import pandas as pd

from sklearn.datasets import load_iris, load_digits
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

digits = load_digits()
df2 = pd.DataFrame(data = digits.data)
