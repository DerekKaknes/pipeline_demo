from __future__ import print_function, division


import numpy as np
import pandas as pd

from sklearn.datasets import load_iris, load_digits
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import Normalizer, VectorAssembler

def run(spark):
    random_seed = 43

    iris = load_iris()
    digits = load_digits()

    df = pd.DataFrame(data = np.c_[iris.data, iris.target], columns =
            iris.feature_names + ["label"])

    df2 = pd.DataFrame(data = digits.data)
    df2["label"] = pd.Series(digits.target)

    data = spark.createDataFrame(df2)
    train, test = data.randomSplit([0.8, 0.2], seed=random_seed)

    assembler = VectorAssembler(
            inputCols = map(str, range(64)),
            outputCol = "features")


    lr = LogisticRegression()
    pipeline = Pipeline(stages = [assembler, lr])

    model = pipeline.fit(train)
    pred = model.transform(test)
    pred.select("features", "label", "prediction").show()


if __name__ == "__main__":
    sc = SparkContext("local")
    spark = SparkSession(sc)
    run(spark)
