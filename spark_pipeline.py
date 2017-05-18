from __future__ import print_function, division


import numpy as np
import pandas as pd

from sklearn.datasets import load_iris, load_digits
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression, MultilayerPerceptronClassifier
from pyspark.ml.feature import Normalizer, VectorAssembler, StringIndexer,\
IndexToString, Binarizer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def run(spark):
    random_seed = 43

    digits = load_digits()
    data = pd.DataFrame(data = digits.data)

    label = "label"
    labels = map(str, digits.target_names)
    features = map(lambda x: "pixel{}".format(x), data.columns)
    data.columns = features
    layers = [len(features), 25, 10, len(labels)]

    data["label"] = pd.Series(digits.target)

    df = spark.createDataFrame(data)
    train, test = df.randomSplit([0.8, 0.2], seed=random_seed)

    assembler = VectorAssembler()\
            .setInputCols(features)

    binarizer = Binarizer()\
            .setInputCol(assembler.getOutputCol())\
            .setThreshold(3.)

    stringIndexer = StringIndexer()\
            .setInputCol(label)
    stringIndexerModel = stringIndexer.fit(train)

    nnet = MultilayerPerceptronClassifier()\
            .setLabelCol(stringIndexer.getOutputCol())\
            .setFeaturesCol(binarizer.getOutputCol())\
            .setLayers(layers)\
            .setSeed(random_seed)\
            .setBlockSize(128)\
            .setMaxIter(100)\
            .setTol(1e-7)

    indexToString = IndexToString()\
            .setInputCol(nnet.getPredictionCol())\
            .setLabels(stringIndexerModel.labels)

    pipeline = Pipeline()\
            .setStages([assembler, binarizer, stringIndexer, nnet, indexToString])

    ev = MulticlassClassificationEvaluator()\
            .setLabelCol(stringIndexer.getOutputCol())\
            .setPredictionCol(nnet.getPredictionCol())\
            .setMetricName("accuracy")

    model = pipeline.fit(train)
    result = model.transform(test)
    precision = ev.evaluate(result)
    print("Precision = {}".format(precision))




if __name__ == "__main__":
    sc = SparkContext("local")
    spark = SparkSession(sc)
    run(spark)
