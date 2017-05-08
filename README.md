# SKLearn Pipeline Demo
Demonstrates simple method to train, ensemble, parameter tune and persist
machine learning objects in sklearn.

## How To
1.  `pip install -r requirements.txt` to pull dependencies within `virtualenv`
2.  Execute `python pipeline.py` to train `iris` classifier and persist final
    ensemble grid to `./models/ens.pkl`
3.  Load ensemble grid using `python load.py` to load and re-score model (should
    produce same final output)

## To Do:
-  Determine best integration with `pyspark`.  `pyspark` has a `spark_sklearn`
   module with a `GridSearchCV` that seems to work occasionally, but can run
into trouble if it encounters ml algorithms that it doesn't like.
