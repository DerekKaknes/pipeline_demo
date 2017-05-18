# SKLearn Pipeline Demo
Demonstrates simple method to train, ensemble, parameter tune and persist
machine learning objects in sklearn.

## How To
1.  `pip install -r requirements.txt` to pull dependencies within `virtualenv`
2.  Execute `python pipeline.py` to train `iris` classifier and persist final
    ensemble grid to `./models/ens.pkl`
3.  Load ensemble grid using `python load.py` to load and re-score model (should
    produce same final output)
4.  Execute `spark_pipeline.py` using `spark-submit spark_pipeline.py`

## To Do:
-  Determine how to best store "sub-workflow" Pipelines that can be integrated
   into more complex workflows (i.e. other Pipelines).  Maybe store them in
separate module so they can be imported into new scripts?
-  Explore ways to pass column names to `VectorAssembler` in `pyspark.Pipeline`
-  Explore ability to execute `pyspark.Pipeline` in parallel, test performance
   verse manual thread control
