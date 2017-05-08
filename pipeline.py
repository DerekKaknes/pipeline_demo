from __future__ import print_function, division

from sklearn.externals import joblib
from sklearn.datasets import load_iris
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
# Need wrapper for estimators/pipelines
from model_transformer import ModelTransformer

def main():
    random_seed = 43

    # Load Iris data set and shuffle rows
    X, y = load_iris(True)
    X, y = shuffle(X, y, random_state=random_seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
            random_state=random_seed)


    # Train Logistic Regression "Sub-Model"
    lr_clf = LogisticRegression(random_state = random_seed)
    lr_pipeline = Pipeline([("normalize", Normalizer()), ("lr", lr_clf)])
    lr_model_feature = ModelTransformer(lr_pipeline)

    # Add LR Submodel as a Feature to a SVM pipeline
    feature_union = FeatureUnion([
        ("normalize", Normalizer()),
        ("lr_model", lr_model_feature),
        ])

    # Create SVM Pipeline
    svm_clf = SVC(kernel="linear")
    svm_pipeline = Pipeline([
        ("features", feature_union),
        ("svm", svm_clf)
    ])

    # Fit Pipeline
    svm_pipeline.fit(X_train, y_train)

    # Compare that to just fitting a stand-alone SVM (w/o LR Model as a feature)
    svm = SVC(kernel="linear")
    svm.fit(X_train, y_train)

    # Build an Ensemble model with grid search params
    ## Note that parameters to a pipeline use <step_name>__<key>: value syntax
    params = {"ensemble__loss": ["deviance"],
            "ensemble__learning_rate": [0.01, 0.1, 0.5],
            "ensemble__n_estimators": [100, 150, 200],
            "ensemble__max_depth": [3, 4, 5]
            }

    ens_model = GradientBoostingClassifier(random_state=random_seed)


    # Ensemble Pipeline
    ensemble = Pipeline([
        ("normalize", Normalizer()),
        ("submodels", FeatureUnion([
            ("lr",
                ModelTransformer(LogisticRegression(random_state=random_seed))),
            ("svm", ModelTransformer(SVC(random_state=random_seed)))
        ])),
        ("ensemble", ens_model)
        ])

    # Fit grid search using params and ensemble pipeline
    ens_grid = GridSearchCV(ensemble, params)
    ens_grid.fit(X_train, y_train)

    # Score test data through tuned ensemble grid
    score = ens_grid.score(X_test, y_test)

    # Print outcomes
    print("SVM-pipeline in-sample accuracy of {}".format(svm_pipeline.score(X_train,
        y_train)))
    print("SVM-standalone in-sample accuracy of {}".format(svm.score(X_train,
        y_train)))
    print("SVM-pipeline out-of-sample accuracy of {}".format(svm_pipeline.score(X_test,
        y_test)))
    print("SVM-standalone out-of-sample accuracy of {}".format(svm.score(X_test,
        y_test)))
    print("Ensemble out-of-sample score of {}".format(score))

    # Persist ensemble model/pipeline
    joblib.dump(ens_grid, "models/ens.pkl")

if __name__ == "__main__":
    main()
