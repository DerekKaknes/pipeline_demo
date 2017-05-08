from __future__ import print_function, division


from sklearn.externals import joblib
from sklearn.datasets import load_iris
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV
# Need wrapper for estimators/pipelines
from model_transformer import ModelTransformer

def main():
    mod = joblib.load("models/ens.pkl")
    random_seed = 43

    # Load Iris data set and shuffle rows
    X, y = load_iris(True)
    X, y = shuffle(X, y, random_state=random_seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
            random_state=random_seed)

    print("Model Predictions for X_test")
    print(mod.predict(X_test))
    print("Model score for X_test = {}".format(mod.score(X_test, y_test)))

if __name__ == "__main__":
    main()
