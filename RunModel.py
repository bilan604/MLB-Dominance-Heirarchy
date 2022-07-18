import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer
from numpy import unique
from numpy import sqrt, dot, array, diagonal, mean, transpose
from numpy import transpose, diag, dot
from numpy.linalg import svd, inv, qr
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor


# Might need to change filepath
df = pd.read_csv("heart_disease_health_indicators_BRFSS2015.csv")


def check(vec):
    if type(vec[0]) not in (int, float, np.float64):
        return np.ravel(vec)
    return vec


def accuracy(Y_pr, Y_ts):
    Y_pr, Y_ts = check(Y_pr), check(Y_ts)
    return sum([1 if pr == ts else 0 for pr,ts in zip(Y_pr, Y_ts)]) / len(Y_pr)


def confusion(Y_pr, Y_ts):
    Y_pr, Y_ts = check(Y_pr), check(Y_ts)
    TP, FP, TN, FN = 0, 0, 0, 0
    dd = {(1,1):TP, (1,0):FP, (0,1):FN, (0,0):TN}
    for pr, ts in zip(Y_pr, Y_ts):
        dd[(pr,ts)] += 1
    return {"TP": dd[(1,1)], "FP": dd[(1,0)], "FN": dd[(0,1)], "TN": dd[(0,0)]}


def convertToBinary(Y_pr, cuttoffPercentile=85):
    cutoff = np.percentile(Y_pr, cuttoffPercentile)
    return [1.0 if pr > cutoff else 0.0 for pr in Y_pr]


def runModel(model, __X, __Y, cuttoffPercentile=90):
    __X_train, __X_test, __Y_train, __Y_test = train_test_split(__X, __Y, test_size=0.30, random_state=1)
    model.fit(__X_train, np.ravel(__Y_train))
    modelPred = model.predict(__X_test)
    if any([0.000001 < pr < 0.99999 or pr < -0.000001 for pr in modelPred]):
        modelPred = convertToBinary(modelPred, cuttoffPercentile)
    print("Accuracy:", accuracy(modelPred, __Y_test))
    print("MSE:", mean_squared_error(modelPred, __Y_test))
    print("Confusion:", confusion(modelPred, __Y_test))
    return model


def runFittedModel(model, __X, __Y, cuttoffPercentile=90):
    __X_train, __X_test, __Y_train, __Y_test = train_test_split(__X, __Y, test_size=0.30, random_state=1)
    modelPred = model.predict(__X_test)
    if any([0.000001 < pr < 0.99999 or pr < -0.000001 for pr in modelPred]):
        modelPred = convertToBinary(modelPred, cuttoffPercentile)
    conf = confusion(modelPred, __Y_test)
    sens = conf["TP"] / (conf["TP"] + conf["FN"])
    spec = conf["TN"] / (conf["FP"] + conf["TN"])
    print(f"{model=}")
    print(f"{accuracy(modelPred, __Y_test)=}")
    print(f"{mean_squared_error(modelPred, __Y_test)=}")
    print(f"Confusion: {conf}\nSensitivity: {sens}")
    print(f"Specificity: {spec}")
    return model

random.seed(1)

colNames = df.columns

dfX = df.copy().drop(columns=["HeartDiseaseorAttack"])
dfY = df["HeartDiseaseorAttack"]

X = array(dfX)
Y = array(dfY).reshape(len(dfY), 1)

# Optional Global train-test-split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=1)


encoder = OneHotEncoder()
X_enc = encoder.fit(X).transform(X).toarray()


# Running a model
"""
GBR_NE460_MD3 = GradientBoostingRegressor(n_estimators=450, max_depth=3, random_state=1)
GBR_NE460_MD3 = runModel(GBR_NE460_MD3, X_enc, Y)
"""









