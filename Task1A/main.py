from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import math

TRAIN_PATH = "task1a_do4bq81me/train.csv"

lamdas = [0.1, 1, 10, 100, 200]

data = pd.read_csv(TRAIN_PATH)

X = data.drop('y', axis=1)
y = data['y']

for lamda in lamdas:   
    ridge_reg = Ridge(alpha=lamda, fit_intercept=False)
    scores = cross_val_score(ridge_reg, X, y, scoring="neg_root_mean_squared_error", cv=10)
    value = pd.DataFrame([-scores.mean()])
    value.to_csv('predictions1.csv', mode='a', index=False, header=False)