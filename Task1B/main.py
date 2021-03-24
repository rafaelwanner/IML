import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score

data = pd.read_csv("task1b_ql4jfi6af0/train.csv")

X = data.drop(['Id', 'y'], axis=1)
y = data['y']

#Computing linear features
lin_features = X.copy()
lin_features.columns = ['phi1', 'phi2', 'phi3', 'phi4', 'phi5']
#Computing quadradic features
quad_features = X.pow(2)
quad_features.columns = ['phi6', 'phi7', 'phi8', 'phi9', 'phi10']
#Computing exponential features
exp_features = X.apply(np.exp)
exp_features.columns = ['phi11', 'phi12', 'phi13', 'phi14', 'phi15']
#Computing trigonometric features
trig_features = X.apply(np.cos)
trig_features.columns = ['phi16', 'phi17', 'phi18', 'phi19', 'phi20']
#Computing constant features
const_features = pd.DataFrame(np.ones(len(X)), columns=['phi21'])

features = pd.concat([lin_features, quad_features, exp_features, trig_features, const_features], axis=1)

lamdas = np.arange(0.1, 10, 0.1).tolist()
scores = []

for ridge_coef in lamdas:
    ridge_reg = Ridge(alpha=ridge_coef)
    score = cross_val_score(ridge_reg, features, y, scoring="neg_root_mean_squared_error", cv=5)
    scores.append((ridge_coef, -score.mean()))

ridge_reg = Ridge(alpha=10)
ridge_reg.fit(features, y)
weights = pd.DataFrame(ridge_reg.coef_)
weights.to_csv("predictions.csv", header=False, index=False)
