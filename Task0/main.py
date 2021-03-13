import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

TEST_PATH = "task0_sl19d1/test.csv"
TRAIN_PATH = "task0_sl19d1/train.csv"

#Get data
train_data = pd.read_csv(TRAIN_PATH)
test_data = pd.read_csv(TEST_PATH)

train_data.head()

#Prepare data
train_set, test_set = train_test_split(train_data, test_size=0.2, random_state=42)

X_train = train_set.drop('y', axis=1)
y_train = train_set['y']

X_test = test_set.drop('y', axis=1)
y_test = test_set['y']

#Train model
model = LinearRegression()
model.fit(X_train, y_train)

#Assess performance
predictions = model.predict(X_test)

MSE = mean_squared_error(y_test, predictions)
print("Mean Squared Error: {}".format(MSE))

MAE = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error: {}".format(MAE))

#Predicting test set
final_predictions = model.predict(test_data)

#Join Id's and predictions
final = np.concatenate((final_predictions, test_data['Id']), axis=0)
final = np.reshape(final, (int(len(final)/2), 2), order='F')

final[:,[0, 1]] = final[:,[1, 0]]
np.savetxt("predictions.csv", final, delimiter=",")