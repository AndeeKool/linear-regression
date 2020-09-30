import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

#print(diabetes_X)
# print(diabetes_y)

diabetes_X = diabetes_X[:, np.newaxis, 2]

#Separate the data in 2:
# 1-> Training data (So algorithm learns)
# 2-> Test data (evaluate algorithm)

#From the beginning to the (final - 20)
diabetes_X_train = diabetes_X[:-20]
#From (final -20) to the final
diabetes_X_test = diabetes_X[-20:]

diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

#Elements must match training vs test

#Create algorithm object
regr = linear_model.LinearRegression()

#train model (with training data)
# fit() -> training; asks for two data
# 1- Features (info that helps predicting)
# 2- Labels (answers)
regr.fit(diabetes_X_train, diabetes_y_train)

# akgorithm has been trained, so now can predict

# create a variable with the prediction "answers" with the data that has trained it
# Use test data (evaluation) to do predictions
# We can't use the same data, as it will do a perfect match
diabetes_y_pred = regr.predict(diabetes_X_test)

#print(diabetes_y_test[0])
#print(diabetes_y_pred[0])

error = mean_squared_error(diabetes_y_test, diabetes_y_pred)
print(f"Mean squared error: {error}")

score = r2_score(diabetes_y_test, diabetes_y_pred)
print(f"Coefficient of determination: {score}")


#visually show results (using graphics)
plt.scatter(diabetes_X_test, diabetes_y_test, color='black')

plt.scatter(diabetes_X_train, diabetes_y_train, color='pink')

#Add regression line
plt.plot(diabetes_X_test, diabetes_y_pred, color='red', linewidth=3)

plt.show()