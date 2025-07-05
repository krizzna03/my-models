import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math

x = np.array([[2],[4],[6],[8],[10]])
y = np.array([500,1000,1500,1800,2000])

model = LinearRegression()
model.fit(x,y)
y_pred = model.predict(x)

mae = mean_absolute_error(y,y_pred)
print("The mean absolute error: ", mae)

mse = mean_squared_error(y,y_pred)
print("The mean squared error: ", mse)

r2 = r2_score(y,y_pred)
print("The R-Squared Score is: ", r2)

n = len(y)
p = x.shape[1]
adjusted_r2 = 1-(1-r2)*((n-1) / (n - p - 1))
print("The adjusted R-Squared is : ", adjusted_r2)

rmse = math.sqrt(mse)
print("The RMSE is: ", rmse)

hours_worked = np.array([[7]])
salary = model.predict(hours_worked)
print(salary[0])

plt.scatter(x, y, color= 'blue', label='Actual data')
plt.plot(x, model.predict(x), color='green', label='Regression Line')
plt.scatter(7, salary, color='red', label='Prediction')
plt.title("Linear regression: Freelancer Income Prediction")
plt.xlabel("Hours worked")
plt.ylabel("Income in rs")
plt.legend()
plt.show()