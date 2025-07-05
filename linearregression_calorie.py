import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression 

x = np.array([[20],[30],[40],[50],[60]])
y = np.array([180, 250, 310, 360, 400])

model = LinearRegression()
model.fit(x,y)

future_time = np.array([[45]])
pre_calorie = model.predict(future_time)
print("Calories burnt if you worked out 45 mins: ", pre_calorie[0])

plt.scatter(x,y, color='blue', label="Actual data")
plt.plot(x, model.predict(x), color='green', label='Regression Line')
plt.scatter(45, pre_calorie, color='red', label= 'Prediction')
plt.title("Linear Regression: Calorie burn")
plt.xlabel("Workout Duration")
plt.ylabel("Calorie burnt")
plt.legend()
plt.show()