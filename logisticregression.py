import numpy as np 
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

x = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]) #hunger level
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]) # 0 means no, 1 means yes(they will buy pizza)

model = LogisticRegression()
model.fit(x,y)

y_pred = model.predict(x)
probs= model.predict_proba(x)
print("predicted probs: ", probs[: ,1])

new_input = np.array([[7.5]])
new_pred = model.predict(new_input)
new_prob = model.predict_proba(new_input)[0][1]

print("\nPrediction for hunger = 7.5", new_pred[0])
print("Probability of buying pizza: ", new_prob )

hunger_range = np.linspace(1,10,300).reshape(-1,1)
sigmoid = model.predict_proba(hunger_range)[:,1]

plt.figure(figsize=(10,6))
plt.plot(hunger_range, sigmoid, label='sigmoid curve', color='blue')
plt.axhline(0.5, color="gray", linestyle="--", label="Threshold = 0.5")
plt.axvline(7.5, color="red", linestyle="--", label="Hunger = 7.5")
plt.scatter(new_input, new_prob, color="red", s=100, zorder=5)

plt.title("Logistic Regression - Sigmoid Curve")
plt.xlabel("Hunger Level")
plt.ylabel("Probability of Buying Pizza")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
