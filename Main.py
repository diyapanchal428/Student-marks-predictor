import pandas as pd
from sklearn.linear_model import LinearRegression
data = {
    "hours": [1, 2, 3, 4, 5],
    "marks": [30, 40, 50, 60, 70]
}
df = pd.DataFrame(data)

X = df[["hours"]]
y = df["marks"]

model = LinearRegression()
model.fit(X, y)

prediction = model.predict([[6]])
print(f"Predicted Marks for 6 hours of study: {prediction[0]}")
