import pandas as pd
import mlflow
from sklearn.linear_model import LinearRegression

# Load data
data = pd.read_csv("Salary_Data.csv")

# Inputs
x = data[["YearsExperience"]]

# Labels
y = data[["Salary"]]

# Initialize model
model = LinearRegression()

# Train model
model.fit(x.values, y.values)

# Calculate coefficient of determination (R^2)
score = model.score(x.values, y.values)
print("Score:", score)

# Log the score
mlflow.log_metric("score", score)

# Log the model
mlflow.sklearn.log_model(model, "model")
