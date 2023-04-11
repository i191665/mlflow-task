import mlflow

model = mlflow.sklearn.load_model(
    "mlruns/0/aa7c3f8e614d4092bc3df6584277176d/artifacts/model"
)

years = 3
prediction = model.predict([[years]]).item()
print("Prediction:", prediction)
