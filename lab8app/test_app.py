# lab8app/app.py

from fastapi import FastAPI, Request
import mlflow
import mlflow.pyfunc
import uvicorn

app = FastAPI()

# Optional but good practice — set tracking URI
mlflow.set_tracking_uri("https://mlflow-tracking-174134742093.us-west2.run.app")
mlflow.set_experiment("lab8-experiment")

# Load model using full artifact URI
MODEL_URI = "mlflow-artifacts:/6/9becda7e7f2c458989dd6235cf570924/artifacts/metaflow_train"

model = mlflow.pyfunc.load_model(MODEL_URI)

@app.post("/predict")
async def predict(request: Request):
    payload = await request.json()
    input_data = payload.get("data")
    prediction = model.predict(input_data)
    return {"prediction": prediction.tolist()}

# Run with:
# uvicorn test_app:app --reload


# curl -X POST http://127.0.0.1:8000/predict \
#   -H "Content-Type: application/json" \
#   -d '{"data": [[0.0381, 0.0507, 0.0617, 0.0219, -0.0442, -0.0348, -0.0434, -0.0026, 0.0199, -0.0176]]}'

