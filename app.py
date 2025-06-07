from fastapi import FastAPI, Request
import joblib
import numpy as np
import pandas as pd

app = FastAPI()
model = joblib.load("house_price_model.pkl")


@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    df = pd.DataFrame([data])
    pred = model.predict(df)
    price_mxn = np.expm1(pred[0])
    return {"prediction": float(price_mxn)}
