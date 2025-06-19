from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # 👈 Allow any origin
    allow_credentials=True,
    allow_methods=["*"],            # 👈 Allow all HTTP methods
    allow_headers=["*"],            # 👈 Allow all headers
)
model = joblib.load("house_price_model.pkl")


@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    df = pd.DataFrame([data])
    pred = model.predict(df)
    price_mxn = np.expm1(pred[0])
    return {"prediction": float(price_mxn)}
