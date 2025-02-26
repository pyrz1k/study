"""Module providing a function printing python version."""
#  Создаем fastapi приложение
from fastapi import FastAPI, File, UploadFile
import pandas as pd
import joblib
from io import BytesIO

app = FastAPI()

# Загрузка обученной модели
model_path = "/Users/ivanscherbakov/Documents/ВУЗ/ИТиП/ЛАБ1/Laptop_price_model.pkl"
model = joblib.load(model_path)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(BytesIO(content))
    predictions = model.predict(df)
    return {"predictions": predictions.tolist()}

# Запускаем FastApi
from pyngrok import ngrok

# Подключаем публичный URL
public_url = ngrok.connect(8000)
print("API доступно по адресу:", public_url)