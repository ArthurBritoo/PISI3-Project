import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

class PropertyFeatures(BaseModel):
    area_construida: float
    area_terreno: float
    ano_construcao: int
    padrao_acabamento: str
    cluster: int
    bairro: str
    tipo_imovel: str

# Carregamento global (executado uma vez por instância serverless)
try:
    MODEL_PATH = "property_classifier_model_optimized2.joblib"
    pipeline = joblib.load(MODEL_PATH)
except Exception as e:
    pipeline = None
    print("Erro ao carregar modelo:", e)

app = FastAPI()

@app.get("/")
async def root():
    return {"status": "ok"}

@app.post("/predict")
async def predict(features: PropertyFeatures):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado")

    try:
        df = pd.DataFrame([features.dict()])
        feature_order = [
            "area_construida", "area_terreno", "ano_construcao",
            "padrao_acabamento", "cluster", "bairro", "tipo_imovel"
        ]
        df = df[feature_order]

        y_pred = pipeline.predict(df)
        return {"predicted_category": y_pred[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
