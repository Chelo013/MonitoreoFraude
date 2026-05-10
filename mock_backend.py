from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random
import time
import json

app = FastAPI(title="API Jurado Antifraude (Paysim) MOCK")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app_stats = {
    "total_requests": 1245,
    "models": {
        "isolation_forest": {"fraud": 42, "normal": 1203},
        "autoencoder": {"fraud": 39, "normal": 1206},
        "deep_svdd": {"fraud": 45, "normal": 1200}
    }
}


class TransactionRequest(BaseModel):
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    type: str

class PredictRequest(BaseModel):
    transaction: TransactionRequest
    weight_iforest: float = 0.33
    weight_autoencoder: float = 0.33
    weight_deep_svdd: float = 0.34
    threshold: float = 0.5

@app.get("/stats")
async def get_stats():
    return app_stats

@app.get("/samples")
async def get_samples():
    try:
        with open('samples.json', 'r', encoding='utf-8') as f:
            samples = json.load(f)
        return samples
    except Exception as e:
        return []

@app.post("/predict")
async def predict_fraud(request: PredictRequest):
    time.sleep(1.5) # Simular carga
    
    # Mockear predicción dependiendo del amount
    voto_if = 1 if request.transaction.amount > 10000 else 0
    voto_ae = 1 if request.transaction.amount > 50000 else 0
    voto_svdd = 1 if request.transaction.amount > 100000 else 0
    
    # Para el ejemplo 1 (181.0) forzamos que sea fraude
    if request.transaction.amount == 181.0:
        voto_if, voto_ae, voto_svdd = 1, 1, 1
        
    score_final = (voto_if * request.weight_iforest) + \
                  (voto_ae * request.weight_autoencoder) + \
                  (voto_svdd * request.weight_deep_svdd)
                  
    es_fraude = bool(score_final >= request.threshold)
    
    app_stats["total_requests"] += 1
    app_stats["models"]["isolation_forest"]["fraud"] += 1 if voto_if == 1 else 0
    app_stats["models"]["isolation_forest"]["normal"] += 1 if voto_if == 0 else 0
    app_stats["models"]["autoencoder"]["fraud"] += 1 if voto_ae == 1 else 0
    app_stats["models"]["autoencoder"]["normal"] += 1 if voto_ae == 0 else 0
    app_stats["models"]["deep_svdd"]["fraud"] += 1 if voto_svdd == 1 else 0
    app_stats["models"]["deep_svdd"]["normal"] += 1 if voto_svdd == 0 else 0
    
    return {
        "status": "success",
        "verdict_is_fraud": es_fraude,
        "score_final": float(score_final),
        "jury_votes": {
            "isolation_forest": voto_if,
            "autoencoder": voto_ae,
            "deep_svdd": voto_svdd
        },
        "applied_weights": {
            "isolation_forest": request.weight_iforest,
            "autoencoder": request.weight_autoencoder,
            "deep_svdd": request.weight_deep_svdd
        },
        "threshold": request.threshold,
        "metrics": {
            "autoencoder_error": 0.0456 if voto_ae else 0.0123,
            "autoencoder_threshold": 0.03,
            "svdd_distance": 1.5 if voto_svdd else 0.4,
            "svdd_threshold": 0.8
        }
    }
