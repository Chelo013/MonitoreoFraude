from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import os
import json

# --- 1. DEFINICIÓN DE LA API ---
app = FastAPI(title="API Jurado Antifraude (Paysim)", version="1.0")

# Habilitar CORS para permitir peticiones desde Vercel
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # En producción cambiar por la URL de Vercel
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- STATS GLOBALES (En memoria para la demo) ---
# Inicializamos con datos simulados para que el portafolio se vea activo
app_stats = {
    "total_requests": 1245,
    "models": {
        "isolation_forest": {"fraud": 42, "normal": 1203},
        "autoencoder": {"fraud": 39, "normal": 1206},
        "deep_svdd": {"fraud": 45, "normal": 1200}
    }
}


# --- 2. ESTRUCTURA DE LOS DATOS DE ENTRADA (JSON desde el Frontend) ---
class TransactionRequest(BaseModel):
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    type: str # Ej: "TRANSFER", "CASH_OUT", "CASH_IN", "PAYMENT", "DEBIT"

# Pesos del jurado configurables (ejemplo, se pueden enviar en la petición o dejar por defecto)
class PredictRequest(BaseModel):
    transaction: TransactionRequest
    weight_iforest: float = 0.33
    weight_autoencoder: float = 0.33
    weight_deep_svdd: float = 0.34
    threshold: float = 0.5 # Umbral para considerar que es fraude


# --- 3. CARGA DE MODELOS (Se ejecuta solo 1 vez al arrancar el servidor) ---
print("Cargando Modelos en Memoria...")

# Cargar Scaler
scaler = joblib.load(os.path.join("processed_data_paysim", "scaler.pkl"))

# Cargar Isolation Forest
iforest = joblib.load(os.path.join("iforest_model_paysim", "iforest_model.pkl"))

# Cargar Autoencoder
autoencoder = tf.keras.models.load_model(os.path.join("autoencoder_model_paysim", "autoencoder_paysim.h5"), compile=False)
ae_threshold = np.load(os.path.join("autoencoder_model_paysim", "ae_threshold.npy"))

# Cargar Deep SVDD
deep_svdd = tf.keras.models.load_model(os.path.join("deep_svdd_model_paysim", "deep_svdd_encoder.keras"), compile=False)
center_c = np.load(os.path.join("deep_svdd_model_paysim", "center_c.npy"))
svdd_threshold = np.load(os.path.join("deep_svdd_model_paysim", "threshold.npy"))

print("¡Modelos cargados exitosamente!")

# Variables categóricas conocidas del entrenamiento (para recrear One-Hot Encoding)
EXPECTED_TYPES = ['type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']

# --- 4. ENDPOINTS ---

@app.get("/stats")
async def get_stats():
    """Devuelve las estadísticas globales de uso de los modelos."""
    return app_stats

@app.get("/samples")
async def get_samples():
    """Devuelve ejemplos predefinidos de transacciones normales y fraudulentas."""
    try:
        with open('samples.json', 'r', encoding='utf-8') as f:
            samples = json.load(f)
        return samples
    except Exception as e:
        # Fallback if samples.json is not found
        return []

@app.post("/predict")
async def predict_fraud(request: PredictRequest):
    try:
        transaction = request.transaction
        
        # 1. Reconstruir la fila de datos en un DataFrame temporal
        input_data = {
            'amount': transaction.amount,
            'oldbalanceOrg': transaction.oldbalanceOrg,
            'newbalanceOrig': transaction.newbalanceOrig,
            'oldbalanceDest': transaction.oldbalanceDest,
            'newbalanceDest': transaction.newbalanceDest
        }
        
        # Inicializar todos los tipos en 0
        for t in EXPECTED_TYPES:
            input_data[t] = 0.0
            
        # Encender (1.0) el tipo de transacción que envió el usuario
        type_key = f"type_{transaction.type}"
        if type_key in input_data:
            input_data[type_key] = 1.0
            
        # Convertir a array (respetando el orden exacto de las columnas de entrenamiento)
        # Importante: El orden debe coincidir con como entrenó el preprocesamiento
        # Las numéricas primero, luego el OHE.
        X_raw = np.array([[
            input_data['amount'],
            input_data['oldbalanceOrg'],
            input_data['newbalanceOrig'],
            input_data['oldbalanceDest'],
            input_data['newbalanceDest'],
            input_data['type_CASH_IN'],
            input_data['type_CASH_OUT'],
            input_data['type_DEBIT'],
            input_data['type_PAYMENT'],
            input_data['type_TRANSFER']
        ]])
        
        # 2. Escalar los datos
        X_scaled = scaler.transform(X_raw)
        
        # 3. Predicciones de los Jueces
        # Juez 1: Isolation Forest
        pred_if_raw = iforest.predict(X_scaled)
        voto_if = 1 if pred_if_raw[0] == -1 else 0
        
        # Juez 2: Autoencoder
        reconstruction = autoencoder.predict(X_scaled, verbose=0)
        mse = np.mean(np.power(X_scaled - reconstruction, 2), axis=1)[0]
        voto_ae = 1 if mse > ae_threshold else 0
        
        # Juez 3: Deep SVDD
        z_point = deep_svdd.predict(X_scaled, verbose=0)
        distance = np.sum(np.square(z_point - center_c), axis=1)[0]
        voto_svdd = 1 if distance > svdd_threshold else 0
        
        # 4. Veredicto del Jurado (Con Pesos)
        score_final = (voto_if * request.weight_iforest) + \
                      (voto_ae * request.weight_autoencoder) + \
                      (voto_svdd * request.weight_deep_svdd)
                      
        es_fraude = bool(score_final >= request.threshold)
        
        # --- Actualizar Estadísticas Globales ---
        app_stats["total_requests"] += 1
        
        if voto_if == 1:
            app_stats["models"]["isolation_forest"]["fraud"] += 1
        else:
            app_stats["models"]["isolation_forest"]["normal"] += 1
            
        if voto_ae == 1:
            app_stats["models"]["autoencoder"]["fraud"] += 1
        else:
            app_stats["models"]["autoencoder"]["normal"] += 1
            
        if voto_svdd == 1:
            app_stats["models"]["deep_svdd"]["fraud"] += 1
        else:
            app_stats["models"]["deep_svdd"]["normal"] += 1
        
        # 5. Respuesta a Vercel
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
                "autoencoder_error": float(mse),
                "autoencoder_threshold": float(ae_threshold),
                "svdd_distance": float(distance),
                "svdd_threshold": float(svdd_threshold)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

