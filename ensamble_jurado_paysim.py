import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import os

DATA_DIR = "processed_data_paysim"

# 1. Cargar datos de Test
print("1. Cargando datos de Test...")
try:
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
except FileNotFoundError:
    print("ERROR: No se encontraron los datos de Test. Ejecuta preprocess_paysim.py")
    exit()

# --- MODELO 1: ISOLATION FOREST ---
print("2. Cargando y prediciendo con Isolation Forest...")
try:
    iforest = joblib.load(os.path.join("iforest_model_paysim", "iforest_model.pkl"))
    pred_if_raw = iforest.predict(X_test)
    pred_if = np.where(pred_if_raw == -1, 1, 0)
except Exception as e:
    print(f"Error cargando Isolation Forest: {e}. Asegúrate de haberlo entrenado.")
    pred_if = np.zeros_like(y_test) # Voto nulo si falla

# --- MODELO 2: AUTOENCODER ---
print("3. Cargando y prediciendo con Autoencoder...")
try:
    autoencoder = tf.keras.models.load_model(os.path.join("autoencoder_model_paysim", "autoencoder_paysim.h5"), compile=False)
    ae_threshold = np.load(os.path.join("autoencoder_model_paysim", "ae_threshold.npy"))
    X_test_pred_ae = autoencoder.predict(X_test, verbose=0)
    mse_test = np.mean(np.power(X_test - X_test_pred_ae, 2), axis=1)
    pred_ae = (mse_test > ae_threshold).astype(int)
except Exception as e:
    print(f"Error cargando Autoencoder: {e}. Asegúrate de haberlo entrenado.")
    pred_ae = np.zeros_like(y_test)

# --- MODELO 3: DEEP SVDD ---
print("4. Cargando y prediciendo con Deep SVDD...")
try:
    # compile=False es crucial porque usa una función de pérdida personalizada que no necesitamos para predecir
    deep_svdd = tf.keras.models.load_model(os.path.join("deep_svdd_model_paysim", "deep_svdd_encoder.keras"), compile=False)
    center_c = np.load(os.path.join("deep_svdd_model_paysim", "center_c.npy"))
    svdd_threshold = np.load(os.path.join("deep_svdd_model_paysim", "threshold.npy"))
    
    Z_test = deep_svdd.predict(X_test, verbose=0)
    distances_test = np.sum(np.square(Z_test - center_c), axis=1)
    pred_svdd = (distances_test > svdd_threshold).astype(int)
except Exception as e:
    print(f"Error cargando Deep SVDD: {e}. Asegúrate de haberlo entrenado.")
    pred_svdd = np.zeros_like(y_test)

# --- SISTEMA DE JURADO (VOTACIÓN ENSEMBLE) ---
print("\n5. Uniendo los votos del Jurado...")
# Cada modelo aporta un 1 (Fraude) o un 0 (Normal). 
# El máximo puntaje es 3 (unanimidad)
votos_totales = pred_if + pred_ae + pred_svdd

# Estrategia A: Mayoría Simple (2 de 3 modelos de acuerdo)
pred_jurado_mayoria = (votos_totales >= 2).astype(int)

# Estrategia B: Unanimidad (Los 3 modelos deben coincidir)
pred_jurado_unanimidad = (votos_totales == 3).astype(int)

# Estrategia C: Voto de Confianza (Con que 1 solo modelo alerte, se revisa)
pred_jurado_conservador = (votos_totales >= 1).astype(int)

print("\n===================================================================")
print("   RESULTADOS DEL JURADO: MAYORÍA (2 de 3 deciden que es Fraude)  ")
print("===================================================================")
print(classification_report(y_test, pred_jurado_mayoria, target_names=['Normal (0)', 'Fraude (1)']))
print("Matriz de Confusión:")
print(confusion_matrix(y_test, pred_jurado_mayoria))

print("\n===================================================================")
print("   RESULTADOS DEL JURADO: UNANIMIDAD (3 de 3 deben coincidir)     ")
print("===================================================================")
print(classification_report(y_test, pred_jurado_unanimidad, target_names=['Normal (0)', 'Fraude (1)']))
print("Matriz de Confusión:")
print(confusion_matrix(y_test, pred_jurado_unanimidad))
