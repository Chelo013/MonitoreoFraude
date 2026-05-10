import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.ensemble import IsolationForest
import joblib 
import os 

# --- 1. CONFIGURACIÓN INICIAL ---
DATA_DIR = "processed_data_paysim"
CONTAMINATION_RATE = 0.075
SEED = 42

# Configuraciones de Guardado
MODEL_DIR = "iforest_model_paysim"
FILTRO_FILE_PATH = "df_if_filtro_train_paysim.csv"

np.random.seed(SEED)

# --- 2. CARGA DE DATOS PREPROCESADOS ---
print("1. Cargando datos preprocesados...")
try:
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    # Opcional: cargar holdout si se desea evaluar también
    # X_holdout = np.load(os.path.join(DATA_DIR, "X_holdout.npy"))
    # y_holdout = np.load(os.path.join(DATA_DIR, "y_holdout.npy"))
except FileNotFoundError:
    print(f"ERROR: No se encontraron los archivos procesados en {DATA_DIR}. Ejecuta preprocess_paysim.py primero.")
    exit()

N_FEATURES = X_train.shape[1]

# Filtrar SOLO transacciones normales para el entrenamiento del Isolation Forest
X_train_normal = X_train[y_train == 0]

print(f"NÚMERO DE CARACTERÍSTICAS (N_FEATURES): {N_FEATURES}")
print(f"Datos Normales para Entrenamiento: {X_train_normal.shape[0]} muestras")

# --- 3. DEFINICIÓN Y ENTRENAMIENTO DEL ISOLATION FOREST ---

print("\n2. Definiendo y entrenando el Isolation Forest...")

iforest = IsolationForest(
    n_estimators=100,
    contamination=CONTAMINATION_RATE, 
    random_state=SEED, 
    n_jobs=-1
)

# Entrenamiento SÓLO con datos normales del Train Set
iforest.fit(X_train_normal)

print(f"Modelo Isolation Forest entrenado con contamination={CONTAMINATION_RATE:.4f}.")

# --- 4. EVALUACIÓN DEL MODELO EN EL CONJUNTO DE TEST ---

print("\n3. Evaluación del Modelo (Test Set)...")

# Predicción y Conversión: -1 (Anomalía) -> 1 (Fraude); 1 (Normal) -> 0 (Normal)
y_pred_iforest_raw = iforest.predict(X_test)
y_pred_final = np.where(y_pred_iforest_raw == -1, 1, 0)

# Score de anomalía para AUC-ROC
score_final = iforest.decision_function(X_test)
auc_score = roc_auc_score(y_test, -score_final) # Se invierte el signo

# Métricas de Evaluación
print("\n--- Resultados de Detección (Conjunto de Test) ---")
print(classification_report(y_test, y_pred_final, target_names=['Normal (0)', 'Fraude (1)']))

print(f"Área bajo la curva ROC (AUC-ROC): {auc_score:.4f}")
print("\nMatriz de Confusión Final:")
cm = confusion_matrix(y_test, y_pred_final)
print(cm)

# --- 5. GUARDAR MODELO Y DATA PARA EL FILTRO SUPERVISADO ---

os.makedirs(MODEL_DIR, exist_ok=True)

# Guardar el modelo (el scaler ya está guardado por el preprocesamiento, pero copiamos el archivo para completitud)
joblib.dump(iforest, os.path.join(MODEL_DIR, 'iforest_model.pkl'))

# Copiamos el scaler a esta carpeta también por conveniencia
import shutil
if os.path.exists(os.path.join(DATA_DIR, "scaler.pkl")):
    shutil.copy(os.path.join(DATA_DIR, "scaler.pkl"), os.path.join(MODEL_DIR, 'scaler.pkl'))

print(f"\nModelo Isolation Forest guardado en: {MODEL_DIR}")

# Crear y Guardar DATASET PARA EL FILTRO
# Identificar Falsos Positivos (FP) y Verdaderos Positivos (VP) en Test
fp_indices = np.where((y_test == 0) & (y_pred_final == 1))[0]
vp_indices = np.where((y_test == 1) & (y_pred_final == 1))[0]

X_falsos_positivos = X_test[fp_indices]
X_verdaderos_positivos = X_test[vp_indices]

# Construir el nuevo dataset
X_filtro_train = np.vstack([X_falsos_positivos, X_verdaderos_positivos])
y_filtro_train = np.hstack([np.zeros(len(X_falsos_positivos)), np.ones(len(X_verdaderos_positivos))])

# Convertir a DataFrame y guardar
df_filtro = pd.DataFrame(X_filtro_train)
df_filtro['is_fraud_label'] = y_filtro_train.astype(int)
df_filtro.to_csv(FILTRO_FILE_PATH, index=False)

print("\n--- Data para el Filtro Guardada ---")
print(f"Dataset para el Filtro supervisado (FP + VP) guardado en: {FILTRO_FILE_PATH}")
print(f"Clase 0 (FPs/Normales difíciles): {len(X_falsos_positivos)}")
print(f"Clase 1 (VPs/Fraude detectado): {len(X_verdaderos_positivos)}")
