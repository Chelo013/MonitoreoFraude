import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

print("1. Cargando el dataset completo (puede tomar un minuto o más)...")
FILE_PATH = "PS_20174392719_1491204439457_log.csv"
data = pd.read_csv(FILE_PATH)

print("2. Limpiando y preparando características...")
# Columnas a eliminar (identificadores y target filtrado)
COLUMNS_TO_DROP = ['nameOrig', 'nameDest', 'isFlaggedFraud', 'step']
data = data.drop(columns=COLUMNS_TO_DROP, errors='ignore')

# Separar Features (X) y Target (y)
y = data['isFraud'].values
X_raw = data.drop(columns=['isFraud'])

# One-Hot Encoding para la columna categórica (type)
categorical_cols = X_raw.select_dtypes(include=['object']).columns
X_processed = pd.get_dummies(X_raw, columns=categorical_cols, dummy_na=False)
X_processed = X_processed.fillna(0)

# Convertir a numpy array
X = X_processed.values

print("3. Haciendo el split estratificado (70% Train, 15% Test, 15% Holdout)...")
# Primero separamos Train (70%) y el resto (30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# Luego dividimos el resto (30%) a la mitad para tener 15% Test y 15% Holdout
X_test, X_holdout, y_test, y_holdout = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"   - Tamaño Train: {X_train.shape[0]} muestras")
print(f"   - Tamaño Test: {X_test.shape[0]} muestras")
print(f"   - Tamaño Holdout: {X_holdout.shape[0]} muestras")

print("4. Escalando los datos (entrenando el scaler SÓLO con Train)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_holdout_scaled = scaler.transform(X_holdout)

print("5. Guardando archivos procesados y listos para los modelos...")
OUTPUT_DIR = "processed_data_paysim"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Guardar los arrays
np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train_scaled)
np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)

np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test_scaled)
np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

np.save(os.path.join(OUTPUT_DIR, "X_holdout.npy"), X_holdout_scaled)
np.save(os.path.join(OUTPUT_DIR, "y_holdout.npy"), y_holdout)

# Guardar el scaler
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))

print(f"¡Listo! Todos los archivos fueron guardados en la carpeta '{OUTPUT_DIR}'.")
print(f"Número de características finales (N_FEATURES): {X_train.shape[1]}")
