import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib 
import os
import shutil

# --- 1. CONFIGURACIÓN INICIAL ---
DATA_DIR = "processed_data_paysim"
ENCODING_DIM = 5 
THRESHOLD_PERCENTILE = 97 

EPOCHS = 200 
BATCH_SIZE = 128
SEED = 42
EARLY_STOPPING_PATIENCE = 10 

MODEL_DIR = "autoencoder_model_paysim"
MODEL_FILE = 'autoencoder_paysim.h5'
FILTRO_FILE_PATH = "df_ae_filtro_train_paysim.csv"

np.random.seed(SEED)
tf.random.set_seed(SEED)

# --- 2. CARGA DE DATOS PREPROCESADOS ---
print("1. Cargando datos preprocesados...")
try:
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
except FileNotFoundError:
    print(f"ERROR: No se encontraron los archivos procesados en {DATA_DIR}. Ejecuta preprocess_paysim.py primero.")
    exit() 

N_FEATURES = X_train.shape[1] 
X_train_normal_all = X_train[y_train == 0]

X_train_normal, X_val_normal = train_test_split(X_train_normal_all, test_size=0.1, random_state=SEED)

print(f"NÚMERO DE CARACTERÍSTICAS (N_FEATURES): {N_FEATURES}")
print(f"Datos Normales para Entrenamiento: {X_train_normal.shape[0]} muestras")

# --- 3. DEFINICIÓN Y ENTRENAMIENTO DEL AUTOENCODER ---
print("\n2. Definiendo y entrenando el Autoencoder...")

input_layer = Input(shape=(N_FEATURES, ))
encoder = Dense(int(N_FEATURES * 0.8), activation="relu")(input_layer) 
encoder = Dense(ENCODING_DIM, activation="relu", name="latent_space")(encoder)
decoder = Dense(int(N_FEATURES * 0.8), activation="relu")(encoder) 
output_layer = Dense(N_FEATURES, activation="linear")(decoder) 

autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer='adam', loss='mae')

early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=EARLY_STOPPING_PATIENCE,
    mode='min',
    restore_best_weights=True 
)

history = autoencoder.fit(
    X_train_normal, X_train_normal,
    epochs=EPOCHS, 
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_data=(X_val_normal, X_val_normal),
    callbacks=[early_stopping], 
    verbose=1
)

# --- 4. UMBRAL DE ANOMALÍA ---
print("\n3. Estableciendo el Umbral de Anomalía...")

X_val_predictions = autoencoder.predict(X_val_normal, verbose=0)
mse_val = np.mean(np.power(X_val_normal - X_val_predictions, 2), axis=1)

THRESHOLD = np.percentile(mse_val, THRESHOLD_PERCENTILE)
print(f"Umbral de Anomalía (Percentil {THRESHOLD_PERCENTILE}): {THRESHOLD:.4f}")

# --- 5. EVALUACIÓN FINAL DEL MODELO ---
print("\n4. Evaluación del Modelo (Test Set)...")

X_test_predictions = autoencoder.predict(X_test, verbose=0)
mse_test = np.mean(np.power(X_test - X_test_predictions, 2), axis=1) 

y_pred_final = (mse_test > THRESHOLD).astype(int)

print("\n--- Resultados de Detección (Conjunto de Test) ---")
print(classification_report(y_test, y_pred_final, target_names=['Normal (0)', 'Fraude (1)']))

print(f"Área bajo la curva ROC (AUC-ROC): {roc_auc_score(y_test, mse_test):.4f}")
print("\nMatriz de Confusión Final:")
print(confusion_matrix(y_test, y_pred_final))

# --- 6. GUARDAR MODELO Y DATA PARA EL FILTRO ---
os.makedirs(MODEL_DIR, exist_ok=True)
try:
    autoencoder.save(os.path.join(MODEL_DIR, MODEL_FILE))
except Exception as e:
    pass

if os.path.exists(os.path.join(DATA_DIR, "scaler.pkl")):
    shutil.copy(os.path.join(DATA_DIR, "scaler.pkl"), os.path.join(MODEL_DIR, 'scaler.pkl'))

np.save(os.path.join(MODEL_DIR, 'ae_threshold.npy'), THRESHOLD)

fp_indices = np.where((y_test == 0) & (y_pred_final == 1))[0]
vp_indices = np.where((y_test == 1) & (y_pred_final == 1))[0]

X_falsos_positivos = X_test[fp_indices]
X_verdaderos_positivos = X_test[vp_indices]

X_filtro_train = np.vstack([X_falsos_positivos, X_verdaderos_positivos])
y_filtro_train = np.hstack([np.zeros(len(X_falsos_positivos)), np.ones(len(X_verdaderos_positivos))])

df_filtro = pd.DataFrame(X_filtro_train)
df_filtro['is_fraud_label'] = y_filtro_train.astype(int)
df_filtro.to_csv(FILTRO_FILE_PATH, index=False)
