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
THRESHOLD_PERCENTILE = 93 

EPOCHS = 200 
BATCH_SIZE = 128
SEED = 42
EARLY_STOPPING_PATIENCE = 10 

NU_PARAM = 0.1 
MODEL_DIR = "deep_svdd_model_paysim"
FILTRO_FILE_PATH = "df_svdd_filtro_train_paysim.csv"

np.random.seed(SEED)
tf.random.set_seed(SEED)

def svdd_loss(nu, R, c):
    R = tf.cast(R, dtype=tf.float32) 
    c = tf.cast(c, dtype=tf.float32) 
    def loss(y_true, y_pred):
        distances = tf.reduce_sum(tf.square(y_pred - c), axis=1)
        slack = tf.maximum(0.0, distances - R)
        loss_value = R + (1.0 / nu) * tf.reduce_mean(slack)
        return loss_value
    return loss

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

# Filtrar solo normales para el entrenamiento
X_train_normal_all = X_train[y_train == 0]

# Separar 10% de las normales de train para validación (Early Stopping y Umbral)
X_train_normal, X_val_normal = train_test_split(X_train_normal_all, test_size=0.1, random_state=SEED)

print(f"NÚMERO DE CARACTERÍSTICAS (N_FEATURES): {N_FEATURES}")
print(f"Datos Normales para Entrenamiento: {X_train_normal.shape[0]} muestras")
print(f"Datos Normales para Validación: {X_val_normal.shape[0]} muestras")

# --- 3. DEFINICIÓN Y ENTRENAMIENTO DEL DEEP SVDD ---
print("\n2. Definiendo y entrenando el Deep SVDD...")

CENTER_C = np.zeros(ENCODING_DIM)
R = tf.Variable(0.0, dtype=tf.float32, trainable=True, name="R")

input_layer = Input(shape=(N_FEATURES, ))
encoder = Dense(int(N_FEATURES * 0.8), activation="elu", use_bias=False)(input_layer) 
encoder = Dense(ENCODING_DIM, activation="linear", use_bias=False, name="latent_space")(encoder)
deep_svdd = Model(inputs=input_layer, outputs=encoder)

z_normal = deep_svdd.predict(X_train_normal, verbose=0)
CENTER_C = np.mean(z_normal, axis=0)

deep_svdd.compile(optimizer='adam', loss=svdd_loss(nu=NU_PARAM, R=R, c=CENTER_C))

early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=EARLY_STOPPING_PATIENCE,
    mode='min',
    restore_best_weights=True 
)

deep_svdd.fit(
    X_train_normal, np.zeros(X_train_normal.shape[0]), 
    epochs=EPOCHS, 
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_data=(X_val_normal, np.zeros(X_val_normal.shape[0])),
    callbacks=[early_stopping],
    verbose=1
)
FINAL_R = R.numpy()

# --- 4. EVALUACIÓN Y UMBRAL ---
print("\n3. Evaluación FINAL y Guardado...")

Z_val = deep_svdd.predict(X_val_normal, verbose=0)
CENTER_C_TENSOR = tf.cast(CENTER_C, dtype=tf.float32)
distances_val = np.sum(np.square(Z_val - CENTER_C_TENSOR.numpy()), axis=1)
THRESHOLD = np.percentile(distances_val, THRESHOLD_PERCENTILE)
print(f"El Umbral de Anomalía (Percentil {THRESHOLD_PERCENTILE} de distancia) es: {THRESHOLD:.4f}")

Z_test = deep_svdd.predict(X_test, verbose=0)
distances_test = np.sum(np.square(Z_test - CENTER_C_TENSOR.numpy()), axis=1) 
score_final = distances_test 

y_pred_final = (score_final > THRESHOLD).astype(int)

print("\n--- Resultados de Detección (Conjunto de Test) ---")
print(classification_report(y_test, y_pred_final, target_names=['Normal (0)', 'Fraude (1)']))
print(f"Área bajo la curva ROC (AUC-ROC): {roc_auc_score(y_test, score_final):.4f}")
print("\nMatriz de Confusión Final:")
print(confusion_matrix(y_test, y_pred_final))

# --- 5. GUARDAR MODELO Y DATA PARA EL FILTRO ---
os.makedirs(MODEL_DIR, exist_ok=True)
try:
    deep_svdd.save(os.path.join(MODEL_DIR, 'deep_svdd_encoder.keras'))
except Exception as e:
    pass

if os.path.exists(os.path.join(DATA_DIR, "scaler.pkl")):
    shutil.copy(os.path.join(DATA_DIR, "scaler.pkl"), os.path.join(MODEL_DIR, 'scaler.pkl'))

np.save(os.path.join(MODEL_DIR, 'center_c.npy'), CENTER_C)
np.save(os.path.join(MODEL_DIR, 'threshold.npy'), THRESHOLD)

fp_indices = np.where((y_test == 0) & (y_pred_final == 1))[0]
vp_indices = np.where((y_test == 1) & (y_pred_final == 1))[0]

X_falsos_positivos = X_test[fp_indices]
X_verdaderos_positivos = X_test[vp_indices]

X_filtro_train = np.vstack([X_falsos_positivos, X_verdaderos_positivos])
y_filtro_train = np.hstack([np.zeros(len(X_falsos_positivos)), np.ones(len(X_verdaderos_positivos))])

df_filtro = pd.DataFrame(X_filtro_train)
df_filtro['is_fraud_label'] = y_filtro_train.astype(int)
df_filtro.to_csv(FILTRO_FILE_PATH, index=False)
