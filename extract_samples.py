import pandas as pd
import json
import random

# Leer el CSV original (solo algunas columnas para no reventar la RAM si se puede, 
# pero pd.read_csv está bien, o podemos leer a trozos).
# Como el archivo pesa 493MB, podemos leer en chunks para ser rápidos.

frauds = []
normals = []

chunksize = 100000
for chunk in pd.read_csv('PS_20174392719_1491204439457_log.csv', chunksize=chunksize):
    if len(frauds) < 25:
        f_chunk = chunk[chunk['isFraud'] == 1]
        frauds.extend(f_chunk.to_dict('records'))
    
    if len(normals) < 25:
        n_chunk = chunk[chunk['isFraud'] == 0]
        normals.extend(n_chunk.to_dict('records'))
        
    if len(frauds) >= 25 and len(normals) >= 25:
        break

# Tomar exactamente 25 de cada
frauds = frauds[:25]
normals = normals[:25]

combined = frauds + normals
random.shuffle(combined) # Mezclar

output = []
for i, row in enumerate(combined):
    # Generar un label amigable basado en type y si es fraude
    label = f"{row['type']} de ${row['amount']:,.2f}"
    
    sample = {
        "id": i + 1,
        "label": label,
        "is_fraud": bool(row['isFraud']),
        "data": {
            "amount": float(row['amount']),
            "oldbalanceOrg": float(row['oldbalanceOrg']),
            "newbalanceOrig": float(row['newbalanceOrig']),
            "oldbalanceDest": float(row['oldbalanceDest']),
            "newbalanceDest": float(row['newbalanceDest']),
            "type": row['type']
        }
    }
    output.append(sample)

with open('samples.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=4, ensure_ascii=False)

print("samples.json generado con 50 ejemplos.")
