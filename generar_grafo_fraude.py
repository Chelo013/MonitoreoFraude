import pandas as pd
import json

print("1. Cargando el dataset original...")
df = pd.read_csv('PS_20174392719_1491204439457_log.csv')

print("2. Filtrando sub-red criminal (Zoom Forense)...")
# Tomar una pequeña muestra de fraudes para no saturar el navegador
df_fraud = df[df['isFraud'] == 1].sample(150, random_state=42)

# Obtener las cuentas involucradas en esos fraudes
cuentas_sospechosas = set(df_fraud['nameOrig']).union(set(df_fraud['nameDest']))

# Buscar transacciones normales que se crucen con estas cuentas
df_normal_relacionado = df[(df['isFraud'] == 0) & 
                           ((df['nameOrig'].isin(cuentas_sospechosas)) | 
                            (df['nameDest'].isin(cuentas_sospechosas)))].head(300)

# Unir ambos para formar nuestra red
df_red = pd.concat([df_fraud, df_normal_relacionado])

print(f"Total de transacciones a graficar: {len(df_red)}")

print("3. Construyendo datos del grafo 3D...")
nodes_dict = {}
links = []

for _, row in df_red.iterrows():
    origen = row['nameOrig']
    destino = row['nameDest']
    monto = row['amount']
    es_fraude = row['isFraud']
    tipo = row['type']
    
    # Tamaño de la flecha/link basado en el monto
    grosor_flecha = max(1, min(monto / 100000, 10))
    
    if es_fraude == 1:
        if origen not in nodes_dict:
            nodes_dict[origen] = {'id': origen, 'color': '#ff1744', 'title': f"Origen Fraude:\n{origen}", 'size': 15}
        if destino not in nodes_dict:
            nodes_dict[destino] = {'id': destino, 'color': '#ff1744', 'title': f"Destino Fraude:\n{destino}", 'size': 20}
        
        links.append({
            'source': origen, 
            'target': destino, 
            'color': 'rgba(255, 23, 68, 0.8)', 
            'title': f"FRAUDE!\nMonto: ${monto:,.2f}\nTipo: {tipo}",
            'value': grosor_flecha
        })
    else:
        if origen not in nodes_dict:
            nodes_dict[origen] = {'id': origen, 'color': '#00e5ff', 'title': f"Cuenta:\n{origen}", 'size': 5}
        if destino not in nodes_dict:
            nodes_dict[destino] = {'id': destino, 'color': '#00e5ff', 'title': f"Cuenta:\n{destino}", 'size': 5}
        
        links.append({
            'source': origen, 
            'target': destino, 
            'color': 'rgba(0, 229, 255, 0.15)', 
            'title': f"Normal\nMonto: ${monto:,.2f}\nTipo: {tipo}",
            'value': grosor_flecha
        })

graph_data = {
    'nodes': list(nodes_dict.values()),
    'links': links
}

html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>3D Fraud Graph</title>
    <style> 
        body {{ margin: 0; background-color: transparent; overflow: hidden; font-family: sans-serif; }} 
        #3d-graph {{ width: 100vw; height: 100vh; overflow: hidden; }}
        #3d-graph canvas {{ display: block; }}
        .scene-tooltip {{
            background: rgba(10, 15, 30, 0.95) !important;
            border: 1px solid rgba(0, 229, 255, 0.4) !important;
            border-radius: 8px !important;
            padding: 10px 14px !important;
            color: #e2e8f0 !important;
            font-size: 14px !important;
            backdrop-filter: blur(8px) !important;
            box-shadow: 0 8px 32px rgba(0,0,0,0.8) !important;
            white-space: pre-wrap !important;
        }}
    </style>
    <script src="https://unpkg.com/3d-force-graph"></script>
</head>
<body>
    <div id="3d-graph"></div>
    <script>
        const gData = {json.dumps(graph_data)};

        const Graph = ForceGraph3D()(document.getElementById('3d-graph'))
            .graphData(gData)
            .nodeLabel('title')
            .nodeColor('color')
            .nodeVal('size')
            .nodeResolution(16) // Nodos más esféricos y definidos
            .linkWidth(link => link.value * 0.4)
            .linkColor('color')
            .linkOpacity(1.0)
            .backgroundColor('rgba(0,0,0,0)') // Totalmente transparente
            .showNavInfo(false);
            
        // Efecto de brillo (Bloom) sutil en la renderización WebGL
        const renderer = Graph.renderer();
        renderer.setClearColor(0x000000, 0); // Asegura transparencia en WebGL

        // Auto-rotación de la cámara para dar un efecto cinemático
        let angle = 0;
        const distance = 800;
        setInterval(() => {{
            angle += Math.PI / 1200;
            Graph.cameraPosition({{
                x: distance * Math.sin(angle),
                z: distance * Math.cos(angle)
            }});
        }}, 30);
    </script>
</body>
</html>"""

print("4. Guardando el archivo HTML...")
with open("grafo_fraude_interactivo.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("¡Listo! Da doble click al archivo 'grafo_fraude_interactivo.html' para verlo en tu navegador.")
