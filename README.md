# 🛡️ Sentinel AI - Monitoreo de Fraude

¡Bienvenido al repositorio de **Sentinel AI**! Este proyecto es un portafolio avanzado diseñado para demostrar la integración de modelos de Machine Learning (Inteligencia Artificial) en un entorno de producción para la **Detección de Fraude en Transacciones Financieras**.

![Sentinel AI Dashboard Preview](https://via.placeholder.com/1200x600.png?text=Sentinel+AI+Dashboard) *(Puedes reemplazar esta imagen después)*

## 🚀 Sobre el Proyecto

Sentinel AI utiliza un innovador sistema de **"Ensamble de Jurado"**. En lugar de depender de un solo modelo de IA, utiliza tres modelos diferentes (Isolation Forest, Autoencoder, y Deep SVDD) que actúan como jueces. Cada modelo analiza una transacción sospechosa y emite su voto. El sistema toma una decisión final calculando un puntaje de riesgo ponderado.

Todo esto está encapsulado en una **Single Page Application (SPA)** moderna, elegante, interactiva y "gamificada" mediante *Glassmorphism*.

### ✨ Características Principales
* **Veredicto por Jurado:** Sistema multi-modelo que pondera resultados de 3 inteligencias artificiales para mitigar Falsos Positivos.
* **Dashboard en Tiempo Real:** Gráficos (vía `Chart.js`) que muestran el uso de los modelos y estadísticas globales de detección al vuelo.
* **Base de Datos Dinámica:** Integración de un catálogo extraído del Dataset real (*PaySim*) para realizar pruebas de estrés con un solo clic de forma automatizada.
* **Gamificación y Predicciones Manuales:** Los usuarios pueden intentar adivinar si un caso es "Fraude" o "Normal" antes de que la IA decida, creando una experiencia interactiva para reclutadores.
* **Arquitectura Desacoplada:** Backend robusto en FastAPI preparado para HuggingFace Spaces y Frontend puro (HTML/CSS/JS) preparado para Vercel.

---

## 🛠️ Stack Tecnológico

**Backend (API & Modelos de Machine Learning)**
* Python 3.9+
* FastAPI & Uvicorn
* TensorFlow / Keras (Modelos Deep Learning)
* Scikit-Learn (Modelos Clásicos)
* Pandas & Numpy

**Frontend (Interfaz de Usuario)**
* HTML5 Semántico
* CSS3 (Vanilla, Glassmorphism, Variables Nativas)
* JavaScript (ES6+, Fetch API)
* Chart.js (Visualización de datos)

---

## 💻 Instalación y Ejecución Local

### 1. Clonar el repositorio
```bash
git clone https://github.com/Chelo013/MonitoreoFraude.git
cd MonitoreoFraude
```

### 2. Configurar el Entorno del Backend
Se recomienda usar un entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Iniciar el Backend
Puedes correr el servidor Mock (ligero) o el servidor de IA real.
```bash
# Servidor Completo (Requiere TensorFlow nativo)
uvicorn api_paysim_backend:app --port 7860

# Servidor Mock (Para pruebas rápidas de UI sin TensorFlow)
uvicorn mock_backend:app --port 7860
```
La API estará disponible en `http://localhost:7860`.

### 4. Iniciar el Frontend
Dado que es Vanilla JS/HTML, puedes simplemente abrir `frontend/index.html` en tu navegador, o usar Live Server en VS Code.

> **Nota:** En `frontend/script.js`, asegúrate de que la variable `API_URL` apunte a `http://127.0.0.1:7860` (o a la URL de producción).

---

## ☁️ Guía de Despliegue (Deploy)

Este proyecto está arquitectado para un despliegue dividido de alta eficiencia:

### Backend (Hugging Face Spaces)
1. Crea un nuevo **Space** en Hugging Face configurado como **Docker**.
2. Sube todos los archivos del repositorio (excluyendo la carpeta `frontend/`).
3. El `Dockerfile` incluido se encargará de instalar las dependencias pesadas de TensorFlow y lanzar FastAPI.

### Frontend (Vercel)
1. Importa el repositorio a **Vercel**.
2. Configura el *Root Directory* a la carpeta `frontend`.
3. Antes de hacer deploy, cambia la variable `API_URL` en `script.js` a la URL pública que te entregó Hugging Face.

---

## 📂 Estructura del Repositorio

```text
├── frontend/                   # Aplicación Web (UI)
│   ├── index.html              # Estructura SPA
│   ├── style.css               # Estilos Glassmorphism
│   └── script.js               # Lógica, conexión API y gráficos
├── modelos/                    # (Directorio para los archivos .pkl, .h5)
├── samples.json                # Extracto de base de datos PaySim
├── api_paysim_backend.py       # API Principal (Full TensorFlow)
├── mock_backend.py             # API Ligera para desarrollo
├── extract_samples.py          # Script de extracción de datos
├── Dockerfile                  # Configuración para Hugging Face
└── requirements.txt            # Dependencias de Python
```

---

<div align="center">
  <b>Creado con 🧠 y ☕ para el futuro de la seguridad financiera.</b>
</div>
