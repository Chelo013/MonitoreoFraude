const API_URL = "https://chelo13-monitoreofraude.hf.space"; // Producción en HF Spaces

document.addEventListener("DOMContentLoaded", () => {
    // 1. Elementos del DOM
    const samplesTableBody = document.querySelector("#samples-table tbody");
    const form = document.getElementById("transaction-form");
    const analyzeBtn = document.getElementById("analyze-btn");
    const resultsSection = document.getElementById("results-section");
    const verdictContent = document.getElementById("verdict-content");
    const emptyState = document.getElementById("empty-results-state");
    const metricsDetails = document.getElementById("metrics-details");
    const btnText = document.querySelector(".btn-text");
    const loader = document.querySelector(".loader");
    
    // Sliders
    const sliders = ['if', 'ae', 'svdd'];
    sliders.forEach(id => {
        const slider = document.getElementById(`weight-${id}`);
        const valDisplay = document.getElementById(`val-${id}`);
        slider.addEventListener("input", (e) => {
            valDisplay.textContent = parseFloat(e.target.value).toFixed(2);
        });
    });

    const thSlider = document.getElementById("threshold");
    const thDisplay = document.getElementById("val-th");
    thSlider.addEventListener("input", (e) => {
        thDisplay.textContent = parseFloat(e.target.value).toFixed(2);
    });

    // --- VARIABLES GLOBALES DE GRÁFICOS ---
    let fraudChart = null;
    let usageChart = null;
    
    // Configuración general de Chart.js para dark mode
    Chart.defaults.color = '#a0aec0';
    Chart.defaults.font.family = "'Inter', sans-serif";

    // 2. Cargar datos de ejemplo y Estadísticas
    let sampleDataMap = {};
    let currentSampleGroundTruth = null;
    let allSamples = [];
    
    // Paginación
    let currentPage = 1;
    const itemsPerPage = 10;
    
    // Elementos de paginación
    const prevBtn = document.getElementById("prev-page");
    const nextBtn = document.getElementById("next-page");
    const pageInfo = document.getElementById("page-info");
    
    async function loadSamples() {
        try {
            const response = await fetch(`${API_URL}/samples`);
            if (!response.ok) throw new Error("No se pudieron cargar los ejemplos");
            
            allSamples = await response.json();
            
            allSamples.forEach(sample => {
                sampleDataMap[sample.id] = sample;
            });
            
            renderTablePage(1);
            
            // Cargar estadísticas iniciales
            loadStats();
        } catch (error) {
            console.error("Error al cargar samples:", error);
            samplesTableBody.innerHTML = `<tr><td colspan="6" style="text-align:center; color:#ef4444;">Error al conectar con la API</td></tr>`;
        }
    }
    
    function renderTablePage(page) {
        currentPage = page;
        const totalPages = Math.ceil(allSamples.length / itemsPerPage);
        
        // Actualizar UI de paginación
        pageInfo.textContent = `Página ${currentPage} de ${totalPages}`;
        prevBtn.disabled = currentPage === 1;
        nextBtn.disabled = currentPage === totalPages;
        
        // Limpiar tabla
        samplesTableBody.innerHTML = "";
        
        // Obtener el slice correcto
        const start = (currentPage - 1) * itemsPerPage;
        const end = start + itemsPerPage;
        const pageSamples = allSamples.slice(start, end);
        
        pageSamples.forEach(sample => {
            const truthBadge = sample.is_fraud ? 
                `<span class="type-badge CASH_OUT">Fraude</span>` : 
                `<span class="type-badge PAYMENT">Normal</span>`;
                
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td>#${sample.id}</td>
                <td><strong>${sample.label}</strong></td>
                <td><span class="type-badge ${sample.data.type}">${sample.data.type}</span></td>
                <td>$${sample.data.amount.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</td>
                <td>${truthBadge}</td>
                <td><button class="use-btn" data-id="${sample.id}">Analizar Automático</button></td>
            `;
            samplesTableBody.appendChild(tr);
        });
        
        // Asignar eventos a los nuevos botones
        document.querySelectorAll(".use-btn").forEach(btn => {
            btn.addEventListener("click", (e) => {
                const id = e.target.getAttribute("data-id");
                loadSampleIntoFormAndAnalyze(id);
                
                // Resaltar fila
                document.querySelectorAll("#samples-table tr").forEach(row => row.classList.remove("selected-row"));
                e.target.closest("tr").classList.add("selected-row");
            });
        });
    }
    
    // Eventos de Paginación
    prevBtn.addEventListener("click", () => {
        if (currentPage > 1) renderTablePage(currentPage - 1);
    });
    
    nextBtn.addEventListener("click", () => {
        const totalPages = Math.ceil(allSamples.length / itemsPerPage);
        if (currentPage < totalPages) renderTablePage(currentPage + 1);
    });

    async function loadStats() {
        try {
            const res = await fetch(`${API_URL}/stats`);
            if (res.ok) {
                const stats = await res.json();
                renderCharts(stats);
            }
        } catch (err) {
            console.error("No se pudieron cargar las estadísticas:", err);
        }
    }

    loadSamples();

    // 3. Rellenar formulario y analizar
    function loadSampleIntoFormAndAnalyze(id) {
        const sampleObj = sampleDataMap[id];
        if (!sampleObj) return;

        const data = sampleObj.data;
        currentSampleGroundTruth = sampleObj.is_fraud;
        
        document.getElementById("type").value = data.type;
        document.getElementById("amount").value = data.amount;
        document.getElementById("oldbalanceOrg").value = data.oldbalanceOrg;
        document.getElementById("newbalanceOrig").value = data.newbalanceOrig;
        document.getElementById("oldbalanceDest").value = data.oldbalanceDest;
        document.getElementById("newbalanceDest").value = data.newbalanceDest;
        
        // Reset user guess a "none" porque es análisis automático
        document.querySelector('input[name="user_guess"][value="none"]').checked = true;
        
        // Animación visual rápida
        form.style.boxShadow = "0 0 20px rgba(59, 130, 246, 0.4)";
        setTimeout(() => {
            form.style.boxShadow = "none";
        }, 500);
        
        // Disparar click en analizar automáticamente
        analyzeBtn.click();
    }

    // 4. Analizar Transacción Manual/Auto
    // Validar manual override (si es custom manual inputs ground truth es null)
    form.addEventListener("input", () => {
        // Al modificar manualmente, quitamos la verdad absoluta
        currentSampleGroundTruth = null;
    });

    analyzeBtn.addEventListener("click", async () => {
        // Validar formulario (básico)
        const amount = document.getElementById("amount").value;
        if (!amount) {
            alert("Por favor, ingrese al menos el monto.");
            return;
        }

        // Preparar UI
        btnText.style.display = "none";
        loader.style.display = "block";
        analyzeBtn.disabled = true;
        
        // Mostrar loader visual
        emptyState.innerHTML = `<div class="loader"></div><p style="margin-top:1rem">Analizando patrones de fraude...</p>`;
        emptyState.classList.remove("hidden");
        verdictContent.classList.add("hidden");
        metricsDetails.classList.add("hidden");

        const requestBody = {
            transaction: {
                type: document.getElementById("type").value,
                amount: parseFloat(document.getElementById("amount").value),
                oldbalanceOrg: parseFloat(document.getElementById("oldbalanceOrg").value),
                newbalanceOrig: parseFloat(document.getElementById("newbalanceOrig").value),
                oldbalanceDest: parseFloat(document.getElementById("oldbalanceDest").value),
                newbalanceDest: parseFloat(document.getElementById("newbalanceDest").value)
            },
            weight_iforest: parseFloat(document.getElementById("weight-if").value),
            weight_autoencoder: parseFloat(document.getElementById("weight-ae").value),
            weight_deep_svdd: parseFloat(document.getElementById("weight-svdd").value),
            threshold: parseFloat(document.getElementById("threshold").value)
        };

        try {
            const response = await fetch(`${API_URL}/predict`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(requestBody)
            });

            if (!response.ok) throw new Error("Error en la respuesta del servidor");
            const result = await response.json();
            
            showResults(result);
            
            // Refrescar gráficos
            loadStats();
        } catch (error) {
            console.error("Error al predecir:", error);
            alert("Error al conectar con el backend. Asegúrate de que HuggingFace Spaces o el servidor local esté corriendo.");
        } finally {
            btnText.style.display = "block";
            loader.style.display = "none";
            analyzeBtn.disabled = false;
        }
    });

    function showResults(data) {
        // Actualizar UI General
        const vCard = document.getElementById("verdict-card");
        const vIcon = document.querySelector(".verdict-icon");
        const vTitle = document.getElementById("verdict-title");
        const scoreSpan = document.querySelector(".score-number");

        scoreSpan.textContent = data.score_final.toFixed(3);

        if (data.verdict_is_fraud) {
            vCard.classList.add("is-fraud");
            vIcon.textContent = "🚨";
            vTitle.textContent = "¡Fraude Detectado!";
        } else {
            vCard.classList.remove("is-fraud");
            vIcon.textContent = "🛡️";
            vTitle.textContent = "Transacción Segura";
        }

        // Comparación con la predicción del usuario
        const userGuessEl = document.querySelector('input[name="user_guess"]:checked').value;
        const compEl = document.getElementById("user-comparison");
        
        if (userGuessEl !== "none") {
            compEl.classList.remove("hidden");
            const userSaidFraud = userGuessEl === "fraud";
            const aiSaidFraud = data.verdict_is_fraud;
            
            let html = `<h4>Tu predicción vs IA</h4>`;
            
            if (userSaidFraud === aiSaidFraud) {
                html += `<div class="comp-match">¡Coincides con la IA! Ambos dicen que es <b>${userSaidFraud ? 'Fraude' : 'Normal'}</b>.</div>`;
            } else {
                html += `<div class="comp-mismatch">No coincides con la IA. Tú dijiste <b>${userSaidFraud ? 'Fraude' : 'Normal'}</b> y la IA dice <b>${aiSaidFraud ? 'Fraude' : 'Normal'}</b>.</div>`;
            }
            
            if (currentSampleGroundTruth !== null) {
                const isActuallyFraud = currentSampleGroundTruth;
                html += `<div class="comp-truth mt-2">La realidad es que esta transacción <b>${isActuallyFraud ? 'SÍ ERA FRAUDE' : 'ERA NORMAL'}</b>.</div>`;
            }
            
            compEl.innerHTML = html;
        } else {
            compEl.classList.add("hidden");
            if (currentSampleGroundTruth !== null) {
                compEl.classList.remove("hidden");
                const isActuallyFraud = currentSampleGroundTruth;
                compEl.innerHTML = `<h4>Verdad de esta transacción</h4>
                <div class="comp-truth">En los datos reales esta transacción <b>${isActuallyFraud ? 'SÍ ERA FRAUDE' : 'ERA NORMAL'}</b>.</div>`;
            }
        }

        // Actualizar Votos
        updateVoteUI("if", data.jury_votes.isolation_forest, "Isolation Forest");
        updateVoteUI("ae", data.jury_votes.autoencoder, "Autoencoder");
        updateVoteUI("svdd", data.jury_votes.deep_svdd, "Deep SVDD");

        // Actualizar Métricas
        const mGrid = document.getElementById("metrics-grid");
        mGrid.innerHTML = `
            <div class="metric-item">
                <span class="metric-label">Autoencoder Error</span>
                ${data.metrics.autoencoder_error.toFixed(4)} <br/>
                <small>(Umbral: ${data.metrics.autoencoder_threshold.toFixed(4)})</small>
            </div>
            <div class="metric-item">
                <span class="metric-label">Deep SVDD Distancia</span>
                ${data.metrics.svdd_distance.toFixed(4)} <br/>
                <small>(Umbral: ${data.metrics.svdd_threshold.toFixed(4)})</small>
            </div>
            <div class="metric-item">
                <span class="metric-label">Umbral de Votación</span>
                ${data.threshold.toFixed(2)}
            </div>
        `;

        emptyState.classList.add("hidden");
        verdictContent.classList.remove("hidden");
        metricsDetails.classList.remove("hidden");
        
        // Mostrar contenedor con diseño flex
        verdictContent.style.display = "flex";
        verdictContent.style.gap = "2rem";

        // Scroll a los resultados suavemente
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    function updateVoteUI(id, isFraud, name) {
        const el = document.getElementById(`vote-${id}`);
        const resultText = el.querySelector(".vote-result");
        
        if (isFraud) {
            el.classList.add("fraud");
            resultText.textContent = "Alerta (Fraude)";
        } else {
            el.classList.remove("fraud");
            resultText.textContent = "Seguro";
        }
    }

    // --- LÓGICA DE GRÁFICOS (Chart.js) ---
    function renderCharts(stats) {
        const models = stats.models;
        const labels = ['Isolation Forest', 'Autoencoder', 'Deep SVDD'];
        
        const fraudData = [
            models.isolation_forest.fraud,
            models.autoencoder.fraud,
            models.deep_svdd.fraud
        ];
        
        const normalData = [
            models.isolation_forest.normal,
            models.autoencoder.normal,
            models.deep_svdd.normal
        ];

        // 1. Gráfico de Barras Apiladas (Fraude vs Normal)
        const ctxFraud = document.getElementById('fraudVsNormalChart').getContext('2d');
        if (fraudChart) fraudChart.destroy();
        
        fraudChart = new Chart(ctxFraud, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Fraude Detectado',
                        data: fraudData,
                        backgroundColor: 'rgba(239, 68, 68, 0.7)',
                        borderColor: 'rgb(239, 68, 68)',
                        borderWidth: 1
                    },
                    {
                        label: 'Normal',
                        data: normalData,
                        backgroundColor: 'rgba(16, 185, 129, 0.7)',
                        borderColor: 'rgb(16, 185, 129)',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Predicciones por Modelo',
                        color: '#f0f4ff',
                        font: { size: 16, family: "'Outfit', sans-serif" }
                    },
                    legend: { position: 'bottom' }
                },
                scales: {
                    x: { stacked: true },
                    y: { stacked: true }
                }
            }
        });

        // 2. Gráfico de Dona (Total de Análisis por Modelo)
        // En este caso todos analizan lo mismo, pero representamos el Total Requests
        const ctxUsage = document.getElementById('usageChart').getContext('2d');
        if (usageChart) usageChart.destroy();
        
        const totalAlerts = fraudData.reduce((a,b)=>a+b, 0);
        
        usageChart = new Chart(ctxUsage, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Alertas Totales',
                    data: fraudData,
                    backgroundColor: [
                        'rgba(59, 130, 246, 0.7)', // Azul
                        'rgba(139, 92, 246, 0.7)', // Morado
                        'rgba(236, 72, 153, 0.7)'  // Rosa
                    ],
                    borderColor: [
                        'rgb(59, 130, 246)',
                        'rgb(139, 92, 246)',
                        'rgb(236, 72, 153)'
                    ],
                    borderWidth: 1,
                    hoverOffset: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Aporte a Alertas Globales',
                        color: '#f0f4ff',
                        font: { size: 16, family: "'Outfit', sans-serif" }
                    },
                    legend: { position: 'bottom' }
                },
                cutout: '70%'
            }
        });
    }
});
