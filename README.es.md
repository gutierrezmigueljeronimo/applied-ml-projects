# Applied ML Projects

**Colección de proyectos de machine learning y deep learning: aprendizaje supervisado, no supervisado y por refuerzo**

Proyectos académicos · Centro EUSA, Sevilla · 2025–2026

---

## Descripción general

Cinco proyectos end-to-end desarrollados durante la especialización en ML y Big Data en el Centro EUSA. Cada proyecto aborda un dominio, una arquitectura y un paradigma de aprendizaje distintos: desde regresión sobre series temporales de sensores hasta segmentación no supervisada de clientes y aprendizaje por refuerzo tabular.

Todos los proyectos incluyen código funcional, modelos o artefactos entrenados, figuras de resultados y READMEs individuales con la metodología completa.

---

## Proyectos

| Proyecto | Tipo | Arquitectura | Dominio | Resultado clave |
|---|---|---|---|---|
| [Predicción RUL Turbinas](#turbine-rul-rnn) | Regresión | Stacked LSTM | Mantenimiento predictivo | MAE = 10,36 ciclos · RMSE = 14,42 |
| [Detección Anomalías ECG](#ecg-anomaly-vae) | Detección de anomalías | Conv1D Autoencoder | Sanidad | Accuracy = 97,77% · F1 = 0,986 |
| [Riesgo de Crédito MLP](#credit-risk-mlp) | Clasificación | MLP | Banca / Finanzas | AUC = 0,9195 · Recall = 85,4% |
| [Segmentación de Clientes](#customer-segmentation-clustering) | Clustering | K-Means · Jerárquico · DBSCAN · GMM | Marketing | 4 segmentos accionables · Silueta = 0,1634 |
| [Donkey Kong RL](#donkey-kong-rl) | Aprendizaje por refuerzo | Monte Carlo · Q-Learning | Juego / Grid world | Q-Learning: 100% éxito · ruta óptima de 9 pasos |

---

## Estructura del repositorio

```
applied-ml-projects/
│
├── turbine-rul-rnn/
│   ├── README.md
│   ├── train.py
│   ├── inference.py
│   ├── requirements.txt
│   └── outputs/
│       ├── model.keras
│       ├── scaler.pkl
│       ├── metadata.pkl
│       ├── history.pkl
│       └── figures/
│           ├── fig1_piecewise_rul.png
│           ├── fig2_correlacion_sensores.png
│           ├── fig3_curvas_aprendizaje.png
│           ├── fig4_trayectorias_degradacion.png
│           ├── fig5_analisis_residuos.png
│           └── fig6_mae_por_tramo.png
│
├── ecg-anomaly-vae/
│   ├── README.md
│   ├── VAE_ECG_Anomaly.py
│   ├── autoencoder_ecg.keras
│   ├── requirements.txt
│   └── output_figures/
│       ├── fig1_muestras_por_clase.png
│       ├── fig2_distribucion_clases.png
│       ├── fig3_curvas_aprendizaje.png
│       ├── fig4_distribucion_errores.png
│       ├── fig5_distribucion_umbral.png
│       ├── fig6_matriz_confusion.png
│       └── fig7_reconstrucciones.png
│
├── credit-risk-mlp/
│   ├── README.md
│   ├── mlp_riesgo_de_crédito.py
│   ├── mejor_modelo_hmeq.keras
│   ├── modelo_LR_variado_hmeq.keras
│   ├── hmeq.csv
│   ├── requirements.txt
│   └── figures/
│       ├── distribucion_de_variables.png
│       ├── matriz_de_correlacion.png
│       ├── comparativa_de_convergencia_y_rendimiento.png
│       ├── curva_ROC.png
│       ├── matriz_umbral_0,3.png
│       ├── matriz_umbral_0,5.png
│       └── matriz_umbral_0,7.png
│
├── customer-segmentation-clustering/
│   ├── README.md
│   ├── clustering_practica.ipynb
│   ├── marketing_campaign.csv
│   ├── requirements.txt
│   └── figures/
│       ├── analisis_de_componentes_principales.png
│       ├── coeficiente_de_silueta.png
│       ├── comparativa_de_metricas_por_algoritmo.png
│       ├── curvas_k-distance_por_min_samples.png
│       ├── DBSCAN.png
│       ├── dendograma.png
│       ├── K-Means.png
│       ├── perfil_de_centroides_por_cluster.png
│       └── seleccion_de_k_optimo.png
│
└── donkey-kong-rl/
    ├── README.md
    ├── Donkey_Kong_Inverso.ipynb
    ├── requirements.txt
    └── figures/
        ├── estructura_de_curvas.png
        ├── exploracion_con_politica_aleatoria.png
        ├── mapa_del_entorno.png
        ├── MC_impacto_del_decaimiento_de_ɛ.png
        ├── MC_vs_QL_entorno_estocastico.png
        ├── Monte_Carlo_politica_greedy.png
        ├── Monte_Carlo_politica_greedy_aprendida.png
        ├── Monte_Carlo_vs_Q-Learning.png
        └── Q-Learning_politica_greedy.png
```

---

## Resúmenes de proyectos

### Predicción RUL Turbinas {#turbine-rul-rnn}

Estimación de vida útil restante (RUL) para motores turbofán sobre el dataset NASA C-MAPSS FD001. Un LSTM apilado (128→64→32→1) procesa ventanas deslizantes de 50 lecturas de sensores para predecir cuántos ciclos le quedan a un motor antes del fallo.

**Arquitectura:** Stacked LSTM · **Parámetros:** 125.761 · **Dataset:** NASA C-MAPSS FD001 (100 motores, 21 sensores)

**Resultados:** MAE = **10,36 ciclos** · RMSE = **14,42 ciclos** · MAE en rango crítico (RUL < 25) = **2,50 ciclos**

→ [README completo](turbine-rul-rnn/README.md)

---

### Detección de Anomalías ECG {#ecg-anomaly-vae}

Detección no supervisada de anomalías cardíacas sobre ECG5000. Un autoencoder convolucional se entrena exclusivamente con latidos normales; los latidos anómalos se detectan en inferencia por su alto error de reconstrucción.

**Arquitectura:** Conv1D Autoencoder · **Parámetros:** 9.969 · **Dataset:** ECG5000 (9.502 latidos, recodificación binaria)

**Resultados:** Accuracy = **97,77%** · F1 (anómalos) = **0,986** · Recall = **98,6%** · Ratio de error (anómalos/normales) = **323×**

→ [README completo](ecg-anomaly-vae/README.md)

---

### Riesgo de Crédito MLP {#credit-risk-mlp}

Predicción binaria de impago en préstamos hipotecarios. Un MLP con pesos de clase y un umbral de decisión ajustado prioriza el recall para minimizar los impagos no detectados: el error crítico en el contexto bancario.

**Arquitectura:** MLP 64→32→16→1 · **Parámetros:** 4.033 · **Dataset:** HMEQ (5.960 préstamos)

**Resultados:** AUC = **0,9195** · Recall (insolventes) = **85,4%** · Umbral = 0,30 · FN = 26

→ [README completo](credit-risk-mlp/README.md)

---

### Segmentación de Clientes {#customer-segmentation-clustering}

Estudio comparativo de clustering sobre un dataset de marketing retail. Cuatro algoritmos (K-Means, Jerárquico Ward, DBSCAN, GMM) se evalúan con las métricas de Silueta y Davies-Bouldin, seguido de un análisis crítico de negocio que explica por qué las métricas superiores de DBSCAN son engañosas.

**Algoritmos:** K-Means · Jerárquico (Ward) · DBSCAN · GMM · **Dataset:** Marketing Campaign (2.240 clientes)

**Resultado:** K-Means recomendado para uso en producción mensual · k = 4 · Silueta = **0,1634**

→ [README completo](customer-segmentation-clustering/README.md)

---

### Donkey Kong RL {#donkey-kong-rl}

Estudio comparativo de RL tabular en un entorno grid 6×6 personalizado inspirado en Donkey Kong. Monte Carlo (on-policy) y Q-Learning (TD off-policy) se implementan desde cero y se comparan con 5 semillas en entornos determinista y estocástico.

**Algoritmos:** Monte Carlo · Q-Learning · **Entorno:** DonkeyKongInverso personalizado · **Proyecto en equipo** (MC: Miguel J. Gutiérrez · QL: Adrián Pavón)

**Resultado:** Ambos convergen a la ruta óptima de 9 pasos · Q-Learning converge en 5/5 semillas · MC en 3/5

→ [README completo](donkey-kong-rl/README.md)

---

## Stack tecnológico

| Área | Librerías |
|---|---|
| Deep learning | TensorFlow / Keras |
| ML clásico | scikit-learn |
| Datos | NumPy · pandas · SciPy |
| Visualización | Matplotlib · Seaborn |
| Aprendizaje por refuerzo | Python puro / NumPy (sin framework de RL) |
| Entornos | Local (Python 3.13) · Kaggle · Google Colab |

---

## Contexto académico

Todos los proyectos fueron desarrollados como prácticas evaluadas durante la especialización en IA y Big Data en el Centro EUSA (Sevilla), curso académico 2025–2026. Los proyectos realizados en equipo incluyen información de autoría en sus READMEs individuales.
