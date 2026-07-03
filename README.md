# Applied ML Projects

**A collection of machine learning and deep learning projects covering supervised, unsupervised, and reinforcement learning**

Academic projects · Centro EUSA, Sevilla · 2025–2026

---

## Overview

Five end-to-end projects built during the ML and Big Data specialisation at Centro EUSA. Each project tackles a different problem domain, architecture type, and learning paradigm — from regression on time-series sensor data to unsupervised customer segmentation and tabular reinforcement learning.

All projects include working code, trained models or artefacts, result figures, and individual READMEs with full methodology documentation.

---

## Projects

| Project | Type | Architecture | Domain | Key result |
|---|---|---|---|---|
| [Turbine RUL Prediction](#turbine-rul-rnn) | Regression | Stacked LSTM | Predictive maintenance | MAE = 10.36 cycles · RMSE = 14.42 |
| [ECG Anomaly Detection](#ecg-anomaly-vae) | Anomaly detection | Conv1D Autoencoder | Healthcare | Accuracy = 97.77% · F1 = 0.986 |
| [Credit Risk MLP](#credit-risk-mlp) | Classification | MLP | Banking / Finance | AUC = 0.9195 · Recall = 85.4% |
| [Customer Segmentation](#customer-segmentation-clustering) | Clustering | K-Means · Hierarchical · DBSCAN · GMM | Marketing | 4 actionable segments · Silhouette = 0.1634 |
| [Donkey Kong RL](#donkey-kong-rl) | Reinforcement learning | Monte Carlo · Q-Learning | Game / Grid world | Q-Learning: 100% success · 9-step optimal path |

---

## Repository structure

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

## Project summaries

### Turbine RUL Prediction {#turbine-rul-rnn}

Remaining Useful Life estimation for turbofan engines on NASA C-MAPSS FD001. A stacked LSTM (128→64→32→1) processes sliding windows of 50 sensor readings to predict how many cycles an engine has left before failure.

**Architecture:** Stacked LSTM · **Parameters:** 125,761 · **Dataset:** NASA C-MAPSS FD001 (100 engines, 21 sensors)

**Results:** MAE = **10.36 cycles** · RMSE = **14.42 cycles** · MAE in critical range (RUL < 25) = **2.50 cycles**

→ [Full README](turbine-rul-rnn/README.md)

---

### ECG Anomaly Detection {#ecg-anomaly-vae}

Unsupervised cardiac anomaly detection on ECG5000. A Conv1D autoencoder trains exclusively on normal heartbeats; anomalous beats are detected at inference time by their high reconstruction error.

**Architecture:** Conv1D Autoencoder · **Parameters:** 9,969 · **Dataset:** ECG5000 (9,502 heartbeats, binary recode)

**Results:** Accuracy = **97.77%** · F1 (anomalies) = **0.986** · Recall = **98.6%** · Error ratio (anomalous/normal) = **323×**

→ [Full README](ecg-anomaly-vae/README.md)

---

### Credit Risk MLP {#credit-risk-mlp}

Binary default prediction on home equity loans. An MLP with class weights and a tuned decision threshold prioritises recall to minimise undetected defaults — the critical error in a banking context.

**Architecture:** MLP 64→32→16→1 · **Parameters:** 4,033 · **Dataset:** HMEQ (5,960 loans)

**Results:** AUC = **0.9195** · Recall (defaulters) = **85.4%** · Threshold = 0.30 · FN = 26

→ [Full README](credit-risk-mlp/README.md)

---

### Customer Segmentation {#customer-segmentation-clustering}

Comparative clustering study on a retail marketing dataset. Four algorithms (K-Means, Hierarchical Ward, DBSCAN, GMM) are evaluated with Silhouette and Davies-Bouldin metrics, followed by a critical business analysis explaining why DBSCAN's superior metrics are misleading.

**Algorithms:** K-Means · Hierarchical (Ward) · DBSCAN · GMM · **Dataset:** Marketing Campaign (2,240 customers)

**Result:** K-Means recommended for monthly production use · k = 4 · Silhouette = **0.1634**

→ [Full README](customer-segmentation-clustering/README.md)

---

### Donkey Kong RL {#donkey-kong-rl}

Comparative tabular RL study on a custom 6×6 grid world inspired by Donkey Kong. Monte Carlo (on-policy) and Q-Learning (off-policy TD) are implemented from scratch and compared across 5 random seeds in deterministic and stochastic environments.

**Algorithms:** Monte Carlo · Q-Learning · **Environment:** Custom DonkeyKongInverso · **Team project** (MC: Miguel J. Gutiérrez · QL: Adrián Pavón)

**Result:** Both converge to the same 9-step optimal path · Q-Learning converges in 5/5 seeds · MC in 3/5

→ [Full README](donkey-kong-rl/README.md)

---

## Tech stack

| Area | Libraries |
|---|---|
| Deep learning | TensorFlow / Keras · PyTorch (MatForge) |
| Classical ML | scikit-learn |
| Data | NumPy · pandas · SciPy |
| Visualisation | Matplotlib · Seaborn |
| Reinforcement learning | Pure Python / NumPy (no RL framework) |
| Environments | Local (Python 3.13) · Kaggle · Google Colab |

---

## Academic context

All projects were developed as graded assignments during the AI and Big Data specialisation at Centro EUSA (Sevilla), academic year 2025–2026. Projects marked as team work include authorship details in their individual READMEs.
