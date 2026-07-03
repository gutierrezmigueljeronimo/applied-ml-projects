# =============================================================================
# inference.py — Evaluación del modelo RUL entrenado sobre el conjunto de test
# =============================================================================

# ── BLOQUE DE CONFIGURACIÓN CENTRAL ──────────────────────────────────────────

BASE_PATH       = r'C:\Users\Nimo\Desktop\RNN_Vida_Util_Turbinas\cmapss'
SEQUENCE_LENGTH = 50                                                                      # Debe coincidir con el valor usado en train.py
MODEL_PATH      = r'C:\Users\Nimo\Desktop\RNN_Vida_Util_Turbinas\outputs\model.keras'
SCALER_PATH     = r'C:\Users\Nimo\Desktop\RNN_Vida_Util_Turbinas\outputs\scaler.pkl'
METADATA_PATH   = r'C:\Users\Nimo\Desktop\RNN_Vida_Util_Turbinas\outputs\metadata.pkl'
OUTPUT_DIR      = r'C:\Users\Nimo\Desktop\RNN_Vida_Util_Turbinas\outputs'
N_MOTORES_VIZ   = 3                                                                       # Nº de motores a visualizar en trayectorias
RANDOM_SEED     = 42


# ── IMPORTACIONES ─────────────────────────────────────────────────────────────

import os
import sys
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import mean_squared_error, mean_absolute_error

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(RANDOM_SEED)


# =============================================================================
# FASE 3: INFERENCIA Y MÉTRICAS DE CALIDAD
# =============================================================================

print("\n" + "=" * 70)
print("FASE 3: INFERENCIA Y MÉTRICAS DE CALIDAD")
print("=" * 70)


# ── 3.0 Carga de artefactos entrenados ───────────────────────────────────────

print("\n[3.0] Cargando artefactos del entrenamiento...")

# Verificamos que todos los archivos necesarios existen antes de continuar.
artefactos = {
    'Modelo':     MODEL_PATH,
    'Scaler':     SCALER_PATH,
    'Metadatos':  METADATA_PATH,
}
for nombre, ruta in artefactos.items():
    if not os.path.exists(ruta):
        sys.exit(f"\n[ERROR] {nombre} no encontrado en: {ruta}\n"
                 f"Asegúrate de haber ejecutado train.py primero.\n")

# Importación de Keras aquí para no ralentizar el arranque del script
# si solo falta algún artefacto (el error saldría antes de importar TF).
from tensorflow import keras
import tensorflow as tf

model_rul = keras.models.load_model(MODEL_PATH)
scaler    = joblib.load(SCALER_PATH)

with open(METADATA_PATH, 'rb') as f:
    metadata = pickle.load(f)

features_cols  = metadata['features_cols']
cols_eliminar  = metadata['cols_eliminadas']
RUL_CLIP       = metadata['rul_clip']
SEQUENCE_LENGTH = metadata['sequence_length']

print(f"  Modelo:     {MODEL_PATH}  (cargado)")
print(f"  Scaler:     {SCALER_PATH}  (cargado)")
print(f"  Features:   {len(features_cols)} sensores → {features_cols}")
print(f"  Ventana:    {SEQUENCE_LENGTH} ciclos  |  RUL clip: {RUL_CLIP}")


# ── 3.1 Carga y preprocesado del test set ────────────────────────────────────

print("\n[3.1] Cargando y preprocesando test set...")

index_names   = ['unit_nr', 'time_cycles']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names  = [f's_{i}' for i in range(1, 22)]
col_names     = index_names + setting_names + sensor_names

def cargar_test(base_path, col_names):
    """Carga test_FD001.txt y RUL_FD001.txt con rutas relativas."""
    for fname in ['test_FD001.txt', 'RUL_FD001.txt']:
        ruta = os.path.join(base_path, fname)
        if not os.path.exists(ruta):
            sys.exit(f"\n[ERROR] Archivo no encontrado: {ruta}\n")

    df_test = pd.read_csv(
        os.path.join(base_path, 'test_FD001.txt'),
        sep=r'\s+', header=None, names=col_names
    )
    df_rul = pd.read_csv(
        os.path.join(base_path, 'RUL_FD001.txt'),
        sep=r'\s+', header=None, names=['RUL']
    )
    return df_test, df_rul

df_test_raw, df_rul = cargar_test(BASE_PATH, col_names)

# Aplicamos exactamente las mismas transformaciones que en train.py,
# pero usando el scaler ya ajustado (sin volver a ajustar = sin leakage).
df_test = df_test_raw.copy()
df_test = df_test.drop(cols_eliminar, axis=1, errors='ignore')
df_test[features_cols] = scaler.transform(df_test[features_cols])

print(f"  Test cargado: {df_test.shape}  |  "
      f"Motores: {df_test['unit_nr'].nunique()}")


# ── 3.2 Generación de tensores de test ───────────────────────────────────────

print("\n[3.2] Generando tensores de test (última ventana por motor)...")

def create_test_tensors(df_test, df_true_rul, seq_length, feature_cols):
    """
    Extrae la última ventana de seq_length ciclos de cada motor de test.
    Es la ventana que corresponde al instante en que NASA cortó la grabación,
    por lo que su RUL real está en RUL_FD001.txt.
    """
    X_list, y_list, ids_validos = [], [], []
    descartados = 0

    for unit_id in df_test['unit_nr'].unique():
        motor = df_test[df_test['unit_nr'] == unit_id]

        if len(motor) < seq_length:
            print(f"  [AVISO] Motor {unit_id} descartado "
                  f"(longitud {len(motor)} < {seq_length})")
            descartados += 1
            continue

        X_list.append(motor[feature_cols].values[-seq_length:])
        y_list.append(df_true_rul.iloc[unit_id - 1]['RUL'])
        ids_validos.append(unit_id)

    if descartados:
        print(f"  Motores descartados: {descartados}")

    return (np.array(X_list, dtype=np.float32),
            np.array(y_list, dtype=np.float32),
            ids_validos)

X_test, y_test, ids_validos = create_test_tensors(
    df_test, df_rul, SEQUENCE_LENGTH, features_cols
)
print(f"  X_test: {X_test.shape}  |  y_test: {y_test.shape}")
print(f"  Motores válidos para evaluación: {len(ids_validos)}")


# ── 3.3 Inferencia ───────────────────────────────────────────────────────────

print("\n[3.3] Realizando predicciones sobre el test set...")

# .flatten() convierte la salida (N, 1) en un vector 1D (N,),
# necesario para que las funciones de métricas no hagan broadcasting erróneo.
y_pred = model_rul.predict(X_test, verbose=1).flatten()

print(f"  Predicciones generadas: {y_pred.shape[0]} motores")
print(f"  Rango de predicciones: [{y_pred.min():.1f}, {y_pred.max():.1f}] ciclos")
print(f"  Rango de valores reales: [{y_test.min():.1f}, {y_test.max():.1f}] ciclos")


# ── 3.4 Cálculo de métricas ──────────────────────────────────────────────────

print("\n[3.4] Calculando métricas de rendimiento...")

def score_asimetrico_nasa(y_true, y_pred):
    """
    Función de puntuación asimétrica de la competición NASA C-MAPSS.

    Penaliza más la sobreestimación (predecir más RUL del real = riesgo de
    accidente) que la subestimación (predecir menos = mantenimiento anticipado):
      · d > 0 (sobreestimación): penalización exponencial con base 10
      · d < 0 (subestimación):   penalización exponencial con base 13

    Un score más bajo es mejor. La asimetría (10 vs 13) refleja que un fallo
    en vuelo es aproximadamente 1.3× más costoso que un mantenimiento extra.
    """
    score = 0.0
    for true, pred in zip(y_true, y_pred):
        d = pred - true
        if d < 0:
            score += np.exp(-d / 13.0) - 1   # subestimación (d negativo)
        else:
            score += np.exp(d / 10.0) - 1    # sobreestimación (d positivo)
    return score

rmse  = np.sqrt(mean_squared_error(y_test, y_pred))
mae   = mean_absolute_error(y_test, y_pred)
score = score_asimetrico_nasa(y_test, y_pred)

print("\n  ┌─────────────────────────────────────────────────────┐")
print(f"  │  RMSE (Error Cuadrático Medio) : {rmse:7.2f} ciclos     │")
print(f"  │  MAE  (Error Absoluto Medio)   : {mae:7.2f} ciclos     │")
print(f"  │  Score Asimétrico (C-MAPSS)    : {score:9.2f}          │")
print("  └─────────────────────────────────────────────────────┘")
print("\n  Interpretación:")
print(f"  · RMSE {rmse:.2f}: el modelo se equivoca en promedio "
      f"~{rmse:.0f} ciclos (penalizando más los errores grandes).")
print(f"  · MAE {mae:.2f}: el error típico de predicción es de "
      f"~{mae:.0f} ciclos por motor.")
print(f"  · Score {score:.0f}: cuanto más bajo, mejor. Referencia literatura: "
      f"modelos LSTM bien afinados en FD001 obtienen 200–400.")


# ── 3.5 Visualización de trayectorias de degradación ─────────────────────────

print(f"\n[3.5] Visualizando trayectorias de {N_MOTORES_VIZ} motores aleatorios...")

# Seleccionamos motores válidos (con ≥ SEQUENCE_LENGTH ciclos en test)
rng            = np.random.default_rng(RANDOM_SEED)
motores_muestra = rng.choice(ids_validos, size=N_MOTORES_VIZ, replace=False)

fig, axes = plt.subplots(1, N_MOTORES_VIZ, figsize=(6 * N_MOTORES_VIZ, 5))
if N_MOTORES_VIZ == 1:
    axes = [axes]

for idx, unit_id in enumerate(motores_muestra):
    motor_df  = df_test[df_test['unit_nr'] == unit_id]
    data_mat  = motor_df[features_cols].values
    n_ciclos  = data_mat.shape[0]

    # Reconstruimos todas las ventanas posibles de este motor de test
    # para graficar la trayectoria completa de predicción.
    ventanas = []
    for start in range(n_ciclos - SEQUENCE_LENGTH + 1):
        stop = start + SEQUENCE_LENGTH
        ventanas.append(data_mat[start:stop])
    ventanas = np.array(ventanas, dtype=np.float32)

    # Predicciones para cada ventana → trayectoria estimada del RUL
    pred_tray = model_rul.predict(ventanas, verbose=0).flatten()

    # Trayectoria real: si al final le quedan X ciclos, un paso antes le quedaban X+1
    rul_final = df_rul.iloc[unit_id - 1]['RUL']
    n_ventanas = len(pred_tray)
    real_tray  = [rul_final + (n_ventanas - 1 - i) for i in range(n_ventanas)]
    real_tray  = np.clip(real_tray, a_min=None, a_max=RUL_CLIP)

    eje_x = range(n_ventanas)
    axes[idx].plot(eje_x, real_tray,  color='black',  lw=2,  label='RUL real (Piecewise)')
    axes[idx].plot(eje_x, pred_tray,  color='crimson', lw=1.5, alpha=0.8,
                   marker='o', markersize=2, label='Predicción LSTM')
    axes[idx].set_title(f'Motor test ID: {unit_id}')
    axes[idx].set_xlabel('Nº de ventana (avance temporal)')
    axes[idx].set_ylabel('RUL estimado (ciclos)')
    axes[idx].legend(fontsize=9)
    axes[idx].grid(True, alpha=0.3)

plt.suptitle('Trayectorias de degradación: RUL real vs. predicción LSTM',
             fontsize=13)
plt.tight_layout()
ruta_fig = os.path.join(OUTPUT_DIR, 'fig4_trayectorias_degradacion.png')
plt.savefig(ruta_fig, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Figura guardada: {ruta_fig}")


# ── 3.6 Análisis de residuos ─────────────────────────────────────────────────

print("\n[3.6] Analizando residuos (y_real - y_pred)...")

# Un histograma de residuos centrado en 0 indica que el modelo no tiene sesgo
# sistemático. Desviaciones indican que tiende a sobrestimar o subestimar.
residuos    = y_test - y_pred
sesgo_medio = float(np.mean(residuos))
std_residuos = float(np.std(residuos))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Histograma de residuos ---
axes[0].hist(residuos, bins=25, color='mediumpurple', edgecolor='white', alpha=0.85)
axes[0].axvline(0,           color='red',   ls='--', lw=2,  label='Error = 0 (perfecto)')
axes[0].axvline(sesgo_medio, color='navy',  ls='-',  lw=2,
                label=f'Sesgo medio = {sesgo_medio:.2f} ciclos')
axes[0].set_title('Distribución de residuos (y_real − y_pred)')
axes[0].set_xlabel('Error (ciclos)   [ < 0 → sobreestimación | > 0 → subestimación ]')
axes[0].set_ylabel('Nº de motores')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# --- Scatter: predicho vs. real ---
lim_min = min(y_test.min(), y_pred.min()) - 5
lim_max = max(y_test.max(), y_pred.max()) + 5
axes[1].scatter(y_test, y_pred, alpha=0.6, color='steelblue', edgecolors='white',
                linewidths=0.5, s=40, label='Motor')
axes[1].plot([lim_min, lim_max], [lim_min, lim_max],
             color='red', ls='--', lw=1.5, label='Predicción perfecta')
axes[1].set_xlim(lim_min, lim_max)
axes[1].set_ylim(lim_min, lim_max)
axes[1].set_title('Predicho vs. Real')
axes[1].set_xlabel('RUL real (ciclos)')
axes[1].set_ylabel('RUL predicho (ciclos)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_aspect('equal')

plt.suptitle('Análisis de residuos y calidad de predicción', fontsize=13)
plt.tight_layout()
ruta_fig = os.path.join(OUTPUT_DIR, 'fig5_analisis_residuos.png')
plt.savefig(ruta_fig, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Figura guardada: {ruta_fig}")

print(f"\n  Sesgo sistemático (media residuos): {sesgo_medio:+.2f} ciclos")
print(f"  Desviación estándar de residuos:    {std_residuos:.2f} ciclos")

if sesgo_medio < -2:
    print("  → El modelo SOBREESTIMA el RUL. Predice que queda más vida de la real.")
    print("  → PELIGRO operativo: el motor podría fallar antes de la fecha prevista.")
elif sesgo_medio > 2:
    print("  → El modelo SUBESTIMA el RUL. Predice menos vida de la que queda.")
    print("  → Impacto económico: adelanta mantenimientos innecesarios.")
else:
    print("  → Sesgo dentro del umbral ±2 ciclos: el modelo no presenta sesgo sistemático.")


# ── 3.7 Análisis de error por tramo de vida ──────────────────────────────────

print("\n[3.7] Análisis de error por tramo de vida del motor...")

# Dividimos los motores en tres grupos según su RUL real:
# alto (>75), medio (25-75) y bajo (<25). El error en el tramo bajo es el
# más crítico desde el punto de vista operativo.
tramos = {
    'RUL alto  (>75 ciclos)':  (y_test > 75),
    'RUL medio (25–75 ciclos)': (y_test >= 25) & (y_test <= 75),
    'RUL bajo  (<25 ciclos)':  (y_test < 25),
}

print(f"\n  {'Tramo':<28}  {'N':<5}  {'MAE':>8}  {'RMSE':>8}  {'Sesgo':>8}")
print("  " + "-" * 62)
for nombre, mascara in tramos.items():
    if mascara.sum() == 0:
        continue
    yt, yp  = y_test[mascara], y_pred[mascara]
    mae_t   = mean_absolute_error(yt, yp)
    rmse_t  = np.sqrt(mean_squared_error(yt, yp))
    sesgo_t = np.mean(yt - yp)
    print(f"  {nombre:<28}  {mascara.sum():<5}  "
          f"{mae_t:>7.2f}  {rmse_t:>7.2f}  {sesgo_t:>+7.2f}")

# Figura: MAE por tramo de vida
fig, ax = plt.subplots(figsize=(8, 4))
etiquetas, maes = [], []
for nombre, mascara in tramos.items():
    if mascara.sum() > 0:
        etiquetas.append(nombre)
        maes.append(mean_absolute_error(y_test[mascara], y_pred[mascara]))

colores = ['steelblue', 'darkorange', 'crimson']
barras  = ax.bar(etiquetas, maes, color=colores[:len(etiquetas)], alpha=0.85,
                 edgecolor='white')
ax.bar_label(barras, fmt='%.1f ciclos', padding=3, fontsize=10)
ax.set_title('MAE por tramo de vida del motor')
ax.set_ylabel('MAE (ciclos)')
ax.set_ylim(0, max(maes) * 1.3)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
ruta_fig = os.path.join(OUTPUT_DIR, 'fig6_mae_por_tramo.png')
plt.savefig(ruta_fig, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Figura guardada: {ruta_fig}")


# ── 3.8 Informe de limitaciones ──────────────────────────────────────────────

print("\n" + "=" * 70)
print("INFORME DE LIMITACIONES DEL MODELO")
print("=" * 70)

# Identificamos en qué tramo el error es mayor (en ciclos absolutos)
mae_alto  = mean_absolute_error(y_test[y_test > 75],  y_pred[y_test > 75])  \
            if (y_test > 75).sum() > 0 else 0
mae_bajo  = mean_absolute_error(y_test[y_test < 25],  y_pred[y_test < 25])  \
            if (y_test < 25).sum() > 0 else 0

print(f"""
1. FASE INICIAL DE VIDA (RUL alto > 75 ciclos)
   MAE en este tramo: {mae_alto:.1f} ciclos.
   El Piecewise RUL fija la etiqueta a {RUL_CLIP} durante los primeros ciclos,
   lo que hace que el modelo aprenda una zona "plana" en la que cualquier
   predicción entre 100 y 125 resulta penalizada de forma similar. El error
   en este tramo es generalmente mayor porque la señal de degradación es
   débil y el modelo tiene poca información para distinguir cuánto queda.

2. FASE CRÍTICA (RUL bajo < 25 ciclos)
   MAE en este tramo: {mae_bajo:.1f} ciclos.
   Es el tramo de mayor riesgo operativo. Un error de 15 ciclos cuando quedan
   20 equivale a un error relativo del 75%, lo que puede comprometer la
   decisión de mantenimiento. La red mejora en este tramo respecto al inicio
   porque la señal de los sensores es mucho más marcada, pero el margen de
   error admisible es mínimo.

3. GENERALIZACIÓN A UN ÚNICO MODO DE FALLO
   Este modelo está entrenado exclusivamente con FD001, que simula un único
   modo de fallo y condiciones de vuelo homogéneas. No generaliza a FD002,
   FD003 ni FD004, que incluyen múltiples modos de fallo o condiciones
   variables. Un despliegue en entornos reales requeriría reentrenar con
   datos representativos de todos los escenarios operativos.

4. AUSENCIA DE INCERTIDUMBRE EN LA PREDICCIÓN
   El modelo emite un único valor puntual de RUL sin intervalo de confianza.
   Para un sistema de alto riesgo bajo el AI Act, sería preferible una
   estimación probabilística (por ejemplo mediante MC Dropout o Deep Ensembles)
   que permitiese cuantificar cuándo la predicción es fiable y cuándo no.

5. DEPENDENCIA DEL PREPROCESADO
   El rendimiento del modelo es sensible a que el scaler (scaler_rul.pkl) sea
   exactamente el mismo que se usó en entrenamiento. Si los datos de entrada
   en producción tuviesen una distribución diferente (por ejemplo, un sensor
   calibrado de forma distinta), el modelo degradaría su rendimiento sin
   emitir ninguna advertencia.
""")


# ── 3.9 Conclusión técnica: AI Act y explicabilidad ──────────────────────────

print("=" * 70)
print("CONCLUSIÓN TÉCNICA: AI ACT Y EXPLICABILIDAD")
print("=" * 70)
print(f"""
Bajo el marco del Reglamento Europeo de Inteligencia Artificial (AI Act,
Reglamento UE 2024/1689), los sistemas de predicción de fallos en
infraestructuras de transporte y energía son clasificados como sistemas
de ALTO RIESGO (Anexo III). Esto implica obligaciones directas sobre la
explicabilidad, la supervisión humana y la gestión de riesgos.

El modelo LSTM desarrollado es, por su naturaleza, una caja negra parcial:
puede justificar sus predicciones de forma indirecta (las curvas de
trayectoria muestran que sigue la tendencia de degradación) pero no puede
señalar con precisión qué sensor o qué combinación de sensores ha
determinado un RUL concreto en un momento dado.

Para cumplir las exigencias del AI Act, un despliegue real requeriría:
  · Técnicas de explicabilidad post-hoc como SHAP o Integrated Gradients
    aplicadas sobre las secuencias de entrada, para poder identificar qué
    sensores han tenido mayor peso en cada predicción.
  · Un umbral de confianza mínima por debajo del cual el sistema escalaría
    la decisión a un operador humano, garantizando supervisión efectiva.
  · Documentación técnica completa (Technical Documentation, Art. 11)
    que incluya la descripción del modelo, el proceso de validación,
    las métricas de rendimiento y las limitaciones conocidas.
  · Registro de logs de inferencia auditables, de modo que cada predicción
    pueda ser rastreada, revisada y, si es necesario, impugnada.

En su estado actual, el modelo puede servir como herramienta de apoyo a la
decisión (bajo supervisión de un ingeniero de mantenimiento), pero no está
certificado para operar de forma autónoma en sistemas de alto riesgo sin
las salvaguardas anteriores.
""")


# ── RESUMEN FINAL ─────────────────────────────────────────────────────────────

print("=" * 70)
print("EVALUACIÓN COMPLETADA")
print("=" * 70)
print(f"  RMSE:             {rmse:.2f} ciclos")
print(f"  MAE:              {mae:.2f} ciclos")
print(f"  Score asimétrico: {score:.2f}")
print(f"  Sesgo medio:      {sesgo_medio:+.2f} ciclos")
print(f"\n  Figuras guardadas en: {OUTPUT_DIR}/")
print("=" * 70 + "\n")
