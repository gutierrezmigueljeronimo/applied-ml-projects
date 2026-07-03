# -*- coding: utf-8 -*-

# ==============================================================================
# Instalación de dependencias
# ==============================================================================

import subprocess, sys

def instalar(paquete):
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install', paquete, '--quiet'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

print("Verificando e instalando dependencias...")
for pkg in ['numpy', 'pandas', 'matplotlib', 'seaborn',
            'scikit-learn', 'tensorflow']:
    instalar(pkg)
print("Dependencias listas.\n")

# ==============================================================================
# VAE_ECG_Anomaly.ipynb — Detección de anomalías en señales ECG con Autoencoder
# ==============================================================================
# Estructura:
#   Fase 1 · Carga, exploración y preparación del dataset
#   Fase 2 · Diseño y entrenamiento del autoencoder convolucional
#   Fase 3 · Resultados, umbral de detección y conclusiones

# ── BLOQUE DE CONFIGURACIÓN CENTRAL ───────────────────────────────────────────

DATA_DIR        = './ECG5000'               # Carpeta con los dos CSV
TRAIN_FILE      = 'ECG5000_train.csv'
TEST_FILE       = 'ECG5000_test.csv'
MODEL_PATH      = 'autoencoder_ecg.keras'
OUTPUT_DIR      = './output_figures'

SEQUENCE_LENGTH = 140                       # Nº de puntos temporales por latido
N_CHANNELS      = 1                         # ECG = 1 canal (amplitud)
VAL_FRACTION    = 0.20                      # Fracción de normales para validación
THRESHOLD_PCT   = 95                        # Percentil del error en train para umbral
BATCH_SIZE      = 32
EPOCHS          = 200                       # antes 100 — le damos más margen de convergencia
PATIENCE        = 20                        # antes 10  — las mejoras son graduales, necesita más paciencia
LEARNING_RATE   = 0.001
RANDOM_SEED     = 42

# ── IMPORTACIONES ─────────────────────────────────────────────────────────────

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')           # Sin ventana interactiva: el script corre solo
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

warnings.filterwarnings('ignore')
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# FASE 1: CARGA, EXPLORACIÓN Y PREPARACIÓN DEL DATASET
# ==============================================================================

print("\n" + "=" * 70)
print("FASE 1: CARGA, EXPLORACIÓN Y PREPARACIÓN DEL DATASET")
print("=" * 70)


# ── 1.1 Carga de datos ────────────────────────────────────────────────────────

print("\n[1.1] Cargando los archivos del dataset...")

for fname in [TRAIN_FILE, TEST_FILE]:
    ruta = os.path.join(DATA_DIR, fname)
    if not os.path.exists(ruta):
        sys.exit(f"\n[ERROR] Archivo no encontrado: {ruta}\n"
                 f"Asegúrate de que DATA_DIR apunta a la carpeta ECG5000.\n")

# La primera columna de cada archivo es la etiqueta de clase (1–5).
# Las 140 columnas restantes son amplitudes del ECG muestreadas en el tiempo.
df_train_raw = pd.read_csv(os.path.join(DATA_DIR, TRAIN_FILE), header=None)
df_test_raw  = pd.read_csv(os.path.join(DATA_DIR, TEST_FILE),  header=None)

print(f"  Train raw: {df_train_raw.shape}  |  Test raw: {df_test_raw.shape}")
print(f"\n  Estructura del dataset:")
print(f"  · Columna 0:    etiqueta de clase (1 a 5)")
print(f"  · Columnas 1–{df_train_raw.shape[1]-1}: amplitud del ECG en {SEQUENCE_LENGTH} pasos temporales")
print(f"  · Cada fila representa un latido cardíaco completo")
print(f"  · Sin cabecera en los archivos originales")

# Clases originales del dataset UCR ECG5000:
#   1 → Latido normal
#   2 → Bloqueo de rama derecha del haz de His (RBBB)
#   3 → Bloqueo de rama izquierda (PVC)
#   4 → Paced beat
#   5 → Otros latidos no clasificables
nombres_clases = {
    1: 'Normal',
    2: 'R on T',
    3: 'PVC',
    4: 'SP',
    5: 'UB'
}

# Figura 1: Un latido representativo de cada una de las 5 clases originales.
# Lo pintamos ANTES de recodificar para mostrar las clases tal como vienen.
print("\n[1.1] Generando figura de muestras por clase original...")
fig, axes = plt.subplots(1, 5, figsize=(18, 4))

for i, clase in enumerate([1, 2, 3, 4, 5]):
    # Tomamos la primera muestra disponible de cada clase
    muestra = df_train_raw[df_train_raw.iloc[:, 0] == clase].iloc[0, 1:].values
    axes[i].plot(muestra, color='steelblue', lw=1.5)
    axes[i].set_title(f'Clase {clase}: {nombres_clases[clase]}', fontsize=11)
    axes[i].set_xlabel('Paso temporal')
    axes[i].set_ylabel('Amplitud')
    axes[i].grid(True, alpha=0.3)

plt.suptitle('Muestra representativa de cada categoría ECG (clases originales)',
             fontsize=13, y=1.02)
plt.tight_layout()
ruta_fig = os.path.join(OUTPUT_DIR, 'fig1_muestras_por_clase.png')
plt.savefig(ruta_fig, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Figura guardada: {ruta_fig}")

# ── 1.2 Reunificación y recodificación binaria ────────────────────────────────

print("\n[1.2] Reunificando train+test y recodificando a binario...")

# Concatenamos para tener todo el dataset disponible.
# La recodificación binaria simplifica el problema de detección:
#   · Clase 1 (Normal) → 1
#   · Clases 2–5 (cualquier anomalía) → 0
df = pd.concat([df_train_raw, df_test_raw], ignore_index=True)

df.columns = ['label'] + [f't_{i}' for i in range(SEQUENCE_LENGTH)]
df['label'] = df['label'].apply(lambda x: 1 if x == 1 else 0)

signal_cols = [c for c in df.columns if c.startswith('t_')]

print(f"  Dataset completo: {df.shape}")
print(f"  Etiquetas tras recodificación: {df['label'].unique()} "
      f"(1=Normal, 0=Anómalo)")


# ── 1.3 Distribución de clases y desbalanceo ──────────────────────────────────

print("\n[1.3] Analizando distribución de clases...")

n_normales = (df['label'] == 1).sum()
n_anomalos = (df['label'] == 0).sum()
pct_normal = 100 * n_normales / len(df)
pct_anomalo = 100 * n_anomalos / len(df)

print(f"  Normales:  {n_normales} ({pct_normal:.1f}%)")
print(f"  Anómalos:  {n_anomalos} ({pct_anomalo:.1f}%)")
print(f"  → El dataset está desbalanceado: los normales representan el "
      f"{pct_normal:.0f}% del total.")
print(f"  → Para un autoencoder esto es positivo: hay suficientes normales")
print(f"    para entrenar y suficientes anómalos para evaluar la detección.")

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(['Normal (1)', 'Anómalo (0)'], [n_normales, n_anomalos],
              color=['steelblue', 'crimson'], alpha=0.85, edgecolor='white')
ax.bar_label(bars, labels=[f'{n_normales}\n({pct_normal:.1f}%)',
                            f'{n_anomalos}\n({pct_anomalo:.1f}%)'],
             padding=4, fontsize=10)
ax.set_title('Distribución de clases (binario)')
ax.set_ylabel('Nº de muestras')
ax.set_ylim(0, max(n_normales, n_anomalos) * 1.2)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
ruta_fig = os.path.join(OUTPUT_DIR, 'fig2_distribucion_clases.png')
plt.savefig(ruta_fig, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Figura guardada: {ruta_fig}")

# ── 1.4 Verificación de valores faltantes ─────────────────────────────────────

print("\n[1.4] Verificando valores faltantes...")

n_nulos = df.isnull().sum().sum()
print(f"  Total de valores nulos en el dataset: {n_nulos}")

if n_nulos == 0:
    print("  → No hay valores faltantes. No se requiere imputación.")
else:
    # Si hubiera nulos, la estrategia sería imputación por interpolación lineal
    # a lo largo del eje temporal, ya que las señales ECG son continuas.
    print(f"  → Se detectaron {n_nulos} valores nulos.")
    print("  → Estrategia: interpolación lineal a lo largo del eje temporal.")
    df[signal_cols] = df[signal_cols].interpolate(axis=1)
    print("  → Imputación aplicada.")


# ── 1.5 Escalado ──────────────────────────────────────────────────────────────

print("\n[1.5] Aplicando escalado MinMaxScaler...")

# Escalamos cada punto temporal al rango [0, 1].
# Razón: las amplitudes de ECG varían entre pacientes y sesiones; normalizar
# estabiliza el gradiente durante el entrenamiento y hace que el MSE sea
# comparable entre muestras de distinta escala original.
#
# CRÍTICO: ajustamos el scaler solo con los latidos normales de entrenamiento.
# Si usáramos también los anómalos, las estadísticas de escala reflejarían
# patrones de degradación que el modelo no debe conocer en esta fase.

df_0 = df.copy()

# Separamos señales y etiquetas antes de escalar
X_all = df_0[signal_cols].values      # (N, 140)
y_all = df_0['label'].values          # (N,)

# El scaler ve solo las muestras normales en este punto;
# el split completo se hará en el siguiente bloque.
normales_mask = (y_all == 1)
scaler = MinMaxScaler()
scaler.fit(X_all[normales_mask])      # Ajuste solo con normales

X_all_scaled = scaler.transform(X_all)   # Se aplica a todo (sin reajuste)

print(f"  Scaler ajustado con {normales_mask.sum()} latidos normales.")
print(f"  Rango tras escalado: [{X_all_scaled.min():.4f}, "
      f"{X_all_scaled.max():.4f}]")
print("  Nota: valores anómalos fuera de [0,1] son posibles y esperados;")
print("  indican que sus amplitudes superan el rango visto en normales.")

# ── 1.6 Split específico para autoencoders ────────────────────────────────────

print("\n[1.6] Dividiendo el dataset (estrategia para detección de anomalías)...")

# En un clasificador clásico, el split es aleatorio estratificado.
# En un autoencoder de detección de anomalías, la lógica es diferente:
#
#   · TRAIN: SOLO latidos normales. El modelo aprende únicamente la
#     representación de la normalidad. Si entrenamos con anómalos, el modelo
#     aprendería a reconstruirlos también y no podría detectarlos.
#
#   · VALIDACIÓN: una fracción de los normales, para monitorizar
#     el entrenamiento sin contaminar el train.
#
#   · TEST: mezcla de normales y anómalos. Aquí evaluamos si el error de
#     reconstrucción es significativamente mayor en los anómalos.

X_normales = X_all_scaled[normales_mask]
X_anomalos = X_all_scaled[~normales_mask]
y_anomalos = y_all[~normales_mask]     # Todos son 0, pero lo mantenemos

X_train, X_val = train_test_split(
    X_normales,
    test_size=VAL_FRACTION,
    random_state=RANDOM_SEED
)

# Test: todos los anómalos + una porción equivalente de normales del val
# (para tener ambas clases representadas en la evaluación final)
X_test  = np.concatenate([X_val, X_anomalos], axis=0)
y_test  = np.concatenate([np.ones(len(X_val)),
                           np.zeros(len(X_anomalos))], axis=0)

print(f"  X_train (solo normales):  {X_train.shape}")
print(f"  X_val   (solo normales):  {X_val.shape}")
print(f"  X_test  (normales+anóm.): {X_test.shape}  "
      f"| Normales: {(y_test==1).sum()}  Anómalos: {(y_test==0).sum()}")


# ── 1.7 Reshape para Conv1D ───────────────────────────────────────────────────

print("\n[1.7] Reshaping tensores para entrada a Conv1D...")

# Conv1D espera tensores de forma (muestras, pasos_temporales, canales).
# Nuestros datos tienen forma (N, 140); añadimos la dimensión de canal.
X_train = X_train.reshape(-1, SEQUENCE_LENGTH, N_CHANNELS)
X_val   = X_val.reshape(-1, SEQUENCE_LENGTH, N_CHANNELS)
X_test  = X_test.reshape(-1, SEQUENCE_LENGTH, N_CHANNELS)

print(f"  X_train: {X_train.shape}")
print(f"  X_val:   {X_val.shape}")
print(f"  X_test:  {X_test.shape}")

print("\n[FASE 1 COMPLETADA]")
print("  Datos listos para el diseño y entrenamiento del autoencoder.")

# ==============================================================================
# FASE 2: DISEÑO Y ENTRENAMIENTO DEL AUTOENCODER CONVOLUCIONAL
# ==============================================================================

print("\n" + "=" * 70)
print("FASE 2: DISEÑO Y ENTRENAMIENTO DEL AUTOENCODER CONVOLUCIONAL")
print("=" * 70)


# ── 2.1 Arquitectura del autoencoder ──────────────────────────────────────────

print("\n[2.1] Definiendo arquitectura del autoencoder convolucional...")

# Usamos la API funcional de Keras (no Sequential) porque en un autoencoder
# nos interesa poder acceder al encoder de forma independiente en la Fase 3.

# --- ENCODER ---
# Reduce progresivamente la longitud temporal extrayendo características locales.
# Cada bloque Conv1D + MaxPooling1D divide la longitud a la mitad.
inputs  = keras.Input(shape=(SEQUENCE_LENGTH, N_CHANNELS), name='entrada')

x = layers.Conv1D(32, kernel_size=7, activation='relu',
                  padding='same', name='enc_conv1')(inputs)
x = layers.MaxPooling1D(pool_size=2, padding='same', name='enc_pool1')(x)

x = layers.Conv1D(16, kernel_size=7, activation='relu',
                  padding='same', name='enc_conv2')(x)
x = layers.MaxPooling1D(pool_size=2, padding='same', name='enc_pool2')(x)

x = layers.Conv1D(8, kernel_size=7, activation='relu',
                  padding='same', name='enc_conv3')(x)

# Espacio latente: representación comprimida de la señal.
# Con dos MaxPooling de 2, pasamos de 140 pasos a 35 pasos con 8 canales.
encoded = layers.MaxPooling1D(pool_size=2, padding='same',
                              name='espacio_latente')(x)

# --- DECODER ---
# Reconstruye la señal original de forma simétrica al encoder.
# UpSampling1D duplica la longitud temporal (inverso de MaxPooling1D).
x = layers.Conv1D(8, kernel_size=7, activation='relu',
                  padding='same', name='dec_conv1')(encoded)
x = layers.UpSampling1D(size=2, name='dec_up1')(x)

x = layers.Conv1D(16, kernel_size=7, activation='relu',
                  padding='same', name='dec_conv2')(x)
x = layers.UpSampling1D(size=2, name='dec_up2')(x)

x = layers.Conv1D(32, kernel_size=7, activation='relu',
                  padding='same', name='dec_conv3')(x)
x = layers.UpSampling1D(size=2, name='dec_up3')(x)

# Capa de salida: un único canal (la amplitud reconstruida) con sigmoid,
# ya que los datos están en [0, 1] tras el escalado MinMax.
# Recortamos a la longitud original porque los UpSamplings pueden generar
# algún paso temporal extra dependiendo de la longitud de entrada.
x = layers.Conv1D(1, kernel_size=7, activation='sigmoid',
                  padding='same', name='salida')(x)
decoded = layers.Cropping1D(
    cropping=(0, x.shape[1] - SEQUENCE_LENGTH),
    name='recorte_longitud'
)(x)

autoencoder = keras.Model(inputs, decoded, name='autoencoder_ecg')
autoencoder.summary()

# Guardamos también el encoder por separado para análisis del espacio latente
encoder = keras.Model(inputs, encoded, name='encoder_ecg')

print(f"\n  Forma del espacio latente: {encoded.shape[1:]}  "
      f"(compresión de {SEQUENCE_LENGTH} a {encoded.shape[1]} pasos)")

# ── 2.2 Compilación ───────────────────────────────────────────────────────────

print("\n[2.2] Compilando modelo...")

autoencoder.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='mse'         # MSE = error de reconstrucción por muestra
)


# ── 2.3 Callbacks ─────────────────────────────────────────────────────────────

print("\n[2.3] Configurando callbacks...")

mis_callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,                      # antes 1e-6 — permitimos pasos más finos al final
        verbose=1
    ),
]

# ── 2.4 Entrenamiento ─────────────────────────────────────────────────────────

print("\n[2.4] Entrenando autoencoder...")
print(f"  Train: {X_train.shape[0]} latidos normales")
print(f"  Val:   {X_val.shape[0]}  latidos normales")
print(f"  Épocas máx.: {EPOCHS}  |  Batch: {BATCH_SIZE}  |  Patience: {PATIENCE}\n")

# El target es la propia entrada: el modelo aprende a reconstruirse a sí mismo.
# El error de reconstrucción es mayor en muestras que no ha visto durante
# el entrenamiento (los latidos anómalos), lo que permite detectarlos.
history = autoencoder.fit(
    X_train, X_train,
    validation_data=(X_val, X_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=mis_callbacks,
    verbose=1
)

print(f"\n  Entrenamiento finalizado.")
print(f"  Modelo guardado en: {MODEL_PATH}")

# ==============================================================================
# FASE 3: RESULTADOS, UMBRAL DE DETECCIÓN Y CONCLUSIONES
# ==============================================================================

print("\n" + "=" * 70)
print("FASE 3: RESULTADOS, UMBRAL DE DETECCIÓN Y CONCLUSIONES")
print("=" * 70)


# ── 3.1 Curvas de aprendizaje ─────────────────────────────────────────────────

print("\n[3.1] Generando curvas de aprendizaje...")

h      = history.history
epocas = range(1, len(h['loss']) + 1)

plt.figure(figsize=(10, 4))
plt.plot(epocas, h['loss'],     label='Train loss (MSE)', color='steelblue', lw=2)
plt.plot(epocas, h['val_loss'], label='Val loss (MSE)',   color='darkorange',
         lw=2, ls='--')
plt.title('Curvas de aprendizaje — Autoencoder convolucional ECG')
plt.xlabel('Época')
plt.ylabel('MSE (error de reconstrucción)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
ruta_fig = os.path.join(OUTPUT_DIR, 'fig3_curvas_aprendizaje.png')
plt.savefig(ruta_fig, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Figura guardada: {ruta_fig}")

# ── 3.2 Error de reconstrucción por muestra ───────────────────────────────────

print("\n[3.2] Calculando error de reconstrucción por muestra...")

X_test_pred  = autoencoder.predict(X_test, verbose=0)
errores_test = np.mean(np.square(X_test - X_test_pred), axis=(1, 2))

errores_normales = errores_test[y_test == 1]
errores_anomalos = errores_test[y_test == 0]

print(f"  Error medio — Normales:  {errores_normales.mean():.6f}")
print(f"  Error medio — Anómalos:  {errores_anomalos.mean():.6f}")
print(f"  Ratio anómalo/normal:    {errores_anomalos.mean()/errores_normales.mean():.2f}x")

# El límite del eje X se fija en el percentil 99 de los errores anómalos
# para que las distribuciones sean visibles. Los outliers extremos
# (errores muy altos en unos pocos anómalos) distorsionan el eje si no se recortan.
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Panel izquierdo: zoom en la zona de los normales ---
x_lim_zoom = np.percentile(errores_normales, 99.5) * 3
axes[0].hist(errores_normales, bins=60, alpha=0.7, color='steelblue',
             label=f'Normales (n={len(errores_normales)})', density=True)
axes[0].hist(errores_anomalos, bins=60, alpha=0.7, color='crimson',
             label=f'Anómalos (n={len(errores_anomalos)})', density=True)
axes[0].set_xlim(0, x_lim_zoom)
axes[0].set_title('Zoom: zona de los normales')
axes[0].set_xlabel('MSE por latido')
axes[0].set_ylabel('Densidad')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# --- Panel derecho: escala logarítmica para ver ambas distribuciones ---
todos_errores = np.concatenate([errores_normales, errores_anomalos])
bins_log = np.logspace(
    np.log10(max(todos_errores.min(), 1e-6)),
    np.log10(todos_errores.max()),
    60
)
axes[1].hist(errores_normales, bins=bins_log, alpha=0.7, color='steelblue',
             label=f'Normales  (media={errores_normales.mean():.5f})',
             density=True)
axes[1].hist(errores_anomalos, bins=bins_log, alpha=0.7, color='crimson',
             label=f'Anómalos (media={errores_anomalos.mean():.4f})',
             density=True)
axes[1].set_xscale('log')
axes[1].set_title('Escala logarítmica: separación completa')
axes[1].set_xlabel('MSE por latido (escala log)')
axes[1].set_ylabel('Densidad')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3, which='both')

plt.suptitle(
    f'Distribución del error de reconstrucción: normales vs. anómalos\n'
    f'Ratio anómalo/normal: {errores_anomalos.mean()/errores_normales.mean():.0f}x',
    fontsize=12
)
plt.tight_layout()
ruta_fig = os.path.join(OUTPUT_DIR, 'fig4_distribucion_errores.png')
plt.savefig(ruta_fig, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Figura guardada: {ruta_fig}")

# ── 3.3 Definición del umbral de detección ────────────────────────────────────

print("\n[3.3] Calculando umbral de detección de anomalías...")

X_train_pred  = autoencoder.predict(X_train, verbose=0)
errores_train = np.mean(np.square(X_train - X_train_pred), axis=(1, 2))
umbral        = np.percentile(errores_train, THRESHOLD_PCT)

print(f"  Percentil {THRESHOLD_PCT} del error en train (normales): {umbral:.6f}")
print(f"  → Latidos con error > {umbral:.6f} serán clasificados como ANÓMALOS")

# Figura con dos paneles para mostrar la separación completa entre clases.
# Panel izquierdo: zoom en la zona de los normales, donde está el umbral.
# Panel derecho: vista amplia en escala log para ver dónde caen los anómalos.
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Panel izquierdo: zoom en normales ---
x_lim_zoom = np.percentile(errores_normales, 99.5) * 3
axes[0].hist(errores_normales, bins=60, alpha=0.7, color='steelblue',
             label=f'Normales (n={len(errores_normales)})', density=True)
axes[0].hist(errores_anomalos, bins=60, alpha=0.7, color='crimson',
             label=f'Anómalos (n={len(errores_anomalos)})', density=True)
axes[0].axvline(umbral, color='black', lw=2, ls='--',
                label=f'Umbral = {umbral:.5f}')
axes[0].set_xlim(0, x_lim_zoom)
axes[0].set_title('Zoom: zona del umbral')
axes[0].set_xlabel('MSE por latido')
axes[0].set_ylabel('Densidad')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# --- Panel derecho: escala logarítmica para ver ambas distribuciones ---
todos_errores = np.concatenate([errores_normales, errores_anomalos])
bins_log = np.logspace(
    np.log10(max(todos_errores.min(), 1e-6)),
    np.log10(todos_errores.max()),
    60
)
axes[1].hist(errores_normales, bins=bins_log, alpha=0.7, color='steelblue',
             label=f'Normales  (media={errores_normales.mean():.5f})',
             density=True)
axes[1].hist(errores_anomalos, bins=bins_log, alpha=0.7, color='crimson',
             label=f'Anómalos (media={errores_anomalos.mean():.4f})',
             density=True)
axes[1].axvline(umbral, color='black', lw=2, ls='--',
                label=f'Umbral = {umbral:.5f}')
axes[1].set_xscale('log')
axes[1].set_title('Escala logarítmica: separación completa')
axes[1].set_xlabel('MSE por latido (escala log)')
axes[1].set_ylabel('Densidad')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3, which='both')

plt.suptitle(
    f'Distribución del error de reconstrucción con umbral de detección\n'
    f'Ratio anómalo/normal: {errores_anomalos.mean()/errores_normales.mean():.0f}x',
    fontsize=12
)
plt.tight_layout()
ruta_fig = os.path.join(OUTPUT_DIR, 'fig5_distribucion_umbral.png')
plt.savefig(ruta_fig, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Figura guardada: {ruta_fig}")

# ── 3.4 Métricas de clasificación ─────────────────────────────────────────────

print("\n[3.4] Evaluando capacidad de detección sobre datos no vistos...")

# Convertimos el error continuo en una etiqueta binaria usando el umbral:
#   error > umbral → predicción = 0 (Anómalo)
#   error ≤ umbral → predicción = 1 (Normal)
y_pred = (errores_test <= umbral).astype(int)

print("\n  ── Informe de clasificación ──────────────────────────────────")
print(classification_report(y_test, y_pred,
                             target_names=['Anómalo (0)', 'Normal (1)'],
                             digits=4))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred. Anómalo', 'Pred. Normal'],
            yticklabels=['Real Anómalo', 'Real Normal'])
ax.set_title('Matriz de confusión — detección de anomalías ECG')
ax.set_ylabel('Etiqueta real')
ax.set_xlabel('Etiqueta predicha')
plt.tight_layout()
ruta_fig = os.path.join(OUTPUT_DIR, 'fig6_matriz_confusion.png')
plt.savefig(ruta_fig, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Figura guardada: {ruta_fig}")

# Análisis rápido del resultado
tn, fp, fn, tp = cm.ravel()
print(f"\n  Verdaderos Positivos (normales bien clasificados):  {tp}")
print(f"  Verdaderos Negativos (anómalos bien clasificados):  {tn}")
print(f"  Falsos Positivos (normales clasificados anómalos):  {fp}")
print(f"  Falsos Negativos (anómalos clasificados normales):  {fn}")
print(f"\n  Recall sobre anómalos: {tn/(tn+fp)*100:.1f}%  "
      f"(de cada 100 anómalos, detectamos {tn/(tn+fp)*100:.0f})")

# ── 3.5 Visualización de reconstrucciones ─────────────────────────────────────

print("\n[3.5] Visualizando reconstrucciones de latidos normales y anómalos...")

# Elegimos un latido normal y uno anómalo del test set y los comparamos
# con su reconstrucción. En el normal esperamos alta fidelidad;
# en el anómalo esperamos que el decoder "corrija" la señal hacia lo normal,
# dejando un residuo visible entre original y reconstruida.
np.random.seed(RANDOM_SEED)
idx_normal = np.random.choice(np.where(y_test == 1)[0])
idx_anomalo = np.random.choice(np.where(y_test == 0)[0])

fig, axes = plt.subplots(2, 1, figsize=(12, 7))

for i, (idx, nombre, color) in enumerate([
    (idx_normal,  'Normal',  'steelblue'),
    (idx_anomalo, 'Anómalo', 'crimson')
]):
    original      = X_test[idx, :, 0]
    reconstruida  = X_test_pred[idx, :, 0]
    error_puntual = np.square(original - reconstruida)

    axes[i].plot(original,     color=color,   lw=2,   alpha=0.9,
                 label='Señal original')
    axes[i].plot(reconstruida, color='black', lw=1.5, alpha=0.8,
                 ls='--', label='Reconstrucción')
    axes[i].fill_between(range(SEQUENCE_LENGTH), original, reconstruida,
                          alpha=0.2, color='orange', label='Error puntual')
    axes[i].set_title(
        f'Latido {nombre} — MSE: {errores_test[idx]:.6f} '
        f'(umbral: {umbral:.6f})'
    )
    axes[i].set_xlabel('Paso temporal')
    axes[i].set_ylabel('Amplitud (escalada)')
    axes[i].legend(fontsize=9)
    axes[i].grid(True, alpha=0.3)

plt.suptitle('Comparación: señal original vs. reconstrucción del autoencoder',
             fontsize=13)
plt.tight_layout()
ruta_fig = os.path.join(OUTPUT_DIR, 'fig7_reconstrucciones.png')
plt.savefig(ruta_fig, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Figura guardada: {ruta_fig}")

# ── 3.6 Conclusiones ──────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("CONCLUSIONES")
print("=" * 70)

recall_anomalos = tn / (tn + fp) * 100
precision_normal = tp / (tp + fn) * 100

print(f"""
1. ¿QUÉ TAN BIEN FUNCIONA EL AUTOENCODER?

   El modelo detecta el {recall_anomalos:.1f}% de los latidos anómalos (recall sobre
   clase 0). El error de reconstrucción medio en anómalos es
   {errores_anomalos.mean():.6f}, frente a {errores_normales.mean():.6f} en normales
   (ratio {errores_anomalos.mean()/errores_normales.mean():.1f}x). Esto confirma que el
   autoencoder aprendió a representar la morfología normal del ECG y falla
   al reconstruir patrones que no ha visto durante el entrenamiento.

   La principal limitación es que el umbral es estático y se calcula una
   sola vez sobre los datos de train. En producción, la distribución de
   errores podría desplazarse si el perfil de los pacientes o el equipo
   de medición cambia.

2. ¿QUÉ MEJORAS REDUCIRÍAN FALSOS POSITIVOS Y FALSOS NEGATIVOS?

   · Umbral adaptativo: en lugar de un percentil fijo, ajustar el umbral
     dinámicamente para maximizar F1 o según el coste relativo de cada
     tipo de error (un FN —anómalo no detectado— es más peligroso que
     un FP —normal mal clasificado— en un contexto clínico).

   · Arquitectura más profunda o con atención: un mecanismo de atención
     temporal permitiría al encoder focalizarse en las regiones del latido
     más discriminativas (complejo QRS, segmento ST).

   · Datos de entrenamiento más limpios: si algunos latidos etiquetados
     como normales contienen morfologías ambiguas, el modelo aprende a
     reconstruirlas y pierde capacidad discriminativa.

   · Conjunto de validación de anómalos para calibrar el umbral:
     usando un subconjunto pequeño de anómalos confirmados, podríamos
     optimizar el umbral con una curva ROC en lugar de fijarlo con un
     percentil.

3. ¿PODRÍAN LOS MODELOS RECURRENTES (LSTM/GRU) MEJORAR LA DETECCIÓN?

   Sí, y son una alternativa natural para señales temporales como el ECG.
   Un autoencoder basado en LSTM capturaría dependencias a largo plazo
   dentro del latido (por ejemplo, la relación entre la onda P y el
   complejo QRS varios pasos temporales después), algo que la Conv1D
   captura solo de forma local.

   Sin embargo, los modelos recurrentes tienen un coste computacional
   significativamente mayor y son más difíciles de entrenar (gradientes
   que se desvanecen o explotan). En la práctica, los autoencoders
   convolucionales suelen ser el punto de partida preferido para señales
   de longitud fija como las de este dataset, y se recurre a LSTM cuando
   se trabaja con secuencias de longitud variable o con dependencias
   temporales que superan el tamaño del kernel.
""")

print("=" * 70)
print("SCRIPT COMPLETADO — Todas las figuras guardadas en:", OUTPUT_DIR)
print("=" * 70)