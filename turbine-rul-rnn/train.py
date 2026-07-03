# =============================================================================
# train.py — Pipeline completo de entrenamiento: RUL en turbinas NASA C-MAPSS
# =============================================================================

# ── BLOQUE DE CONFIGURACIÓN CENTRAL ──────────────────────────────────────────
# Todos los hiperparámetros y rutas están aquí

BASE_PATH       = r'C:\Users\Nimo\Desktop\RNN_Vida_Util_Turbinas\cmapss' # Carpeta con los tres .txt del dataset
SEQUENCE_LENGTH = 50                  # Nº de ciclos históricos por ventana
RUL_CLIP        = 125                 # Techo del Piecewise RUL
VAL_FRACTION    = 0.20                # Fracción de motores para validación
BATCH_SIZE      = 32
EPOCHS          = 100                 # EarlyStopping parará antes si procede
PATIENCE        = 10                  # Épocas sin mejora antes de detener
PATIENCE_LR     = 5                   # Épocas sin mejora antes de reducir LR
LSTM_UNITS_1    = 128                 # Unidades de la primera capa LSTM
LSTM_UNITS_2    = 64                  # Unidades de la segunda capa LSTM
DROPOUT_RATE    = 0.3
LEARNING_RATE   = 0.001
RANDOM_SEED     = 42
MODEL_PATH      = r'C:\Users\Nimo\Desktop\RNN_Vida_Util_Turbinas\outputs\model.keras'
SCALER_PATH     = r'C:\Users\Nimo\Desktop\RNN_Vida_Util_Turbinas\outputs\scaler.pkl'
METADATA_PATH   = r'C:\Users\Nimo\Desktop\RNN_Vida_Util_Turbinas\outputs\metadata.pkl'
HISTORY_PATH    = r'C:\Users\Nimo\Desktop\RNN_Vida_Util_Turbinas\outputs\history.pkl'
OUTPUT_DIR      = r'C:\Users\Nimo\Desktop\RNN_Vida_Util_Turbinas\outputs'


# ── IMPORTACIONES ─────────────────────────────────────────────────────────────

import os
import sys
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

warnings.filterwarnings('ignore')
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# FASE 1: CARGA, EDA Y PREPARACIÓN DE TENSORES
# =============================================================================

print("\n" + "=" * 70)
print("FASE 1: CARGA, EDA Y PREPARACIÓN DE TENSORES")
print("=" * 70)


# ── 1.1 Carga de datos ────────────────────────────────────────────────────────

print("\n[1.1] Cargando datos...")

index_names   = ['unit_nr', 'time_cycles']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names  = [f's_{i}' for i in range(1, 22)]
col_names     = index_names + setting_names + sensor_names

def load_data(base_path):
    """
    Carga train, test y RUL desde la carpeta del dataset C-MAPSS.
    Falla con mensaje claro si algún archivo no existe.
    """
    archivos = {
        'train': 'train_FD001.txt',
        'test':  'test_FD001.txt',
        'rul':   'RUL_FD001.txt'
    }
    for nombre, fname in archivos.items():
        ruta = os.path.join(base_path, fname)
        if not os.path.exists(ruta):
            sys.exit(f"\n[ERROR] Archivo no encontrado: {ruta}\n"
                     f"Asegúrate de que BASE_PATH apunta a la carpeta correcta.\n")

    df_train = pd.read_csv(
        os.path.join(base_path, archivos['train']),
        sep=r'\s+', header=None, names=col_names
    )
    df_test = pd.read_csv(
        os.path.join(base_path, archivos['test']),
        sep=r'\s+', header=None, names=col_names
    )
    df_rul = pd.read_csv(
        os.path.join(base_path, archivos['rul']),
        sep=r'\s+', header=None, names=['RUL']
    )
    return df_train, df_test, df_rul

df, df_test, df_rul = load_data(BASE_PATH)
print(f"  Train: {df.shape}  |  Test: {df_test.shape}  |  RUL: {df_rul.shape}")
print(f"  Nº de motores — Train: {df['unit_nr'].nunique()}  |  "
      f"Test: {df_test['unit_nr'].nunique()}")


# ── 1.2 EDA: estadísticas y varianza ─────────────────────────────────────────

print("\n[1.2] Análisis exploratorio (EDA)...")

df_0      = df.copy()
df_test_0 = df_test.copy()

desc = df_0.describe().T
print("\n  Estadísticas descriptivas (sensores):")
print(desc.loc[sensor_names, ['mean', 'std', 'min', 'max']].to_string())

# Sensores con desviación típica ≈ 0 son constantes: no aportan información
# sobre degradación y solo añaden ruido al cálculo de pesos.
cols_cte = desc[desc['std'] < 0.0001].index.tolist()
print(f"\n  Columnas constantes detectadas (std < 0.0001): {cols_cte}")


# ── 1.3 EDA: Detección de outliers ───────────────────────────────────────────

print("\n[1.3] Detección de outliers (método IQR sobre sensores variables)...")

# Usamos IQR con factor 3 (outliers extremos). Factor 1.5 sería demasiado
# agresivo en datos de degradación, donde los valores extremos son reales.
sensores_var = [c for c in sensor_names if c not in cols_cte]
outlier_info = {}

for col in sensores_var:
    Q1, Q3  = df_0[col].quantile(0.25), df_0[col].quantile(0.75)
    IQR     = Q3 - Q1
    mascara = (df_0[col] < Q1 - 3 * IQR) | (df_0[col] > Q3 + 3 * IQR)
    n_out   = mascara.sum()
    if n_out > 0:
        outlier_info[col] = n_out

if outlier_info:
    print("  Sensores con lecturas fuera de 3·IQR:")
    for col, n in sorted(outlier_info.items(), key=lambda x: -x[1]):
        pct = 100 * n / len(df_0)
        print(f"    {col}: {n} lecturas ({pct:.2f}%)")
    print("\n  Decisión: se conservan. En C-MAPSS los valores extremos reflejan")
    print("  degradación física real, no errores de sensor. Eliminarlos")
    print("  distorsionaría la tendencia de degradación que aprende la LSTM.")
else:
    print("  No se detectaron outliers extremos (ningún valor fuera de 3·IQR).")


# ── 1.4 Cálculo del target RUL Piecewise ─────────────────────────────────────

print("\n[1.4] Calculando RUL Piecewise...")

# El RUL real = (ciclo máximo del motor) - (ciclo actual).
# Piecewise: recortamos el máximo a RUL_CLIP porque los motores no presentan
# señales medibles de degradación en los primeros ciclos. Esto mejora la
# estabilidad del entrenamiento al evitar que la red intente aprender una
# pendiente plana de 200-300 ciclos antes de que empiece la degradación real.
max_cycle = (
    df_0.groupby('unit_nr')['time_cycles']
    .max()
    .reset_index()
    .rename(columns={'time_cycles': 'max_cycle'})
)
df_0 = df_0.merge(max_cycle, on='unit_nr', how='left')
df_0['RUL'] = (df_0['max_cycle'] - df_0['time_cycles']).clip(upper=RUL_CLIP)
df_0.drop('max_cycle', axis=1, inplace=True)

print(f"  RUL calculado. Rango: [0, {df_0['RUL'].max()}]  |  "
      f"Media: {df_0['RUL'].mean():.1f} ciclos")

# Figura: Piecewise RUL en 3 motores representativos
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for i, mid in enumerate([1, 2, 3]):
    datos = df_0[df_0['unit_nr'] == mid]
    axes[i].plot(datos['time_cycles'], datos['RUL'], color='steelblue', lw=2)
    axes[i].axhline(RUL_CLIP, color='orange', ls='--', alpha=0.7,
                    label=f'Clip = {RUL_CLIP}')
    axes[i].set_title(f'Motor {mid} — RUL Piecewise')
    axes[i].set_xlabel('Ciclos')
    axes[i].set_ylabel('RUL')
    axes[i].legend(fontsize=9)
    axes[i].grid(True, alpha=0.3)
plt.suptitle('Visualización del RUL Piecewise (3 motores de ejemplo)', y=1.02)
plt.tight_layout()
ruta_fig = os.path.join(OUTPUT_DIR, 'fig1_piecewise_rul.png')
plt.savefig(ruta_fig, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Figura guardada: {ruta_fig}")


# ── 1.5 Filtrado de características ──────────────────────────────────────────

print("\n[1.5] Filtrado de características...")

df_1      = df_0.copy()
df_test_1 = df_test_0.copy()

# s_14 tiene correlación ~1.0 con s_9 (comprobado en el mapa de calor):
# mantener ambas no añade información pero aumenta el coste computacional.
cols_eliminar = cols_cte + ['s_14']
df_1      = df_1.drop(cols_eliminar, axis=1, errors='ignore')
df_test_1 = df_test_1.drop(cols_eliminar, axis=1, errors='ignore')

features_cols = [c for c in df_1.columns
                 if c not in ['unit_nr', 'time_cycles', 'RUL']]
print(f"  Características seleccionadas ({len(features_cols)}): {features_cols}")

# Mapa de calor: correlaciones entre sensores y RUL
plt.figure(figsize=(11, 9))
corr = df_1[features_cols + ['RUL']].corr()
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True          # Mostramos solo la mitad inferior
sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True, fmt='.2f',
            linewidths=0.3, annot_kws={'size': 8})
plt.title('Correlación entre sensores seleccionados y RUL')
plt.tight_layout()
ruta_fig = os.path.join(OUTPUT_DIR, 'fig2_correlacion_sensores.png')
plt.savefig(ruta_fig, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Figura guardada: {ruta_fig}")


# ── 1.6 Split train / validación por motor ───────────────────────────────────

print("\n[1.6] Dividiendo en train y validación (por motor)...")

# Dividimos por ID de motor, no por filas, para garantizar que la red nunca
# ve en validación ciclos del mismo motor que en entrenamiento. Dividir por
# filas mezclaría el inicio y el final de vida de un mismo motor, lo que
# introduciría leakage temporal.
all_units = df_1['unit_nr'].unique().copy()
rng       = np.random.default_rng(RANDOM_SEED)
rng.shuffle(all_units)

n_val   = max(1, int(len(all_units) * VAL_FRACTION))
val_ids = set(all_units[:n_val])
trn_ids = set(all_units[n_val:])

df_trn = df_1[df_1['unit_nr'].isin(trn_ids)].copy()
df_val = df_1[df_1['unit_nr'].isin(val_ids)].copy()

print(f"  Motores → entrenamiento: {len(trn_ids)}  |  validación: {len(val_ids)}")


# ── 1.7 Normalización ────────────────────────────────────────────────────────

print("\n[1.7] Normalizando (MinMaxScaler ajustado solo en train)...")

# CRÍTICO: el scaler se ajusta ÚNICAMENTE con datos de entrenamiento.
# Ajustarlo con test o validación filtraría información del futuro al modelo
# (leakage estadístico). Los mismos parámetros se aplican después a val y test.
df_trn_2  = df_trn.copy()
df_val_2  = df_val.copy()
df_test_2 = df_test_1.copy()

scaler = MinMaxScaler()
scaler.fit(df_trn_2[features_cols])

df_trn_2[features_cols]  = scaler.transform(df_trn_2[features_cols])
df_val_2[features_cols]  = scaler.transform(df_val_2[features_cols])
df_test_2[features_cols] = scaler.transform(df_test_2[features_cols])

# Guardamos el scaler para poder reproducir exactamente la misma transformación
# en inferencia sin necesidad de tener los datos de entrenamiento.
joblib.dump(scaler, SCALER_PATH)
print(f"  Scaler guardado: {SCALER_PATH}")
print(f"  Rango post-escala (train): "
      f"[{df_trn_2[features_cols].min().min():.4f}, "
      f"{df_trn_2[features_cols].max().max():.4f}]")


# ── 1.8 Generación de tensores 3D (Sliding Window) ───────────────────────────

print("\n[1.8] Generando tensores 3D con Sliding Window...")

def create_tensors(df, seq_length, feature_cols, label_col):
    """
    Convierte un DataFrame de series temporales en tensores 3D para LSTM.

    Por cada motor desliza una ventana de 'seq_length' ciclos y genera:
      X[i] → bloque de seq_length ciclos consecutivos  (seq_length, n_features)
      y[i] → RUL en el ÚLTIMO ciclo de esa ventana

    La etiqueta usa el índice (stop - 1), no (stop), porque la ventana
    X[start:stop] cubre los índices start … stop-1 inclusive. Usar stop
    desplazaría el target un ciclo hacia el futuro (bug original).

    Motores con menos ciclos que seq_length se descartan con aviso.
    """
    X_list, y_list = [], []
    descartados = 0

    for unit_id in df['unit_nr'].unique():
        motor   = df[df['unit_nr'] == unit_id]
        data    = motor[feature_cols].values
        labels  = motor[label_col].values
        n       = len(data)

        if n < seq_length:
            print(f"  [AVISO] Motor {unit_id} descartado (longitud {n} < {seq_length})")
            descartados += 1
            continue

        # Generamos todas las ventanas posibles para este motor.
        # range(n - seq_length + 1) incluye la ventana que termina en el
        # último ciclo disponible (la más relevante para predecir el fallo).
        for start in range(n - seq_length + 1):
            stop = start + seq_length
            X_list.append(data[start:stop])       # ventana: ciclos start…stop-1
            y_list.append(labels[stop - 1])       # RUL al final de la ventana

    if descartados:
        print(f"  Total descartados: {descartados} motores")

    return (np.array(X_list, dtype=np.float32),
            np.array(y_list, dtype=np.float32))


def create_test_tensors(df_test, df_true_rul, seq_length, feature_cols):
    """
    Genera tensores de test: una única ventana por motor (la última disponible).

    En C-MAPSS los datos de test están truncados en un punto aleatorio antes
    del fallo. Solo tenemos el RUL real de ese punto final, por lo que solo
    tiene sentido predecir en la última ventana de cada motor.
    """
    X_list, y_list = [], []
    descartados = 0

    for unit_id in df_test['unit_nr'].unique():
        motor = df_test[df_test['unit_nr'] == unit_id]

        if len(motor) < seq_length:
            print(f"  [AVISO] Motor {unit_id} (test) descartado "
                  f"(longitud {len(motor)} < {seq_length})")
            descartados += 1
            continue

        # Última ventana: los seq_length ciclos más recientes del motor
        X_list.append(motor[feature_cols].values[-seq_length:])
        # RUL real proporcionado por NASA (RUL_FD001.txt), indexado por motor
        y_list.append(df_true_rul.iloc[unit_id - 1]['RUL'])

    if descartados:
        print(f"  Total descartados (test): {descartados} motores")

    return (np.array(X_list, dtype=np.float32),
            np.array(y_list, dtype=np.float32))


X_train, y_train = create_tensors(df_trn_2, SEQUENCE_LENGTH, features_cols, 'RUL')
X_val,   y_val   = create_tensors(df_val_2, SEQUENCE_LENGTH, features_cols, 'RUL')
X_test,  y_test  = create_test_tensors(df_test_2, df_rul, SEQUENCE_LENGTH, features_cols)

print(f"\n  X_train: {X_train.shape}  |  y_train: {y_train.shape}")
print(f"  X_val:   {X_val.shape}    |  y_val:   {y_val.shape}")
print(f"  X_test:  {X_test.shape}   |  y_test:  {y_test.shape}")

# Guardamos metadatos necesarios para reproducir exactamente el preprocesado
# en inference.py sin depender de los datos de entrenamiento.
metadata = {
    'features_cols':   features_cols,
    'sequence_length': SEQUENCE_LENGTH,
    'rul_clip':        RUL_CLIP,
    'cols_eliminadas': cols_eliminar,
}
with open(METADATA_PATH, 'wb') as f:
    pickle.dump(metadata, f)
print(f"\n  Metadatos guardados: {METADATA_PATH}")


# =============================================================================
# FASE 2: DISEÑO, ENTRENAMIENTO Y OPTIMIZACIÓN
# =============================================================================

print("\n" + "=" * 70)
print("FASE 2: DISEÑO, ENTRENAMIENTO Y OPTIMIZACIÓN")
print("=" * 70)


# ── 2.1 Arquitectura LSTM ────────────────────────────────────────────────────

print("\n[2.1] Definiendo arquitectura Stacked LSTM...")

def crear_modelo_lstm(input_shape):
    """
    Arquitectura Stacked LSTM para regresión de RUL.

    Topología de embudo (128 → 64 → 32 → 1):
    - Cada capa comprime la información, forzando a la red a extraer solo
      los patrones más relevantes de la secuencia temporal.
    - Dropout tras cada LSTM reduce la dependencia de sensores individuales.
    - Salida con activación lineal: correcta para regresión. ReLU bloquearía
      los gradientes negativos e introduciría sesgo hacia valores altos.

    Args:
        input_shape: (seq_length, n_features), ej. (50, 14)
    Returns:
        Modelo Keras compilado listo para entrenar.
    """
    model = keras.Sequential([
        keras.Input(shape=input_shape),

        # Primera LSTM: lee la secuencia completa de 50 ciclos.
        # return_sequences=True: devuelve el estado en cada paso de tiempo,
        # necesario para que la segunda LSTM pueda seguir procesando la secuencia.
        layers.LSTM(LSTM_UNITS_1, return_sequences=True, name='lstm_1'),
        layers.Dropout(DROPOUT_RATE, name='dropout_1'),

        # Segunda LSTM: sintetiza la secuencia en un único vector.
        # return_sequences=False: solo devuelve el estado del último paso.
        layers.LSTM(LSTM_UNITS_2, return_sequences=False, name='lstm_2'),
        layers.Dropout(DROPOUT_RATE, name='dropout_2'),

        # Capa densa intermedia: combina no linealmente los estados LSTM
        # para aprender relaciones entre los distintos sensores.
        layers.Dense(32, activation='relu', name='dense_hidden'),

        # Capa de salida: una sola neurona con activación lineal.
        # Lineal = sin restricción en el rango de salida, que es lo correcto
        # para predecir un valor continuo (ciclos restantes).
        layers.Dense(1, activation='linear', name='output'),
    ])
    return model


input_shape = (X_train.shape[1], X_train.shape[2])
model_rul   = crear_modelo_lstm(input_shape)
model_rul.summary()


# ── 2.2 Compilación ──────────────────────────────────────────────────────────

print("\n[2.2] Compilando modelo...")

# MSE como función de pérdida: penaliza cuadráticamente los errores grandes,
# lo que es adecuado aquí porque los errores de predicción grandes (predecir
# 100 ciclos cuando quedan 20) son más graves que los errores pequeños.
# MAE como métrica adicional: más interpretable (error promedio en ciclos).
model_rul.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='mse',
    metrics=['mae']
)


# ── 2.3 Callbacks ────────────────────────────────────────────────────────────

print("\n[2.3] Configurando callbacks de entrenamiento...")

mis_callbacks = [
    # Detiene el entrenamiento si val_loss no mejora en PATIENCE épocas,
    # y restaura los pesos del mejor momento (no de la última época).
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        mode='min',
        restore_best_weights=True,
        verbose=1
    ),
    # Guarda el modelo solo cuando mejora val_loss, no en cada época.
    callbacks.ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    ),
    # Reduce el learning rate a la mitad si val_loss lleva PATIENCE_LR épocas
    # sin mejorar. Ayuda a salir de mesetas de entrenamiento sin overfitting.
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=PATIENCE_LR,
        min_lr=1e-6,
        verbose=1
    ),
]


# ── 2.4 Entrenamiento ────────────────────────────────────────────────────────

print("\n[2.4] Entrenando modelo...")
print(f"  Motores entrenamiento: {len(trn_ids)}  |  "
      f"Muestras: {X_train.shape[0]:,}")
print(f"  Motores validación:    {len(val_ids)}  |  "
      f"Muestras: {X_val.shape[0]:,}")
print(f"  Batch size: {BATCH_SIZE}  |  Épocas máx.: {EPOCHS}  |  "
      f"Patience: {PATIENCE}\n")

history = model_rul.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),   # Validación con motores no vistos en train
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=mis_callbacks,
    verbose=1
)

# Guardamos el historial para poder graficar las curvas de aprendizaje
# en cualquier momento sin necesidad de reentrenar.
with open(HISTORY_PATH, 'wb') as f:
    pickle.dump(history.history, f)
print(f"\n  Historial guardado: {HISTORY_PATH}")


# ── 2.5 Curvas de aprendizaje ────────────────────────────────────────────────

print("\n[2.5] Generando curvas de aprendizaje...")

h            = history.history
epocas       = range(1, len(h['loss']) + 1)
mejor_epoca  = int(np.argmin(h['val_loss'])) + 1
mejor_val_loss = min(h['val_loss'])
mejor_val_mae  = h['val_mae'][mejor_epoca - 1]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico 1: pérdida MSE
axes[0].plot(epocas, h['loss'],     label='Train MSE',  color='steelblue',  lw=2)
axes[0].plot(epocas, h['val_loss'], label='Val MSE',    color='darkorange', lw=2, ls='--')
axes[0].axvline(mejor_epoca, color='gray', ls=':', alpha=0.7,
                label=f'Mejor época ({mejor_epoca})')
axes[0].set_title(f'Pérdida MSE — mejor val_loss: {mejor_val_loss:.2f}')
axes[0].set_xlabel('Épocas')
axes[0].set_ylabel('MSE')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Gráfico 2: MAE — más interpretable porque está en ciclos
axes[1].plot(epocas, h['mae'],     label='Train MAE',  color='seagreen',  lw=2)
axes[1].plot(epocas, h['val_mae'], label='Val MAE',    color='crimson',   lw=2, ls='--')
axes[1].axvline(mejor_epoca, color='gray', ls=':', alpha=0.7,
                label=f'Mejor época ({mejor_epoca})')
axes[1].set_title(f'MAE en ciclos — mejor val_mae: {mejor_val_mae:.2f}')
axes[1].set_xlabel('Épocas')
axes[1].set_ylabel('MAE (ciclos)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Curvas de aprendizaje — entrenamiento vs. validación', fontsize=13)
plt.tight_layout()
ruta_fig = os.path.join(OUTPUT_DIR, 'fig3_curvas_aprendizaje.png')
plt.savefig(ruta_fig, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Figura guardada: {ruta_fig}")

# Diagnóstico automático de overfitting/underfitting
gap_loss = h['val_loss'][-1] - h['loss'][-1]
print(f"\n  Diagnóstico de aprendizaje:")
print(f"    Mejor época:       {mejor_epoca}")
print(f"    Mejor val_loss:    {mejor_val_loss:.4f}")
print(f"    Mejor val_MAE:     {mejor_val_mae:.2f} ciclos")
print(f"    Gap train/val MSE: {gap_loss:.4f}", end=' ')
if gap_loss > 50:
    print("→ Posible overfitting. Considera aumentar Dropout o reducir LSTM_UNITS.")
elif gap_loss < 0:
    print("→ Underfitting. Considera aumentar épocas o capacidad del modelo.")
else:
    print("→ Generalización correcta.")


# ── RESUMEN FINAL ─────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("ENTRENAMIENTO COMPLETADO")
print("=" * 70)
print(f"  Modelo:    {MODEL_PATH}")
print(f"  Scaler:    {SCALER_PATH}")
print(f"  Metadatos: {METADATA_PATH}")
print(f"  Historial: {HISTORY_PATH}")
print(f"  Figuras:   {OUTPUT_DIR}/")
print("\n  Ejecutar 'python inference.py' para evaluar sobre el test set.")
print("=" * 70 + "\n")
