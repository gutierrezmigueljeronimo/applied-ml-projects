import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("==================== Fase 1 ====================")

url = "http://www.creditriskanalytics.net/uploads/1/9/5/1/19511601/hmeq.csv"

try:
    df = pd.read_csv(url)
    print(f"Datos cargados. Shape: {df.shape}")
except:
    if os.path.exists('hmeq.csv'):
        df = pd.read_csv('hmeq.csv')
    else:
        print("Error: No se encuentra el dataset.")
        exit()

print("Análisis inicial de variables:")
print("\t- BAD (Variable categórica nominal binaria): La etiqueta objetivo. 1 = El cliente incumplió o se retrasó seriamente en el pago. 0 = Cliente solvente.\n")
print("\t- LOAN (Variable numérica continua): Monto de la solicitud de préstamo en dólares.\n")
print("\t- MORTDUE (Variable numérica continua): Monto adeudado en la hipoteca existente (lo que ya debe de la casa).\n")
print("\t- VALUE (Variable numérica continua): Valor actual de la propiedad (la casa que sirve de garantía).\n")
print("\t- REASON (Variable categórica nominal): Motivo del préstamo. DebtCon = Consolidación de deuda, HomeImp = Mejoras en el hogar.\n")
print("\t- JOB (Variable categórica nominal): Categoría ocupacional (Mgr, Office, Other, ProfExe, Sales, Self).\n")
print("\t- YOJ (Variable numérica continua): Años de antigüedad en el trabajo actual.\n")
print("\t- DEROG (Variable numérica discreta): Número de informes negativos/despectivos importantes en el historial crediticio.\n")
print("\t- DELINQ (Variable numérica discreta): Número de líneas de crédito morosas (pagos atrasados).\n")
print("\t- CLAGE (Variable numérica continua): Antigüedad de la línea de crédito más antigua (en meses).\n")
print("\t- NINQ (Variable numérica discreta): Número de consultas de crédito recientes.\n")
print("\t- CLNO (Variable numérica discreta): Número total de líneas de crédito existentes.\n")
print("\t- DEBTINC (Variable numérica continua): Ratio Deuda-Ingresos (porcentaje de los ingresos mensuales destinado a pagar deudas).\n")

print("========== EDA ==========")

print("Información básica del dataset:")

df.head(10)

print("Proporción de la variable objetivo:")

balance_y = df['BAD'].value_counts(normalize=True)
balance_y

df.describe()

df.info()

print("Todos los nulos:\n", df.isnull().sum()[df.isnull().sum() > 0])

print("Detección de outliers con IQR:")
def iqr_outliers(data, variable):
    Q1 = data[variable].quantile(0.25)
    Q3 = data[variable].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = data[(data[variable] < lower) | (data[variable] > upper)]
    return outliers, lower, upper

for variable in df[["LOAN", "MORTDUE", "VALUE", "YOJ", "DEROG", "DELINQ", "CLAGE", "NINQ", "CLNO", "DEBTINC"]]:
    outliers, lower, upper = iqr_outliers(df, variable)
    print(f"Outliers en {variable}: {len(outliers)}")
    print(f"Rango intercuartílico en {variable}: ({lower:.2f}, {upper:.2f})\n")

print("Visualizamos los datos con gráficas:")

print("Histograma de distribuciones:")

var_num = ["LOAN", "MORTDUE", "VALUE", "YOJ", "DEROG", "DELINQ", "CLAGE", "NINQ", "CLNO", "DEBTINC"]

fig, axes = plt.subplots(2, 5, figsize=(20, 15))
axes = axes.flatten()

for i, col in enumerate(df[var_num]):
    ax = axes[i]
    sns.histplot(df[col], bins=20, kde=True, color="steelblue", ax=ax)
    ax.set_xlabel(col)
    ax.set_ylabel("Frecuencia")
    ax.set_title(f"Distribución de {col}")

plt.tight_layout()
plt.show()

print("Matriz de correlación:")

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Matriz de Correlación")
plt.show()

print("Podemos detectar 5 puntos claves tras el EDA:\n")
print("\t- Hay muchas variables con demasiados nulos, la más afectada es DEBTINC. Usaremos la imputación por mediana para las varibles numéricas y por moda para las categóricas.")
print("\t- La variable objetivo BAD está extremadamente desbalanceada. Usaremos stratify ajustaremos el umbral para intentar paliarlo. Las mejores métricas en este caso serán Precision/Recall/AUC antes que Accuracy.")
print("\t- No parece haber variables con demasiado peso para y, pero si hay demasiada correlación entre VALUE y MORTDUE, asi que es posible que haya que ejercer ingienería de características.")
print("\t- Varias variables numéricas (Sobre todo las relacionadas directamente con dinero) deberán ser ajustadas con una transformación logarítmica para acercarlas a una distribución normal")
print("\t- Variables numéricas con escalas desproporcionadas, las escalaremos entre 0 y 1 para que el modelo no sufra.")
print("\t- Variables categóricas que serán necesarias codificarlas con OneHot para que el modelo no interprete ningún orden.")

print("========== Preprocesamiento ==========")

df_0 = df.copy()

print("Aplicaremos ingienería de características con 2 nuevas variables sintéticas:\n")
print("\t- LTV (Loan-to-Value)(LOAN/VALUE): Cómo de arriesgado es el préstamo respecto al valor de la casa.")
print("\t- EQUITY (Equidad disponible)(VALUE-MORTDUE): Cuánto dinero libre tiene el cliente en su casa antes de pedir el nuevo préstamo.")
print("\t- DEBTINC_MISSING: Para decirle a la red si eran valores faltantes o no.")

temp_VALUE = df_0['VALUE'].fillna(df['VALUE'].median())
temp_MORTDUE = df_0['MORTDUE'].fillna(df['MORTDUE'].median())

df_0['LTV'] = df_0["LOAN"] / temp_VALUE
df_0['EQUITY'] = temp_VALUE - temp_MORTDUE

df_0['DEBTINC_MISSING'] = df_0['DEBTINC'].isnull().astype(int)

df_0[["LTV", "EQUITY", "DEBTINC_MISSING"]]

print("Creación de Pipelines:")

var_objetivo = 'BAD'
var_trans_log = ['LOAN', 'MORTDUE', 'VALUE', 'LTV', 'EQUITY', 'DEBTINC']
var_escalar = ['YOJ', 'CLAGE', 'CLNO', 'NINQ', 'DEROG', 'DELINQ']
var_cat = ['REASON', 'JOB']
passthrough_var = ['DEBTINC_MISSING']

X = df_0.drop(var_objetivo, axis=1)
y = df_0[var_objetivo]

# Pipeline para variables que necesitan una transformación logarítmica (Imputar -> Log -> Escalar)
log_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('log', FunctionTransformer(np.log1p, validate=False, feature_names_out="one-to-one")), # log(1+x) para evitar log(0)
    ('scaler', StandardScaler())
])

# Pipeline para variables que solo necesitan ser escaladas (Imputar -> Escalar)
scaler_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # Mediana es robusta a outliers
    ('scaler', StandardScaler())
])

# Pipeline para variables categóricas que deben ser codificadas (Imputar Moda -> OneHot)
cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Unimos todo en el ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('log_var', log_pipeline, var_trans_log),
        ('std_var', scaler_pipeline, var_escalar),
        ('cat_var', cat_pipeline, var_cat),
        ('pass', 'passthrough', passthrough_var)
    ],
    verbose_feature_names_out=False # Para nombres de columnas más limpios
)

print("========== División Train, Validation y Test ==========")

X_train_full, X_test, y_train_full, y_test = train_test_split( # Primero dividimos Train y Test, por eso lo llamamos _full
    X, y, test_size=0.15, stratify=y, random_state=42 # random_state para replicabilidad
)

X_train, X_val, y_train, y_val = train_test_split( # Luego dividimos lo que será el Train final y Validation
    X_train_full, y_train_full, test_size=0.176, stratify=y_train_full, random_state=42 # Usamos un tamaño de 0.176, ya que el resultado final es 70/15/15 (La estrategia común)
)

print("Ajustamos el preprocesador con Train y transformamos Train, Validation y Test:")

X_train_procesado = preprocessor.fit_transform(X_train)
X_val_procesado = preprocessor.transform(X_val)
X_test_procesado = preprocessor.transform(X_test)

X_train_procesado = np.nan_to_num(X_train_procesado, posinf=0.0, neginf=0.0) # Medida de seguridad para evitar infinitos
X_val_procesado = np.nan_to_num(X_val_procesado, posinf=0.0, neginf=0.0)
X_test_procesado = np.nan_to_num(X_test_procesado, posinf=0.0, neginf=0.0)

print("Convertimos en Arrays para asegurarnos de que son tensores válidos:")

X_train_procesado = np.array(X_train_procesado)
X_val_procesado = np.array(X_val_procesado)
X_test_procesado = np.array(X_test_procesado)

y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)

print("Observamos como han cambiado los datos antes de entrenar al modelo:")

nombre_var = preprocessor.get_feature_names_out()
X_train_df = pd.DataFrame(X_train_procesado, columns=nombre_var)

print(f"Features finales ({len(nombre_var)}): {list(nombre_var)}")
print(f"Dimensiones Train: {X_train_procesado.shape}\n")

print(X_train_df.head())

input_shape = X_train_procesado.shape[1]
print("\nForma del input:", input_shape)

print("==================== Fase 2 ====================")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks

print("Primero calculamos cuanto peso debe tener la clase minoritaria, para penalizar más fuertemente un error en ella:")

negativos = len(y_train[y_train == 0])
positivos = len(y_train[y_train == 1])

peso_0 = (1 / negativos) * ((negativos + positivos) / 2) # Con esto conseguimos igualar la fuerza necesaria para balancear cada peso
peso_1 = (1 / positivos) * ((negativos + positivos) / 2)

pesos = {0: peso_0, 1: peso_1}
pesos

print("========== Definición de la arquitectura ==========")

def crear_modelo(input_shape):
    model = keras.Sequential([
        keras.Input(shape=(input_shape,)),

        layers.Dense(64, activation='relu', input_shape=(input_shape,),
                     kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.3),

        layers.Dense(32, activation='relu',
                     kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.2),

        layers.Dense(16, activation='relu'),

        layers.Dense(1, activation='sigmoid',
                     bias_initializer=keras.initializers.Constant(np.log([positivos/negativos])))])
    return model

print("========== Compilación y entrenamiento ==========")

print("Primero, probamos con un Learning Rate inusual:")

modelo_LR_variado = crear_modelo(input_shape=21) # Ya que nuestro input es de 21

metrics = [
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.AUC(name='auc'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.Precision(name='precision')
]

modelo_LR_variado.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01, clipnorm=1), # Adam es el estándar actual, clipnorm=1 evita que los gradientes sean mayores a 1, evitando NaNs, el LR es el estándar
    loss='binary_crossentropy', # Función de pérdida obligatoria para output binario (0/1)
    metrics=metrics
)

modelo_LR_variado.summary()

print("Configuramos early_stopping con callbacks:")

early_stopping = callbacks.EarlyStopping(
    monitor='val_auc', # Vigilamos el AUC en validación (métrica más robusta que loss)
    verbose=1,
    patience=15, # Si no mejora en 15 épocas, paramos (ahorra tiempo y evita overfitting)
    mode='max', # Ya que necesitamos saber el máximo AUC alcanzado
    restore_best_weights=True # Al final, nos quedamos con la mejor versión del modelo
)

checkpoint = callbacks.ModelCheckpoint(
    filepath='modelo_LR_variado_hmeq.keras',
    monitor='val_auc',
    mode='max',
    save_best_only=True,
    verbose=1
)

print("Iniciando entrenamiento...\n")
history_LR_variado = modelo_LR_variado.fit(
    X_train_procesado, y_train,
    batch_size=32,
    epochs=100,
    validation_data=(X_val_procesado, y_val),
    class_weight=pesos, # Aplicamos pesos para reducir el desbalanceo
    callbacks=[early_stopping, checkpoint],
    verbose=1
)

print("Ahora, realizamos el msimo modelo con un LR estándar:")

mejor_modelo = crear_modelo(input_shape=21)

metrics = [
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.AUC(name='auc'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.Precision(name='precision')
]

mejor_modelo.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001, clipnorm=1),
    loss='binary_crossentropy',
    metrics=metrics
)

mejor_modelo.summary()

early_stopping = callbacks.EarlyStopping(
    monitor='val_auc',
    verbose=1,
    patience=15,
    mode='max',
    restore_best_weights=True
)

checkpoint = callbacks.ModelCheckpoint(
    filepath='mejor_modelo_hmeq.keras',
    monitor='val_auc',
    mode='max',
    save_best_only=True,
    verbose=1
)

print("Iniciando entrenamiento...\n")
history_mejor_modelo = mejor_modelo.fit(
    X_train_procesado, y_train,
    batch_size=32,
    epochs=100,
    validation_data=(X_val_procesado, y_val),
    class_weight=pesos,
    callbacks=[early_stopping, checkpoint],
    verbose=1
)

print("=================== Fase 3 ====================")

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve

print("Cargamos el modelo con Learning Rate variado:")

modelo_LR_variado = keras.models.load_model("modelo_LR_variado_hmeq.keras")

print("Cargamos el mejor modelo:")

mejor_modelo = keras.models.load_model("mejor_modelo_hmeq.keras")

print("========== Visualización del entrenamiento ==========")

print("Veamos las diferencias entre modelos:")

def comparar_historiales(h_estandar, h_variado):
    plt.figure(figsize=(15, 6))

    # Gráfico de LOSS
    plt.subplot(1, 2, 1)
    # Modelo Estándar (LR 0.001)
    plt.plot(h_estandar.history['loss'], 'b-', label='Train Estándar (0.001)')
    plt.plot(h_estandar.history['val_loss'], 'r--', label='Val Estándar (0.001)')
    # Modelo Variado (LR 0.01)
    plt.plot(h_variado.history['loss'], 'g-', label='Train Variado (0.01)')
    plt.plot(h_variado.history['val_loss'], 'y--', label='Val Variado (0.01)')

    plt.title('Comparativa de Convergencia: LOSS')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid(True)

    # Gráfico de AUC
    plt.subplot(1, 2, 2)
    plt.plot(h_estandar.history['auc'], 'b-', label='Train Estándar')
    plt.plot(h_estandar.history['val_auc'], 'r--', label='Val Estándar')
    plt.plot(h_variado.history['auc'], 'g-', label='Train Variado')
    plt.plot(h_variado.history['val_auc'], 'y--', label='Val Variado')

    plt.title('Comparativa de Rendimiento: AUC')
    plt.xlabel('Épocas')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Llamada correcta asegurando que las variables existen
if 'history_mejor_modelo' in locals() and 'history_LR_variado' in locals():
    comparar_historiales(history_mejor_modelo, history_LR_variado)
else:
    print("Error: No se encuentran los historiales de entrenamiento.")

print("========== Predicciones ==========")

y_pred_probs = mejor_modelo.predict(X_test_procesado, batch_size=32, verbose=0)

y_pred_probs = y_pred_probs.ravel() # Aplanamos la matriz par facilitar cálculos

fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend()
plt.grid(True)
plt.show()

print(f"\nAUC final en Test: {roc_auc:.4f}")

print("========== Ajustamos el umbral ==========")

umbrales = [0.5, 0.3, 0.7]

for umbral in umbrales:
    # Convertimos probabilidad a clase 0/1 según el umbral actual
    y_pred_class = (y_pred_probs >= umbral).astype(int)

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred_class)
    tn, fp, fn, tp = cm.ravel()

    # Métricas manuales
    recall = tp / (tp + fn) # Capacidad de detectar fraude
    precision = tp / (tp + fp) # Fiabilidad de la alarma

    # Visualización de matriz
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Matriz de Confusión (Umbral: {umbral:.2f})')
    plt.xlabel('Predicción')
    plt.ylabel('Realidad')
    plt.show()

    print(f"TP (Fraudes detectados): {tp}")
    print(f"FN (Fraudes NO detectados, nuestra prioridad): {fn}")
    print(f"FP (Clientes molestados): {fp}")
    print(f"TN (Clientes NO molestados): {tn}")
    print(f"Recall: {recall:.4f} | Precision: {precision:.4f}\n")

print("========== Reporte final ==========")

mejor_umbral = 0.3 # Según las métricas dadas
y_final_pred = (y_pred_probs >= mejor_umbral).astype(int)

print(classification_report(y_test, y_final_pred, target_names=['Solvente (0)', 'Insolvente (1)']))