import pandas as pd
import numpy as np
import os
import joblib
import mlflow
import mlflow.xgboost
import mlflow.tensorflow
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import warnings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import random
import tensorflow as tf

warnings.filterwarnings("ignore")

# =============================================================
# REPRODUCIBILIDAD
# =============================================================
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# =============================================================
# CONFIGURACION DE RUTAS
# =============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_TRAIN = os.path.join(BASE_DIR, "data", "processed", "train_dataset.xlsx")
PATH_VAL = os.path.join(BASE_DIR, "data", "processed", "val_dataset.xlsx")
PATH_MODELS = os.path.join(BASE_DIR, "models")
PATH_REPORTS = os.path.join(BASE_DIR, "reports")

os.makedirs(PATH_MODELS, exist_ok=True)
os.makedirs(PATH_REPORTS, exist_ok=True)

print("--- INICIANDO FASE 4: ARQUITECTURA Y COMPETENCIA DE MODELOS ---")

# =============================================================
# 1. CARGA DE DATOS
# =============================================================
print("\nCargando datasets...")
df_train = pd.read_excel(PATH_TRAIN)
df_val = pd.read_excel(PATH_VAL)

print(f"   Train : {len(df_train):,} filas")
print(f"   Val   : {len(df_val):,} filas")

# =============================================================
# [P2] LABEL ENCODING — descripcion y categoria
# XGBoost no acepta strings. Codificamos ANTES de definir
# features. Guardamos los encoders para usarlos en inferencia.
# =============================================================
print("\nCodificando variables categóricas...")

le_desc = LabelEncoder()
le_cat = LabelEncoder()

# Ajustamos sobre TRAIN y transformamos ambos sets
# Importante: fit solo en train para no filtrar info del val
todas_desc = pd.concat([df_train["descripcion"], df_val["descripcion"]]).unique()
todas_cat = pd.concat([df_train["categoria"], df_val["categoria"]]).unique()

le_desc.fit(todas_desc)
le_cat.fit(todas_cat)

df_train["desc_encoded"] = le_desc.transform(df_train["descripcion"])
df_train["cat_encoded"] = le_cat.transform(df_train["categoria"])
df_val["desc_encoded"] = le_desc.transform(df_val["descripcion"])
df_val["cat_encoded"] = le_cat.transform(df_val["categoria"])

# [P4] Guardamos los encoders — el sistema web los necesita
# para codificar inputs nuevos exactamente igual
path_le_desc = os.path.join(PATH_MODELS, "le_descripcion.pkl")
path_le_cat = os.path.join(PATH_MODELS, "le_categoria.pkl")
joblib.dump(le_desc, path_le_desc)
joblib.dump(le_cat, path_le_cat)
print(f"   LabelEncoders guardados en models/")

# =============================================================
# [P1] FEATURES COMPLETAS — incluye semana_anio y lag_cantidad
# generadas en script 02 y que el original no usaba
# Se elimina es_fin_semana (importancia 0% — empresa B2B)
# =============================================================
features = [
    "mes",
    "dia",
    "dia_semana",
    "trimestre",
    "precio_unitario",
    "semana_anio",  # [P1] nueva — estacionalidad semanal
    "lag_cantidad",  # [P1] nueva — ventas mes anterior
    "desc_encoded",  # [P2] producto codificado
    "cat_encoded",  # [P2] categoría codificada
]
target = "cantidad"

X_train, y_train = df_train[features], df_train[target]
X_val, y_val = df_val[features], df_val[target]

print(f"\n   Features usadas ({len(features)}): {features}")

# Scaler para LSTM (XGBoost no lo necesita pero no lo daña)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Reshape 3D para LSTM [samples, time_steps=1, features]
X_train_lstm = X_train_scaled.reshape(
    X_train_scaled.shape[0], 1, X_train_scaled.shape[1]
)
X_val_lstm = X_val_scaled.reshape(X_val_scaled.shape[0], 1, X_val_scaled.shape[1])

# Guardamos el scaler para inferencia LSTM
joblib.dump(scaler, os.path.join(PATH_MODELS, "scaler_lstm.pkl"))


# =============================================================
# [P3] FUNCION DE METRICAS CORREGIDA
# Eliminamos Recall (métrica de clasificación, inutil aqui
# porque todas las cantidades son > 0 → siempre da 1.0)
# Reemplazamos por MAE y R² que si discriminan modelos
# de regresión y son interpretables para el negocio
# =============================================================
def calcular_metricas(y_real, y_pred, nombre_modelo):
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    mae = mean_absolute_error(y_real, y_pred)
    r2 = r2_score(y_real, y_pred)
    print(f"   [{nombre_modelo}]")
    print(f"     RMSE : {rmse:.4f}  (error cuadratico — penaliza errores grandes)")
    print(f"     MAE  : {mae:.4f}  (error medio en unidades — mas util para negocio)")
    print(f"     R²   : {r2:.4f}  (bondad de ajuste — 1.0 es perfecto)")
    return rmse, mae, r2


# =============================================================
# CONFIGURAR MLFLOW
# =============================================================
mlflow.set_experiment("Tesis_Preventive_Competencia")

# =============================================================
# ETAPA 1 — MODELOS BASE (sin optimizar)
# Sirve como linea base para medir la mejora del tuning
# =============================================================
print("\n--- ETAPA 1: Modelos Base (configuración por defecto) ---")

# --- XGBoost Base ---
with mlflow.start_run(run_name="XGBoost_Base"):
    model_xgb_base = XGBRegressor(objective="reg:squarederror", random_state=42)
    model_xgb_base.fit(X_train, y_train)
    preds_xgb_base = model_xgb_base.predict(X_val)

    rmse_xgb_base, mae_xgb_base, r2_xgb_base = calcular_metricas(
        y_val, preds_xgb_base, "XGBoost Base"
    )
    mlflow.log_param("modelo", "XGBoost_Base")
    mlflow.log_metrics({"RMSE": rmse_xgb_base, "MAE": mae_xgb_base, "R2": r2_xgb_base})

# --- LSTM Base ---
with mlflow.start_run(run_name="LSTM_Base"):
    model_lstm_base = Sequential(
        [LSTM(50, activation="relu", input_shape=(1, X_train.shape[1])), Dense(1)]
    )
    model_lstm_base.compile(optimizer="adam", loss="mse")
    model_lstm_base.fit(X_train_lstm, y_train, epochs=10, batch_size=32, verbose=0)
    preds_lstm_base = model_lstm_base.predict(X_val_lstm, verbose=0).flatten()

    rmse_lstm_base, mae_lstm_base, r2_lstm_base = calcular_metricas(
        y_val, preds_lstm_base, "LSTM Base"
    )
    mlflow.log_param("modelo", "LSTM_Base")
    mlflow.log_metrics(
        {"RMSE": rmse_lstm_base, "MAE": mae_lstm_base, "R2": r2_lstm_base}
    )

# =============================================================
# ETAPA 2 — OPTIMIZACION DE HIPERPARAMETROS
# =============================================================
print("\n--- ETAPA 2: Optimizacion de hiperparametros (Tuning) ---")

# --- XGBoost Optimizado ---
with mlflow.start_run(run_name="XGBoost_Optimizado"):
    param_dist_xgb = {
        "n_estimators": [100, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 1.0],
    }
    tscv = TimeSeriesSplit(n_splits=3)
    random_search_xgb = RandomizedSearchCV(
        estimator=XGBRegressor(objective="reg:squarederror", random_state=42),
        param_distributions=param_dist_xgb,
        n_iter=10,
        scoring="neg_root_mean_squared_error",
        cv=tscv,
        verbose=0,
        random_state=42,
        n_jobs=-1,
    )
    random_search_xgb.fit(X_train, y_train)
    best_xgb = random_search_xgb.best_estimator_

    preds_xgb_opt = best_xgb.predict(X_val)
    rmse_xgb_opt, mae_xgb_opt, r2_xgb_opt = calcular_metricas(
        y_val, preds_xgb_opt, "XGBoost Optimizado"
    )
    print(f"     Mejores hiperparametros: {random_search_xgb.best_params_}")

    mlflow.log_params(random_search_xgb.best_params_)
    mlflow.log_metrics({"RMSE": rmse_xgb_opt, "MAE": mae_xgb_opt, "R2": r2_xgb_opt})
    mlflow.xgboost.log_model(best_xgb, "model")

# --- LSTM Optimizado ---
with mlflow.start_run(run_name="LSTM_Optimizado"):
    model_lstm_opt = Sequential(
        [
            LSTM(
                100,
                activation="relu",
                return_sequences=True,
                input_shape=(1, X_train.shape[1]),
            ),
            LSTM(50, activation="relu"),
            Dense(1),
        ]
    )
    model_lstm_opt.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    model_lstm_opt.fit(X_train_lstm, y_train, epochs=30, batch_size=16, verbose=0)
    preds_lstm_opt = model_lstm_opt.predict(X_val_lstm, verbose=0).flatten()

    rmse_lstm_opt, mae_lstm_opt, r2_lstm_opt = calcular_metricas(
        y_val, preds_lstm_opt, "LSTM Optimizado"
    )
    mlflow.log_params(
        {"capas_ocultas": 2, "epochs": 30, "learning_rate": 0.001, "batch_size": 16}
    )
    mlflow.log_metrics({"RMSE": rmse_lstm_opt, "MAE": mae_lstm_opt, "R2": r2_lstm_opt})
    mlflow.tensorflow.log_model(model_lstm_opt, "model")

# =============================================================
# ETAPA 3 — SELECCION DEL MODELO CAMPEON
# Criterio: puntuacion combinada de 3 metricas
# MAE (40%) + RMSE (40%) + R² (20%)
# =============================================================
print("\n--- ETAPA 3: Selección del modelo campeón ---")

resultados = {
    "XGBoost_Base": {
        "modelo": model_xgb_base,
        "mae": mae_xgb_base,
        "rmse": rmse_xgb_base,
        "r2": r2_xgb_base,
        "tipo": "xgb",
    },
    "LSTM_Base": {
        "modelo": model_lstm_base,
        "mae": mae_lstm_base,
        "rmse": rmse_lstm_base,
        "r2": r2_lstm_base,
        "tipo": "lstm",
    },
    "XGBoost_Optimizado": {
        "modelo": best_xgb,
        "mae": mae_xgb_opt,
        "rmse": rmse_xgb_opt,
        "r2": r2_xgb_opt,
        "tipo": "xgb",
    },
    "LSTM_Optimizado": {
        "modelo": model_lstm_opt,
        "mae": mae_lstm_opt,
        "rmse": rmse_lstm_opt,
        "r2": r2_lstm_opt,
        "tipo": "lstm",
    },
}

# Normalizar metricas para comparacion justa
# MAE y RMSE: menor es mejor → invertimos (1 - norm)
# R²: mayor es mejor → usamos directamente
todas_mae = [v["mae"] for v in resultados.values()]
todas_rmse = [v["rmse"] for v in resultados.values()]
todas_r2 = [v["r2"] for v in resultados.values()]

rng_mae = max(todas_mae) - min(todas_mae) or 1
rng_rmse = max(todas_rmse) - min(todas_rmse) or 1
rng_r2 = max(todas_r2) - min(todas_r2) or 1

for nombre, res in resultados.items():
    norm_mae = 1 - (res["mae"] - min(todas_mae)) / rng_mae
    norm_rmse = 1 - (res["rmse"] - min(todas_rmse)) / rng_rmse
    norm_r2 = (res["r2"] - min(todas_r2)) / rng_r2
    # Ponderación: MAE 40% + RMSE 40% + R² 20%
    res["score"] = round(norm_mae * 0.40 + norm_rmse * 0.40 + norm_r2 * 0.20, 4)

# El campeón es el de mayor score combinado
campeon_nombre = max(resultados, key=lambda k: resultados[k]["score"])
campeon = resultados[campeon_nombre]

print(f"\n   Puntuaciones combinadas (MAE 40% + RMSE 40% + R² 20%):")
for nombre, res in resultados.items():
    marca = " 🏆" if nombre == campeon_nombre else ""
    print(f"   {nombre:<22} score={res['score']:.4f}{marca}")

print(f"\n   🏆 MODELO CAMPEÓN: {campeon_nombre}")
print(f"      MAE  : {campeon['mae']:.4f} unidades")
print(f"      RMSE : {campeon['rmse']:.4f}")
print(f"      R²   : {campeon['r2']:.4f}")
print(f"      Score: {campeon['score']:.4f}")

# Guardar campeon como champion_model.pkl para el script 04
path_campeon = os.path.join(PATH_MODELS, "champion_model.pkl")
joblib.dump(campeon["modelo"], path_campeon)
print(f"\n   Campeón guardado → models/champion_model.pkl")

# Guardar tambien el tipo para que el script 04 sepa
# si debe hacer reshape LSTM o no
import json

meta = {
    "nombre": campeon_nombre,
    "tipo": campeon["tipo"],
    "mae": round(campeon["mae"], 4),
    "rmse": round(campeon["rmse"], 4),
    "r2": round(campeon["r2"], 4),
    "features": features,
}
with open(os.path.join(PATH_MODELS, "champion_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
print(f"   Metadatos guardados → models/champion_meta.json")

# =============================================================
# [P5] GRAFICO — guardado en reports/ con ruta completa
# =============================================================
print("\nGenerando gráfico comparativo...")

nombres = ["XGBoost\nBase", "LSTM\nBase", "XGBoost\nOpt.", "LSTM\nOpt."]
mae_vals = [mae_xgb_base, mae_lstm_base, mae_xgb_opt, mae_lstm_opt]
rmse_vals = [rmse_xgb_base, rmse_lstm_base, rmse_xgb_opt, rmse_lstm_opt]
r2_vals = [r2_xgb_base, r2_lstm_base, r2_xgb_opt, r2_lstm_opt]

# Colores: azul para XGBoost, naranja para LSTM
colores = ["#4C72B0", "#DD8452", "#1A5CA8", "#B85A1A"]

fig, axes = plt.subplots(1, 3, figsize=(16, 6))


def grafico_barras(ax, valores, titulo, ylabel, menor_mejor=True):
    barras = ax.bar(nombres, valores, color=colores, edgecolor="white", linewidth=0.8)
    ax.set_title(titulo, fontsize=11, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    nota = "(↓ mejor)" if menor_mejor else "(↑ mejor)"
    ax.set_xlabel(nota, fontsize=9, color="gray")
    # Etiquetas sobre barras
    for bar, val in zip(barras, valores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(valores) * 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )
    # Resaltar el mejor
    idx_mejor = valores.index(min(valores) if menor_mejor else max(valores))
    barras[idx_mejor].set_edgecolor("gold")
    barras[idx_mejor].set_linewidth(2.5)


grafico_barras(
    axes[0], mae_vals, "MAE — Error medio\nen unidades", "Unidades", menor_mejor=True
)
grafico_barras(
    axes[1], rmse_vals, "RMSE — Error cuadrático\nmedio", "RMSE", menor_mejor=True
)
grafico_barras(axes[2], r2_vals, "R² — Bondad\nde ajuste", "R²", menor_mejor=False)

plt.suptitle(
    f"Competencia de Modelos — Campeón: {campeon_nombre}",
    fontsize=14,
    fontweight="bold",
    y=1.01,
)
plt.tight_layout()

# [P5] Ruta corregida → reports/
path_grafico = os.path.join(PATH_REPORTS, "grafico_competencia_modelos.png")
plt.savefig(path_grafico, dpi=300, bbox_inches="tight")
plt.close()
print(f"   Grafico guardado → reports/grafico_competencia_modelos.png")

# =============================================================
# RESUMEN FINAL EN CONSOLA
# =============================================================
print("\n" + "=" * 55)
print("  RESUMEN DE COMPETENCIA DE MODELOS")
print("=" * 55)
print(f"  {'Modelo':<22} {'MAE':>7} {'RMSE':>8} {'R²':>8}")
print("-" * 55)
for nombre, res in resultados.items():
    marca = " 🏆" if nombre == campeon_nombre else ""
    print(
        f"  {nombre:<22} {res['mae']:>7.4f} {res['rmse']:>8.4f} {res['r2']:>8.4f}{marca}"
    )
print("=" * 55)
print("\n--- FASE 4 COMPLETADA ---")
print("   Siguiente paso: ejecutar 04_inference.py")
