import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import time
import json
import joblib
import warnings

warnings.filterwarnings("ignore")

# =============================================================
# CONFIGURACIÓN DE RUTAS
# =============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_VAL = os.path.join(BASE_DIR, "data", "processed", "val_dataset.xlsx")
PATH_MODELS = os.path.join(BASE_DIR, "models")
PATH_REPORTS = os.path.join(BASE_DIR, "reports")

# Artefactos generados por script 03
PATH_CHAMPION_MODEL = os.path.join(PATH_MODELS, "champion_model.pkl")
PATH_CHAMPION_META = os.path.join(PATH_MODELS, "champion_meta.json")
PATH_LE_DESC = os.path.join(PATH_MODELS, "le_descripcion.pkl")
PATH_LE_CAT = os.path.join(PATH_MODELS, "le_categoria.pkl")

# Salidas de este script
OUTPUT_EXCEL = os.path.join(PATH_REPORTS, "prediccion_demanda_logistica.xlsx")
OUTPUT_PLOT = os.path.join(PATH_REPORTS, "feature_importance_xai.png")
# Ruta final del .pkl de producción (renombrado para claridad)
OUTPUT_MODEL_PROD = os.path.join(PATH_MODELS, "xgboost_preventive.pkl")

os.makedirs(PATH_MODELS, exist_ok=True)
os.makedirs(PATH_REPORTS, exist_ok=True)

print("=" * 60)
print("  FASES V y VI — INFERENCIA Y OPERACIONALIZACIÓN")
print("  Modelo: XGBoost Optimizado (Campeón)")
print("=" * 60)

# =============================================================
# [A6] PASO 1 — LEER METADATOS DEL CAMPEÓN
# champion_meta.json generado por script 03 contiene las
# features exactas, hiperparámetros y métricas del campeón.
# Esto garantiza sincronización total sin hardcodear nada.
# =============================================================
print("\n[1] Cargando metadatos del modelo campeón...")

try:
    with open(PATH_CHAMPION_META, "r") as f:
        meta = json.load(f)
except FileNotFoundError:
    print("   No se encontro champion_meta.json")
    print("   Ejecuta primero el script 03_entrenamiento_y_validacion.py")
    exit()

FEATURES = meta["features"]  # lista exacta de features del campeón
NOMBRE = meta["nombre"]
MAE_TRAIN = meta["mae"]
RMSE_TRAIN = meta["rmse"]
R2_TRAIN = meta["r2"]

print(f"   Campeon     : {NOMBRE}")
print(f"   MAE val     : {MAE_TRAIN}")
print(f"   RMSE val    : {RMSE_TRAIN}")
print(f"   R²  val     : {R2_TRAIN}")
print(f"   Features ({len(FEATURES)}): {FEATURES}")

# =============================================================
# [A2] PASO 2 — CARGAR EL MODELO CAMPEÓN REAL
# El script original reentrenaba con parámetros incorrectos
# (max_depth=3 en vez de 7, subsample=0.7 en vez de 0.85).
# Ahora cargamos directamente el .pkl del campeón real.
# =============================================================
print("\n[2] Cargando modelo campeón desde disco...")

try:
    champion_model = joblib.load(PATH_CHAMPION_MODEL)
    print(f"   champion_model.pkl cargado correctamente")
    print(f"   Parámetros reales: {champion_model.get_params()}")
except FileNotFoundError:
    print("   No se encontro champion_model.pkl")
    print("   Ejecuta primero el script 03_entrenamiento_y_validacion.py")
    exit()

# Copiar como xgboost_preventive.pkl (nombre de producción)
joblib.dump(champion_model, OUTPUT_MODEL_PROD)
print(f"   Artefacto de produccion: models/xgboost_preventive.pkl")
print(f"   Tamaño: {os.path.getsize(OUTPUT_MODEL_PROD)/1024:.2f} KB")

# =============================================================
# [A4] PASO 3 — CARGAR ENCODERS Y PREPARAR DATOS
# Los LabelEncoders guardados en el script 03 garantizan
# que descripcion y categoria se codifiquen exactamente igual
# que durante el entrenamiento. Sin esto el modelo falla.
# =============================================================
print("\n[3] Cargando LabelEncoders y preparando datos de validacion...")

try:
    le_desc = joblib.load(PATH_LE_DESC)
    le_cat = joblib.load(PATH_LE_CAT)
    print("   LabelEncoders cargados")
except FileNotFoundError:
    print("   No se encontraron los LabelEncoders")
    print("   Ejecuta primero el script 03_entrenamiento_y_validacion.py")
    exit()

df_val = pd.read_excel(PATH_VAL)
df_val["fecha"] = pd.to_datetime(df_val["fecha"])

# Aplicar encoders a las columnas categóricas
# [A1] Usamos FEATURES del meta — no hardcodeamos nada
df_val["desc_encoded"] = le_desc.transform(df_val["descripcion"])
df_val["cat_encoded"] = le_cat.transform(df_val["categoria"])

# Verificar que todas las features del campeón estén presentes
faltantes = [f for f in FEATURES if f not in df_val.columns]
if faltantes:
    print(f" Features faltantes en val_dataset: {faltantes}")
    exit()

print(f"    Dataset listo: {len(df_val)} filas · {len(FEATURES)} features")

# =============================================================
# [A1] PASO 4 — PRUEBA DE ESTRÉS (KPI TECNOLÓGICO)
# Usamos exactamente las mismas features con las que se
# entrenó el campeón — leídas desde champion_meta.json
# =============================================================
print("\n[4] Prueba de estres — Inferencia por lotes (Batch Inference)...")

df_batch = df_val.head(71).copy()

# [A1] X_batch usa FEATURES del meta — mismo orden exacto
X_batch = df_batch[FEATURES]

# INICIO CRONÓMETRO
start_time = time.time()
preds_batch = champion_model.predict(X_batch)
end_time = time.time()

latencia = end_time - start_time
print(f"   SKUs procesados  : {len(df_batch)}")
print(f"   Latencia total   : {latencia:.4f} segundos")
print(f"   Latencia por SKU : {latencia/len(df_batch)*1000:.3f} ms/prediccion")

SLA_SEGUNDOS = 4.0
if latencia < SLA_SEGUNDOS:
    print(f"       EXITO — Latencia < {SLA_SEGUNDOS}s (Cumple SLA on-premise)")
else:
    print(f"       ALERTA — Latencia supera el SLA de {SLA_SEGUNDOS}s")

# =============================================================
# PASO 5 — INFERENCIA COMPLETA SOBRE TODO EL VAL SET
# =============================================================
print("\n[5] Ejecutando inferencia completa sobre val_dataset...")

X_val_full = df_val[FEATURES]
preds_full = champion_model.predict(X_val_full)
preds_full = np.array([max(0, round(p)) for p in preds_full])

# =============================================================
# [A3] PASO 6 — REPORTE EXCEL CON REGLAS DE NEGOCIO POR RANGOS
# El original tenía un solo umbral (x > 0 → "REPONER MÍNIMOS")
# lo que hacía que toda predicción dijera lo mismo.
# Ahora usamos rangos que permiten priorizar la reposición.
# =============================================================
print("\n[6] Generando reporte de negocio para logistica...")


def regla_accion(pred):
    """
    Reglas de negocio basadas en unidades predichas de demanda.
    Permiten al jefe de logística priorizar órdenes de compra.
    """
    if pred >= 10:
        return "🔴 CRÍTICO — Reponer urgente"
    elif pred >= 5:
        return "🟠 REPONER — Programar compra"
    elif pred >= 1:
        return "🟡 MONITOREAR — Revisión próxima semana"
    else:
        return "🟢 STOCK OK"


df_reporte = df_val.copy()
df_reporte["Prediccion_IA"] = preds_full
df_reporte["Accion_Sugerida"] = df_reporte["Prediccion_IA"].apply(regla_accion)

# [A5] Reporte enriquecido con categoría y precio para priorizar por valor
cols_reporte = [
    "fecha",
    "descripcion",
    "categoria",  # [A5] nueva — permite agrupar por tipo
    "precio_unitario",  # [A5] nueva — valor económico del producto
    "cantidad",  # cantidad real (para comparar con predicción)
    "Prediccion_IA",
    "Accion_Sugerida",
]
cols_final = [c for c in cols_reporte if c in df_reporte.columns]
df_export = df_reporte[cols_final].sort_values(
    "Prediccion_IA", ascending=False  # los más urgentes primero
)

df_export.to_excel(OUTPUT_EXCEL, index=False)

# Resumen de acciones para consola
print(f"\n   Resumen de acciones sugeridas:")
resumen = df_reporte["Accion_Sugerida"].value_counts()
for accion, count in resumen.items():
    print(f"   {accion:<40} : {count} productos")

print(f"\n    Reporte guardado → reports/prediccion_demanda_logistica.xlsx")

# =============================================================
# PASO 7 — GRÁFICO XAI — INTERPRETABILIDAD DEL MODELO
# =============================================================
print("\n[7] Generando grafico de interpretabilidad (XAI)...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Panel 1: Importancia por peso (número de veces que se usa la feature)
xgb.plot_importance(
    champion_model,
    importance_type="weight",
    max_num_features=9,
    color="#4C72B0",
    ax=axes[0],
)
axes[0].set_title(
    "Importancia por Frecuencia de Uso\n(Weight)", fontsize=11, fontweight="bold"
)
axes[0].set_xlabel("Score F (veces usada en splits)")
axes[0].grid(axis="x", linestyle="--", alpha=0.4)
axes[0].spines[["top", "right"]].set_visible(False)

# Panel 2: Importancia por ganancia (cuánto mejora el modelo)
xgb.plot_importance(
    champion_model,
    importance_type="gain",
    max_num_features=9,
    color="#2A9D8F",
    ax=axes[1],
)
axes[1].set_title(
    "Importancia por Ganancia Predictiva\n(Gain)", fontsize=11, fontweight="bold"
)
axes[1].set_xlabel("Ganancia media por split")
axes[1].grid(axis="x", linestyle="--", alpha=0.4)
axes[1].spines[["top", "right"]].set_visible(False)

plt.suptitle(
    f"Transparencia Algoritmica (XAI) — Modelo Campeon: {NOMBRE}\n"
    f"MAE={MAE_TRAIN} · RMSE={RMSE_TRAIN} · R²={R2_TRAIN}",
    fontsize=13,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches="tight")
plt.close()
print(f"    Grafico XAI guardado → reports/feature_importance_xai.png")

# =============================================================
# RESUMEN FINAL
# =============================================================
print("\n" + "=" * 60)
print("  RESUMEN DE OPERACIONALIZACIÓN")
print("=" * 60)
print(f"  Modelo desplegado  : {NOMBRE}")
print(f"  Métricas (val set) : MAE={MAE_TRAIN} · RMSE={RMSE_TRAIN} · R²={R2_TRAIN}")
print(f"  Latencia inferencia: {latencia:.4f}s para {len(df_batch)} SKUs")
print(f"  Features usadas    : {len(FEATURES)}")
print(f"  Registros evaluados: {len(df_val)}")
print(f"\n  Artefactos generados:")
print(f"  · models/xgboost_preventive.pkl   ← modelo de producción")
print(f"  · reports/prediccion_demanda_logistica.xlsx")
print(f"  · reports/feature_importance_xai.png")
print("=" * 60)
print("\n--- PIPELINE MLOps COMPLETO ✅ ---")
print("   01_etl → 02_transformacion → 03_entrenamiento → 04_inferencia")
print("\n   Siguiente paso: integrar xgboost_preventive.pkl en preventive-web")
