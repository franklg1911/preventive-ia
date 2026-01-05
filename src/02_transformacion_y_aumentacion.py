import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split

# CONFIGURACIÓN

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_SEED = os.path.join(BASE_DIR, "data", "processed", "dataset_reconstruido.xlsx")
PATH_FINAL = os.path.join(
    BASE_DIR, "data", "processed", "dataset_final_entrenamiento.xlsx"
)
PATH_TRAIN = os.path.join(BASE_DIR, "data", "processed", "train_dataset.xlsx")
PATH_VAL = os.path.join(BASE_DIR, "data", "processed", "val_dataset.xlsx")
PATH_TEST = os.path.join(BASE_DIR, "data", "processed", "test_dataset.xlsx")

# Sin esto cada ejecución genera un dataset diferente
random.seed(42)
np.random.seed(42)

# TARGET_ROWS ajustado de 6000 -> 1300
# Razón: 322 reales + 1300 sinteticos = 1622 total
# Ratio: 80.2% sintetico / 19.8% real
# Un ratio 95/5 diluye demasiado la señal real aprendida en field
TARGET_ROWS = 1300
FECHA_INICIO = datetime(2023, 1, 1)
FECHA_FIN = datetime(2025, 12, 31)

print("--- INICIANDO FASE 3: TRANSFORMACION Y AUMENTACION ---")

# =============================================================
# 1. CARGAR DATA SEMILLA
# =============================================================
try:
    df_seed = pd.read_excel(PATH_SEED)
    df_seed["fecha"] = pd.to_datetime(df_seed["fecha"])
    print(f"Semilla cargada: {len(df_seed)} registros reales")
except Exception as e:
    print(f"Error cargando semilla: {e}")
    exit()

# =============================================================
# 2. AUMENTACION DE DATOS
# Aprendemos distribuciones directamente de la semilla real
# para que los sinteticos respeten el comportamiento observado
# =============================================================
print(f"\nGenerando {TARGET_ROWS} registros sinteticos...")

# Distribucion de productos (respeta frecuencia real)
probs_producto = df_seed["descripcion"].value_counts(normalize=True)
lista_prods = probs_producto.index.tolist()
pesos_prods = probs_producto.values.tolist()

# Mapeos desde la semilla real
dict_precios = df_seed.groupby("descripcion")["precio_maestro"].mean().to_dict()
dict_cats = df_seed.groupby("descripcion")["categoria"].first().to_dict()

# [P2] Distribucion de cantidades aprendida por producto desde la semilla
# En vez de pesos fijos, calculamos la distribución real por producto
# Esto preserva que pedestales se venden en lotes y extintores en unidades
dict_cantidades = df_seed.groupby("descripcion")["cantidad"].apply(list).to_dict()

data_fake = []
dias_totales = (FECHA_FIN - FECHA_INICIO).days

for _ in range(TARGET_ROWS):
    # Fecha aleatoria dentro del rango 2023-2025
    fecha = FECHA_INICIO + timedelta(days=random.randint(0, dias_totales))

    # Producto basado en distribucion real
    prod = random.choices(lista_prods, weights=pesos_prods, k=1)[0]

    # [P2] Cantidad: samplea de las cantidades historicas reales del producto
    # Si el producto tiene historial, toma un valor real con ruido pequeño
    # Si no tiene historial suficiente, usa distribucion conservadora
    cantidades_reales = dict_cantidades.get(prod, [1])
    cant_base = random.choice(cantidades_reales)
    # Ruido ±1 para no repetir exactamente los mismos valores
    cant = max(1, cant_base + random.randint(-1, 1))

    data_fake.append(
        {
            "fecha": fecha,
            "descripcion": prod,
            "categoria": dict_cats.get(prod, "Otros"),
            "cantidad": cant,
            "precio_unitario": round(dict_precios.get(prod, 0), 2),
            "origen_dato": "Sintetico",
        }
    )

df_sint = pd.DataFrame(data_fake)

# =============================================================
# 3. UNIR SEMILLA REAL + SINTETICOS
# =============================================================
df_seed["origen_dato"] = "Real"

if "precio_maestro" in df_seed.columns:
    df_seed = df_seed.rename(columns={"precio_maestro": "precio_unitario"})

cols_comunes = [
    "fecha",
    "descripcion",
    "categoria",
    "cantidad",
    "precio_unitario",
    "origen_dato",
]

df_final = pd.concat([df_seed[cols_comunes], df_sint[cols_comunes]], ignore_index=True)

total = len(df_final)
n_real = (df_final["origen_dato"] == "Real").sum()
n_sint = (df_final["origen_dato"] == "Sintetico").sum()
print(f"\nDataset combinado: {total} registros")
print(f"   Real     : {n_real} ({n_real/total*100:.1f}%)")
print(f"   Sintetico: {n_sint} ({n_sint/total*100:.1f}%)")

# =============================================================
# 4. INGENIERIA DE CARACTERISTICAS
# =============================================================
print("\nProcesando features temporales...")

df_final["fecha"] = pd.to_datetime(df_final["fecha"])
df_final["mes"] = df_final["fecha"].dt.month
df_final["dia"] = df_final["fecha"].dt.day
df_final["dia_semana"] = df_final["fecha"].dt.dayofweek  # 0=lunes, 6=domingo
df_final["trimestre"] = df_final["fecha"].dt.quarter
df_final["es_fin_semana"] = (df_final["dia_semana"] >= 5).astype(int)

# [P4] Feature: semana del año
# Captura estacionalidad semanal — semana 1 vs semana 40 tienen demandas distintas
df_final["semana_anio"] = df_final["fecha"].dt.isocalendar().week.astype(int)

# [P4] Feature: lag_cantidad (ventas del mes anterior por producto)
# Es la feature mas predictiva en demanda tabular:
# "si el mes pasado vendi 10 unidades de X, probablemente este mes venda algo similar"
# Proceso:
#   1. Agrupamos por producto + año + mes -> cantidad total vendida ese mes
#   2. Desplazamos 1 mes hacia adelante (shift) -> eso es el "mes anterior"
#   3. Unimos de vuelta al dataset principal
print("Calculando lag_cantidad (ventas mes anterior por producto)...")

ventas_mensuales = (
    df_final.groupby(
        [
            "descripcion",
            df_final["fecha"].dt.year.rename("anio"),
            df_final["fecha"].dt.month.rename("mes_num"),
        ]
    )["cantidad"]
    .sum()
    .reset_index()
    .rename(columns={"cantidad": "ventas_mes"})
)

# Crear columna de mes siguiente para hacer el join
ventas_mensuales["mes_num_siguiente"] = ventas_mensuales["mes_num"] + 1
ventas_mensuales["anio_siguiente"] = ventas_mensuales["anio"]

# Ajustar cuando mes = 12 -> siguiente es enero del año siguiente
mask_dic = ventas_mensuales["mes_num"] == 12
ventas_mensuales.loc[mask_dic, "mes_num_siguiente"] = 1
ventas_mensuales.loc[mask_dic, "anio_siguiente"] = (
    ventas_mensuales.loc[mask_dic, "anio"] + 1
)

# Join para traer el lag al dataset principal
df_final["anio_tmp"] = df_final["fecha"].dt.year
df_final["mes_tmp"] = df_final["fecha"].dt.month

df_final = df_final.merge(
    ventas_mensuales[
        ["descripcion", "anio_siguiente", "mes_num_siguiente", "ventas_mes"]
    ],
    left_on=["descripcion", "anio_tmp", "mes_tmp"],
    right_on=["descripcion", "anio_siguiente", "mes_num_siguiente"],
    how="left",
).rename(columns={"ventas_mes": "lag_cantidad"})

# El primer mes de cada producto no tiene lag -> rellenamos con la mediana del producto
mediana_por_producto = df_final.groupby("descripcion")["cantidad"].median()
df_final["lag_cantidad"] = df_final.apply(
    lambda r: (
        mediana_por_producto[r["descripcion"]]
        if pd.isna(r["lag_cantidad"])
        else r["lag_cantidad"]
    ),
    axis=1,
)

# Limpiar columnas temporales de ayuda
df_final = df_final.drop(
    columns=["anio_tmp", "mes_tmp", "anio_siguiente", "mes_num_siguiente"],
    errors="ignore",
)

print(f"   lag_cantidad — nulos restantes: {df_final['lag_cantidad'].isna().sum()}")

# Ordenar cronologicamente antes del split
df_final = df_final.sort_values("fecha").reset_index(drop=True)

# =============================================================
# 5. DIVISION TEMPORAL DE DATOS
# =============================================================

print("\nAplicando división aleatoria 70/15/15...")

df_train, df_temp = train_test_split(
    df_final, test_size=0.30, random_state=42, shuffle=True
)
df_val, df_test = train_test_split(
    df_temp, test_size=0.50, random_state=42, shuffle=True
)

# Reordenamiento cronologico dentro de cada set
df_train = df_train.sort_values("fecha").reset_index(drop=True)
df_val = df_val.sort_values("fecha").reset_index(drop=True)
df_test = df_test.sort_values("fecha").reset_index(drop=True)

# =============================================================
# 6. VALIDACION DE COBERTURA TEMPORAL
# =============================================================
print("\nANALISIS DE COBERTURA TEMPORAL:")


def analizar_cobertura(df, nombre):
    anios = sorted(df["fecha"].dt.year.unique())
    meses = df["fecha"].dt.month.nunique()
    print(f"  {nombre}: {len(df):,} registros")
    print(f"    Años presentes : {anios}")
    print(f"    Meses únicos   : {meses}/12")
    print(
        f"    Rango          : {df['fecha'].min().date()} → {df['fecha'].max().date()}"
    )


analizar_cobertura(df_train, "Train")
analizar_cobertura(df_val, "Val  ")
analizar_cobertura(df_test, "Test ")

# =============================================================
# 7. RESUMEN DE FEATURES FINALES
# =============================================================
features_modelo = [
    "mes",
    "dia",
    "dia_semana",
    "trimestre",
    "es_fin_semana",
    "semana_anio",
    "lag_cantidad",
    "precio_unitario",
]

print(f"\nFeatures para el modelo ({len(features_modelo)}):")
for f in features_modelo:
    print(f"   · {f}")
print("   · descripcion  (se codificara en script 03)")
print("   · categoria    (se codificara en script 03)")

# =============================================================
# 8. GUARDAR ARCHIVOS
# =============================================================
df_train.to_excel(PATH_TRAIN, index=False)
df_val.to_excel(PATH_VAL, index=False)
df_test.to_excel(PATH_TEST, index=False)
df_final.to_excel(PATH_FINAL, index=False)

print(f"\nArchivos guardados en data/processed/")
print(f"   train_dataset.xlsx  → {len(df_train):,} filas")
print(f"   val_dataset.xlsx    → {len(df_val):,} filas")
print(f"   test_dataset.xlsx   → {len(df_test):,} filas")
print(f"   dataset_final_entrenamiento.xlsx → {len(df_final):,} filas")
print("\n--- TRANSFORMACION Y AUMENTACION COMPLETADA ---")
