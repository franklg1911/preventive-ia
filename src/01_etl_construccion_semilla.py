import pandas as pd
import os

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_RAW = os.path.join(BASE_DIR, "data", "raw")
PATH_PROCESSED = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(PATH_PROCESSED, exist_ok=True)


print("--- INICIANDO PROCESO ETL (MODO EXCEL) ---")

try:
    print("Cargando archivos Excel...")

    path_productos = os.path.join(PATH_RAW, "productos", "productos.xlsx")
    path_facturas = os.path.join(PATH_RAW, "tiendas", "factura-cabecera.xlsx")
    path_cot_cab = os.path.join(PATH_RAW, "tiendas", "cotizaciones-cabecera.xlsx")
    path_cot_det = os.path.join(PATH_RAW, "tiendas", "cotizaciones-detalle.xlsx")

    df_productos = pd.read_excel(path_productos)
    df_facturas = pd.read_excel(path_facturas)
    df_cot_cab = pd.read_excel(path_cot_cab)
    df_cot_det = pd.read_excel(path_cot_det)

    # Normalizamos las columnas
    df_productos.columns = df_productos.columns.str.lower().str.strip()
    df_facturas.columns = df_facturas.columns.str.lower().str.strip()
    df_cot_cab.columns = df_cot_cab.columns.str.lower().str.strip()
    df_cot_det.columns = df_cot_det.columns.str.lower().str.strip()

    # Desduplicación preventiva
    len_fac_antes = len(df_facturas)
    df_facturas = df_facturas.drop_duplicates()
    df_cot_cab = df_cot_cab.drop_duplicates()
    df_cot_det = df_cot_det.drop_duplicates()

    if len(df_facturas) < len_fac_antes:
        print(
            f"Se eliminaron {len_fac_antes - len(df_facturas)} duplicados en Facturas."
        )

    print("-> Archivos cargados y desduplicados exitosamente.")


except Exception as e:
    print(f"Error al cargar los archivos:{e}")
    exit()


# =============================================================
# 2. LIMPIEZA Y PREPARACION DE TIPOS
# =============================================================

print("\n--- LIMPIANDO CLAVES DE CRUCE ---")

# Limpieza - RUC
if "ruc" in df_facturas.columns:
    df_facturas["ruc_str"] = (
        df_facturas["ruc"].astype(str).str.replace(".0", "", regex=False).str.strip()
    )

if "ruc" in df_cot_cab.columns:
    df_cot_cab["ruc_str"] = (
        df_cot_cab["ruc"].astype(str).str.replace(".0", "", regex=False).str.strip()
    )

# Limpieza - MONTOS
col_monto_factura = "monto" if "monto" in df_facturas.columns else "total"
col_monto_cotiza = "total_pagar" if "total_pagar" in df_cot_cab.columns else "total"

df_facturas["monto_limpio"] = pd.to_numeric(
    df_facturas[col_monto_factura], errors="coerce"
)

df_cot_cab["monto_limpio"] = pd.to_numeric(
    df_cot_cab[col_monto_cotiza], errors="coerce"
)


# -------------------------------------------------------------
# [C3] LIMPIEZA Y VALIDACION DE FECHA
# Viene como fecha real desde Excel, pd.read_excel ya la
# convierte a datetime. Forzamos por si alguna celda vino
# como texto, y detectamos nulos y rangos fuera de lo esperado.
# -------------------------------------------------------------

print("\n--- VALIDANDO COLUMNA FECHA ---")

df_facturas["fecha"] = pd.to_datetime(df_facturas["fecha"], errors="coerce")

fechas_nulas = df_facturas["fecha"].isna().sum()

if fechas_nulas > 0:
    print(f"⚠️  {fechas_nulas} filas con fecha inválida — se eliminarán.")
    df_facturas = df_facturas.dropna(subset=["fecha"])

print(
    f"   Rango de fechas: {df_facturas['fecha'].min().date()} → "
    f"{df_facturas['fecha'].max().date()}"
)

print(f"   Total facturas con fecha válida: {len(df_facturas)}")


# =============================================================
# 3. CRUCE — RECONSTRUCCION DE VENTAS
# =============================================================

print("\n--- RECONSTRUYENDO VENTAS (CRUCE FACTURAS - COTIZACIONES) ---")

df_merge_1 = pd.merge(
    df_facturas,
    df_cot_cab[["idcotizaciones", "ruc_str", "monto_limpio"]],
    on=["ruc_str", "monto_limpio"],
    how="inner",
)

print(f"   Cruce 1 exitoso: {df_merge_1.shape[0]} ventas vinculadas.")

# -------------------------------------------------------
# [C1] VALIDACION DE DUPLICADOS EN EL CRUCE
# Si un cliente (ruc) tiene dos cotizaciones con el mismo
# monto, el join multiplica filas silenciosamente.
# Detectamos y reportamos antes de continuar.
# -------------------------------------------------------

print("\n--- [C1] VERIFICANDO INTEGRIDAD DEL CRUCE ---")

duplicados_cruce = (
    df_merge_1.groupby(["ruc_str", "monto_limpio"])
    .size()
    .reset_index(name="ocurrencias")
)

ambiguos = duplicados_cruce[duplicados_cruce["ocurrencias"] > 1]

if len(ambiguos) > 0:
    print(
        f"Se detectaron {len(ambiguos)} combinaciones RUC+Monto con mas de una coincidencia."
    )
    print("   Esto puede generar filas duplicadas en el resultado.")
    print("   Muestra de casos ambiguos:")
    print(ambiguos.head(5).to_string(index=False))

    # Eliminamos duplicados del cruce conservando solo el primer match
    # para no inflar artificialmente el dataset

    df_merge_1 = df_merge_1.drop_duplicates(subset=["ruc_str", "monto_limpio"])

    print(f"Duplicados del cruce eliminados. Filas restantes: {df_merge_1.shape[0]}")

else:
    print("Cruce limpio — sin combinaciones ambiguas.")

# =============================================================
# 4. UNIR CON DETALLE DE COTIZACIÓN
# =============================================================

df_final = pd.merge(df_merge_1, df_cot_det, on="idcotizaciones", how="inner")

print(f"\nCruce con detalle exitoso: {df_final.shape[0]} lineas de venta.")

# =============================================================
# 5. ENRIQUECER CON MAESTRO DE PRODUCTOS
# =============================================================

print("\n--- ENRIQUECIENDO CON MAESTRO DE PRODUCTOS ---")

cols_producto = ["id", "codigo", "descripcion", "categoria", "precio"]

df_final = pd.merge(
    df_final,
    df_productos[cols_producto],
    left_on="idproducto",
    right_on="id",
    how="left",
    suffixes=("", "_maestro"),
)

# ---------------------------------------------------------
# [C2] VERIFICACION DE PRODUCTOS SIN MATCH
# El left join conserva todas las líneas de venta aunque
# el idproducto no exista en el maestro. Esas filas quedan
# con descripcion = NaN y no sirven para entrenar el modelo.
# ----------------------------------------------------------
print("\n--- [C2] VERIFICANDO PRODUCTOS SIN MATCH ---")

sin_producto = df_final["descripcion"].isna().sum()

if sin_producto > 0:
    print(
        f"{sin_producto} lineas sin producto en maestro "
        f"({sin_producto/len(df_final)*100:.1f}% del total)."
    )
    print("   IDs de producto sin match:")
    ids_faltantes = (
        df_final[df_final["descripcion"].isna()]["idproducto"].value_counts().head(10)
    )
    print(ids_faltantes.to_string())
    print("-> Estas filas se eliminan para no contaminar el modelo.")
    df_final = df_final.dropna(subset=["descripcion"])
else:
    print("Todos los productos tienen match en el maestro.")

# =============================================================
# 6. RESUMEN Y GUARDADO
# =============================================================

print("\n--- RESUMEN FINAL ---")
print(df_final[["fecha", "descripcion", "cantidad", "precio_maestro"]].head(5))
print(f"\nTotal filas finales: {df_final.shape[0]}")
print(f"Productos unicos   : {df_final['descripcion'].nunique()}")
print(
    f"Rango de fechas    : {df_final['fecha'].min().date()} → "
    f"{df_final['fecha'].max().date()}"
)

output_file = os.path.join(PATH_PROCESSED, "dataset_reconstruido.xlsx")
df_final.to_excel(output_file, index=False)
print(f"\nArchivo guardado en: {output_file}")
print("--- ETL COMPLETADO ---")
