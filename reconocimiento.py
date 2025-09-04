# app_debug_streamlit.py
import streamlit as st
import pandas as pd
import os
import hashlib
import traceback
from deepface import DeepFace
from io import BytesIO

# ---------- RUTAS (relativas a cwd) ----------
RUTA_EXCEL = os.path.join(os.getcwd(), "informacion.xlsx")
RUTA_IMAGENES = os.path.join(os.getcwd(), "IMAGENES")
LOG_PATH = os.path.join(os.getcwd(), "app_error.log")

# ---------- UTIL ----------
def write_log(text: str):
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    except Exception:
        pass

def hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def safe_read_excel(path):
    try:
        df = pd.read_excel(path, dtype=str)
        return df
    except Exception as e:
        raise RuntimeError(f"Error leyendo {path}: {e}")

def find_column_name(df: pd.DataFrame, candidates):
    # devuelve la primera columna del df que coincida (case-insensitive) con algún candidato
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

# ---------- CARGAR BASE (con chequeos) ----------
def cargar_base():
    if not os.path.exists(RUTA_EXCEL):
        st.error(f"No se encontró el archivo Excel en: {RUTA_EXCEL}")
        return pd.DataFrame()

    try:
        df = safe_read_excel(RUTA_EXCEL)
    except Exception as e:
        st.error(str(e))
        write_log("TRACEBACK reading excel:\n" + traceback.format_exc())
        return pd.DataFrame()

    # normalizar nombres de columnas a strings sin espacios extra
    df.columns = [str(c).strip() for c in df.columns]

    # identificar columnas importantes (aceptar variantes)
    col_id = find_column_name(df, ["ID", "Id", "id"])
    col_nombre = find_column_name(df, ["NOMBRE", "Nombre", "name", "NOMBRE_COMPLETO"])
    col_imagen = find_column_name(df, ["IMAGEN", "Imagen", "imagen", "FILE", "FILENAME"])

    # Informar si faltan columnas (no lanzar KeyError más adelante)
    missing = []
    if col_id is None: missing.append("ID")
    if col_nombre is None: missing.append("NOMBRE")
    if col_imagen is None: missing.append("IMAGEN")

    if missing:
        st.warning(f"Columnas faltantes o no detectadas en Excel: {', '.join(missing)}. Se intentará continuar, pero pueden faltar datos.")
    # Añadir columnas estandarizadas al df (si faltan, crear vacías)
    df = df.copy()
    df["ID"] = df[col_id] if col_id is not None else ""
    df["NOMBRE"] = df[col_nombre] if col_nombre is not None else ""
    df["IMAGEN"] = df[col_imagen] if col_imagen is not None else ""

    # normalizar nombre de archivo para matching (minusculas y basename)
    df["IMAGEN_NORM"] = df["IMAGEN"].fillna("").apply(lambda p: os.path.basename(str(p)).lower())

    return df

# ---------- BÚSQUEDA ----------
def buscar_por_imagen(img_path, base_df, model_name="ArcFace", detector_backend="mtcnn", enforce_detection=False, umbral=0.68, use_hash=False):
    """
    Retorna DataFrame con columnas: ID, NOMBRE, IMAGEN, distance, match_type
    """
    results = []

    # 1) verificar coincidencia por nombre EXACTA
    basename_input = os.path.basename(img_path)
    basename_input_lower = basename_input.lower()
    matches_by_name = base_df[base_df["IMAGEN_NORM"] == basename_input_lower]
    if not matches_by_name.empty:
        for _, r in matches_by_name.iterrows():
            results.append({
                "ID": r.get("ID", ""),
                "NOMBRE": r.get("NOMBRE", ""),
                "IMAGEN": r.get("IMAGEN", ""),
                "distance": 0.0,
                "match_type": "match_nombre_archivo"
            })
        # devolvemos inmediatamente, es match perfecto
        return pd.DataFrame(results).sort_values("distance")

    # 2) (opcional) comparar por hash si está activado
    hash_input = None
    if use_hash:
        try:
            hash_input = hash_file(img_path)
        except Exception as e:
            write_log("Error calculando hash input: " + str(e))
            hash_input = None

    if use_hash and hash_input:
        for _, r in base_df.iterrows():
            img_db = os.path.join(RUTA_IMAGENES, str(r["IMAGEN"]))
            if not os.path.exists(img_db):
                continue
            try:
                if hash_file(img_db) == hash_input:
                    results.append({
                        "ID": r.get("ID", ""),
                        "NOMBRE": r.get("NOMBRE", ""),
                        "IMAGEN": r.get("IMAGEN", ""),
                        "distance": 0.0,
                        "match_type": "match_hash"
                    })
            except Exception as e:
                write_log(f"Error calculando hash para {img_db}: {e}")

        if results:
            return pd.DataFrame(results).sort_values("distance")

    # 3) Intentar DeepFace.find (rápido) - con comprobaciones seguras
    try:
        found = DeepFace.find(
            img_path=img_path,
            db_path=RUTA_IMAGENES,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection
        )
    except Exception as e:
        # guardar traceback y volver con mensaje
        write_log("TRACEBACK DeepFace.find:\n" + traceback.format_exc())
        raise

    # 'found' suele ser una lista (una entrada por modelo), comprobarlo correctamente
    if isinstance(found, list) and len(found) > 0:
        df_found = found[0]
        if isinstance(df_found, pd.DataFrame) and not df_found.empty:
            # ordenar por distance y recoger los top 10
            df_found_sorted = df_found.sort_values("distance").reset_index(drop=True)
            # convertir identity (ruta) a basename y minusculas
            df_found_sorted["identity_basename"] = df_found_sorted["identity"].apply(lambda x: os.path.basename(str(x)).lower())
            # filtrar por umbral (si hay umbral) pero si ninguno pasa, devolver al menos el mejor
            df_filtered = df_found_sorted[df_found_sorted["distance"] <= umbral]
            if df_filtered.empty:
                topk = df_found_sorted.head(1)
            else:
                topk = df_filtered.head(3)
            # mapear con base_df para traer info
            for _, r in topk.iterrows():
                ident = r["identity_basename"]
                distancia = float(r["distance"])
                matched_row = base_df[base_df["IMAGEN_NORM"] == ident]
                if not matched_row.empty:
                    # si hay varias coincidencias con mismo archivo, devolvemos todas
                    for _, mr in matched_row.iterrows():
                        results.append({
                            "ID": mr.get("ID", ""),
                            "NOMBRE": mr.get("NOMBRE", ""),
                            "IMAGEN": mr.get("IMAGEN", ""),
                            "distance": distancia,
                            "match_type": "deepface"
                        })
                else:
                    # La imagen encontrada en IMAGENES no está en el Excel; aun así la devolvemos con ruta
                    results.append({
                        "ID": "",
                        "NOMBRE": "",
                        "IMAGEN": ident,
                        "distance": distancia,
                        "match_type": "deepface_not_in_excel"
                    })
    else:
        # no hubo resultados
        pass

    return pd.DataFrame(results).sort_values("distance") if results else pd.DataFrame()

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Búsqueda debug", layout="wide")
st.title("🔍 Búsqueda de Personas — Debug")

# panel lateral settings
st.sidebar.header("Opciones")
model_name = st.sidebar.selectbox("Modelo", ["ArcFace", "Facenet512", "VGG-Face"])
detector_backend = st.sidebar.selectbox("Detector", ["mtcnn", "opencv", "retinaface", "mediapipe"])
enforce_detection = st.sidebar.checkbox("Enforce detection (estricto)", value=False)
umbral = st.sidebar.slider("Umbral distancia (<=)", 0.0, 1.0, 0.68, 0.01)
use_hash = st.sidebar.checkbox("Usar comparación por hash (identico contenido)", value=True)
show_debug = st.sidebar.checkbox("Mostrar debug extra", value=True)

# Cargar base y mostrar info debug sobre columnas/archivos
base = cargar_base()

st.markdown("**Estado de archivos y base**")
col1, col2 = st.columns(2)
with col1:
    st.write(f"- Excel en: `{RUTA_EXCEL}` → {'EXISTE' if os.path.exists(RUTA_EXCEL) else 'NO ENCONTRADO'}")
    st.write(f"- IMAGENES en: `{RUTA_IMAGENES}` → {'EXISTE' if os.path.exists(RUTA_IMAGENES) else 'NO ENCONTRADO'}")
    st.write(f"- Filas en Excel: {len(base)}")
with col2:
    st.write("Columnas del Excel (original):")
    st.write(base.columns.tolist() if not base.empty else "Excel vacío o no cargado")

if show_debug:
    if not base.empty:
        st.write("Primeras filas (normalized):")
        st.dataframe(base.head(5))
    # listar primeros 20 archivos en IMAGENES
    if os.path.exists(RUTA_IMAGENES):
        files = sorted(os.listdir(RUTA_IMAGENES))[:50]
        st.write(f"Archivos en IMAGENES (mostrando hasta 50): {len(files)}")
        st.write(files)
    else:
        st.info("Crea la carpeta IMAGENES y coloca los archivos ahí.")

# Uploader
st.markdown("---")
st.header("Búsqueda por imagen")
uploaded = st.file_uploader("Sube una imagen (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    temp_dir = os.path.join(os.getcwd(), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, uploaded.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.image(temp_path, width=240, caption="Imagen subida")

    if st.button("Buscar"):
        try:
            # Run search
            results_df = buscar_por_imagen(
                img_path=temp_path,
                base_df=base,
                model_name=model_name,
                detector_backend=detector_backend,
                enforce_detection=enforce_detection,
                umbral=umbral,
                use_hash=use_hash
            )

            if results_df.empty:
                st.warning("⚠️ No se encontraron coincidencias (ni por nombre, ni por hash, ni por DeepFace con el umbral seleccionado).")
                # mostrar top1 bruto si existe (ejecutar DeepFace.find de nuevo sin filtrar para mostrar el mejor)
                try:
                    fallback = DeepFace.find(
                        img_path=temp_path,
                        db_path=RUTA_IMAGENES,
                        model_name=model_name,
                        detector_backend=detector_backend,
                        enforce_detection=False
                    )
                    if isinstance(fallback, list) and len(fallback) > 0 and isinstance(fallback[0], pd.DataFrame) and not fallback[0].empty:
                        fdf = fallback[0].sort_values("distance").head(5)
                        fdf["identity_basename"] = fdf["identity"].apply(lambda x: os.path.basename(str(x)).lower())
                        st.info("Top 5 encontrados (sin aplicar umbral):")
                        st.dataframe(fdf[["identity_basename", "distance"]])
                except Exception as e:
                    write_log("TRACEBACK fallback find:\n" + traceback.format_exc())

            else:
                st.success("✅ Resultados ordenados por distancia:")
                st.dataframe(results_df[["ID", "NOMBRE", "IMAGEN", "distance", "match_type"]].reset_index(drop=True))
                # mostrar imágenes encontradas
                for _, r in results_df.iterrows():
                    img_db = os.path.join(RUTA_IMAGENES, str(r["IMAGEN"]))
                    st.markdown(f"**{r.get('NOMBRE','(sin nombre)')}** — distancia: {r.get('distance'):.4f} — tipo: {r.get('match_type')}")
                    if os.path.exists(img_db):
                        st.image(img_db, width=220)
                    else:
                        st.write(f"Imagen no encontrada en IMAGENES: {r.get('IMAGEN')}")

        except Exception as e:
            st.error("Ha ocurrido un error durante la búsqueda. Revisa el log (expand) o descarga el archivo app_error.log.")
            trace = traceback.format_exc()
            write_log("TRACEBACK main search:\n" + trace)
            with st.expander("Mostrar detalle del error (traceback)"):
                st.code(trace)

# botón para ver log
if st.button("Mostrar app_error.log"):
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            st.text(f.read())
    else:
        st.info("No existe app_error.log todavía.")






































