import streamlit as st
import pandas as pd
import os
from deepface import DeepFace
from PIL import Image
import numpy as np
import hashlib

# ======================
# CONFIGURACIÓN
# ======================
RUTA_EXCEL = os.path.join(os.getcwd(), "informacion.xlsx")
RUTA_IMAGENES = os.path.join(os.getcwd(), "IMAGENES")

# ======================
# FUNCIONES
# ======================

def normalizar_texto(texto: str) -> str:
    """Normaliza texto para comparaciones consistentes"""
    if not isinstance(texto, str):
        return ""
    return texto.strip().lower()

def calcular_hash_imagen(imagen_path):
    """Devuelve hash SHA256 de una imagen"""
    with open(imagen_path, "rb") as f:
        bytes_img = f.read()
        return hashlib.sha256(bytes_img).hexdigest()

def cargar_excel():
    if os.path.exists(RUTA_EXCEL):
        df = pd.read_excel(RUTA_EXCEL)
        # Normalizamos nombres de columnas
        df.columns = [c.strip().upper() for c in df.columns]
        # Columna auxiliar para nombre de archivo
        if "IMAGEN" in df.columns:
            df["IMAGEN_NORM"] = df["IMAGEN"].apply(lambda x: str(x).strip())
        return df
    return pd.DataFrame()

def buscar_por_hash(uploaded_file, df):
    """Verifica si la imagen subida existe idéntica en IMAGENES"""
    uploaded_hash = hashlib.sha256(uploaded_file.read()).hexdigest()
    uploaded_file.seek(0)  # Resetear puntero
    for _, row in df.iterrows():
        img_path = os.path.join(RUTA_IMAGENES, row["IMAGEN_NORM"])
        if os.path.exists(img_path):
            if uploaded_hash == calcular_hash_imagen(img_path):
                return row
    return None

def buscar_por_embeddings(uploaded_file, df, modelo, detector, umbral):
    """Usa DeepFace para comparar embeddings"""
    results = []
    for _, row in df.iterrows():
        img_db_path = os.path.join(RUTA_IMAGENES, row["IMAGEN_NORM"])
        if os.path.exists(img_db_path):
            try:
                verify = DeepFace.verify(
                    uploaded_file,
                    img_db_path,
                    model_name=modelo,
                    detector_backend=detector,
                    enforce_detection=False
                )
                distance = verify["distance"]
                if distance <= umbral:
                    results.append((row, distance))
            except Exception as e:
                print(f"Error con {img_db_path}: {e}")
    return results

# ======================
# INTERFAZ STREAMLIT
# ======================

st.set_page_config(page_title="Reconocimiento Facial", layout="wide")
st.title("🔍 Búsqueda de Personas")

# Opciones en sidebar
modelo = st.sidebar.selectbox("Modelo", ["ArcFace", "Facenet", "VGG-Face"])
detector = st.sidebar.selectbox("Detector", ["mtcnn", "opencv", "ssd", "retinaface"])
enforce_detection = st.sidebar.checkbox("Enforce detection (estricto)", value=False)
umbral = st.sidebar.slider("Umbral distancia (<=)", 0.0, 1.0, 0.4, 0.01)
usar_hash = st.sidebar.checkbox("Usar comparación por hash (idéntico contenido)", value=True)

# Cargar base
df = cargar_excel()

if df.empty:
    st.error("❌ No se encontró el archivo informacion.xlsx")
else:
    uploaded_file = st.file_uploader("📤 Subir imagen para buscar", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.image(uploaded_file, caption="Imagen cargada", width=250)

        # Primero: buscar por hash
        if usar_hash:
            match_hash = buscar_por_hash(uploaded_file, df)
            if match_hash is not None:
                st.success(f"✅ Coincidencia exacta encontrada (hash) → {match_hash['NOMBRE']}")
                st.image(os.path.join(RUTA_IMAGENES, match_hash["IMAGEN_NORM"]), width=250)
            else:
                st.info("ℹ️ No hay coincidencia exacta por hash.")

        # Segundo: búsqueda por embeddings
        results = buscar_por_embeddings(uploaded_file, df, modelo, detector, umbral)

        if results:
            st.subheader("👥 Posibles coincidencias")
            for row, distance in sorted(results, key=lambda x: x[1]):
                if distance == 0.0:
                    st.success(f"🎯 MATCH EXACTO → {row['NOMBRE']} (distance={distance:.4f})")
                else:
                    st.warning(f"{row['NOMBRE']} (distance={distance:.4f})")

                st.image(os.path.join(RUTA_IMAGENES, row["IMAGEN_NORM"]), width=200)
        else:
            st.error("❌ No se encontró coincidencia con el umbral definido.")







































