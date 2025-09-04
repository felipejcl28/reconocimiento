import streamlit as st
import pandas as pd
from deepface import DeepFace
import os
import hashlib

# ======================
# CONFIGURACIÓN
# ======================
RUTA_EXCEL = os.path.join(os.getcwd(), "informacion.xlsx")
RUTA_IMAGENES = os.path.join(os.getcwd(), "IMAGENES")

# ======================
# FUNCIONES
# ======================
def cargar_base():
    if os.path.exists(RUTA_EXCEL):
        return pd.read_excel(RUTA_EXCEL)
    return pd.DataFrame()

def hash_file(path):
    """Devuelve hash SHA256 de un archivo"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def buscar_por_imagen(img_path, base, modelo="ArcFace", detector="mtcnn", umbral=0.68):
    resultados = []

    # Hash de la imagen que subió el usuario
    hash_input = hash_file(img_path)

    for _, fila in base.iterrows():
        img_db = os.path.join(RUTA_IMAGENES, str(fila["imagen"]))

        if not os.path.exists(img_db):
            continue

        # 🔑 1. Comparación directa por nombre de archivo
        if os.path.basename(img_db) == os.path.basename(img_path):
            resultados.append({
                "ID": fila.get("ID", ""),
                "Nombre": fila.get("Nombre", ""),
                "Imagen": fila.get("imagen", ""),
                "distance": 0.0,
                "match": True,
                "tipo": "Match exacto (nombre de archivo)"
            })
            continue

        # 🔑 2. Comparación por hash (contenido idéntico)
        if hash_file(img_db) == hash_input:
            resultados.append({
                "ID": fila.get("ID", ""),
                "Nombre": fila.get("Nombre", ""),
                "Imagen": fila.get("imagen", ""),
                "distance": 0.0,
                "match": True,
                "tipo": "Match exacto (hash de archivo)"
            })
            continue

        # 🔑 3. Comparación normal con DeepFace
        try:
            verif = DeepFace.verify(
                img1_path=img_path,
                img2_path=img_db,
                model_name=modelo,
                detector_backend=detector,
                enforce_detection=False,   # permite imágenes pequeñas
            )

            distancia = verif["distance"]
            resultados.append({
                "ID": fila.get("ID", ""),
                "Nombre": fila.get("Nombre", ""),
                "Imagen": fila.get("imagen", ""),
                "distance": distancia,
                "match": verif["verified"],
                "tipo": "DeepFace"
            })

        except Exception as e:
            print(f"Error comparando {img_db}: {e}")

    if not resultados:
        return pd.DataFrame()

    df = pd.DataFrame(resultados).sort_values("distance", ascending=True)

    # Filtrar por umbral
    df_filtrado = df.query("distance <= @umbral or distance == 0.0")

    if df_filtrado.empty:
        # 🔑 Si no encuentra nada bajo el umbral, devuelve el más cercano
        return df.head(1)
    else:
        return df_filtrado.head(3)

# ======================
# INTERFAZ STREAMLIT
# ======================
st.set_page_config(page_title="Búsqueda de Personas", layout="wide")

st.title("🔍 Búsqueda de Personas")

# Configuración lateral
st.sidebar.header("⚙️ Configuración de búsqueda")

modelo = st.sidebar.selectbox("Modelo de reconocimiento", ["ArcFace", "Facenet", "VGG-Face"])
umbral = st.sidebar.slider("Umbral máximo de distancia (menor es más parecido)", 0.0, 1.0, 0.68, 0.01)
detector = st.sidebar.selectbox("Detector de rostro", ["mtcnn", "opencv", "retinaface", "mediapipe", "dlib"])

# Base de datos
base = cargar_base()

# Elección de búsqueda
opcion = st.radio("Elige cómo buscar:", ["Por nombre", "Por ID", "Por imagen"])

if opcion == "Por imagen":
    archivo = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
    if archivo is not None:
        img_path = os.path.join("temp", archivo.name)
        os.makedirs("temp", exist_ok=True)
        with open(img_path, "wb") as f:
            f.write(archivo.read())

        if st.button("Buscar"):
            resultados = buscar_por_imagen(img_path, base, modelo, detector, umbral)

            if not resultados.empty:
                st.success("✅ Resultados encontrados:")

                for _, fila in resultados.iterrows():
                    img_show = os.path.join(RUTA_IMAGENES, str(fila["Imagen"]))
                    if os.path.exists(img_show):
                        if fila["distance"] == 0.0:
                            st.markdown(f"### 🌟 {fila['Nombre']} (Match exacto - {fila['tipo']})")
                        else:
                            st.markdown(f"**{fila['Nombre']}** - Distancia: `{fila['distance']:.3f}`")

                        st.image(img_show, caption=f"ID: {fila['ID']}", width=250)
                        st.markdown("---")
            else:
                st.warning("⚠️ No se encontraron coincidencias.")





































