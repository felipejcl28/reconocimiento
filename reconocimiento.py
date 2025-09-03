import streamlit as st
import pandas as pd
import os
import unicodedata
from PIL import Image
from deepface import DeepFace

# 📂 Rutas de archivos
RUTA_EXCEL = os.path.join(os.getcwd(), "informacion.xlsx")
RUTA_IMAGENES = os.path.join(os.getcwd(), "IMAGENES")

# 🔤 Normalizar texto (para búsquedas sin acentos ni mayúsculas)
def normalizar_texto(texto: str) -> str:
    if not isinstance(texto, str):
        return ""
    texto = texto.strip().lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return texto

@st.cache_data
def cargar_datos():
    df = pd.read_excel(RUTA_EXCEL, dtype=str)
    df = df.fillna("")
    return df

df = cargar_datos()

st.title("🔎 Búsqueda de Personas")

# 📌 Opción de búsqueda
opcion = st.radio("Elige cómo deseas buscar:", ["POR IDENTIFICACION", "POR NOMBRE", "POR FOTO"])

# =====================
# 🔍 Búsqueda por ID
# =====================
if opcion == "POR IDENTIFICACION":
    id_buscar = st.text_input("Escribe el ID a buscar:")
    if st.button("Buscar por ID"):
        resultados = df[df["ID"] == id_buscar.strip()]
        if not resultados.empty:
            for _, row in resultados.iterrows():
                img_path = os.path.join(RUTA_IMAGENES, row["IMAGEN"])
                if os.path.exists(img_path):
                    st.image(Image.open(img_path), width=150)
                st.markdown(f"**ID:** {row['ID']}")
                st.markdown(f"**Nombre:** {row['NOMBRE']}")
                st.markdown(f"**Tipo ID:** {row['TIPO DE ID']}")
                st.markdown(f"**NUNC:** {row['NUNC']}")
        else:
            st.warning("⚠️ No se encontró ninguna persona con ese ID.")

# =====================
# 🔍 Búsqueda por Nombre
# =====================
elif opcion == "POR NOMBRE":
    nombre_buscar = st.text_input("Escribe el nombre (o parte del nombre) a buscar:")
    if st.button("Buscar por Nombre"):
        nombre_normalizado = normalizar_texto(nombre_buscar)
        resultados = df[df["NOMBRE"].apply(lambda x: nombre_normalizado in normalizar_texto(x))]
        if not resultados.empty:
            for _, row in resultados.iterrows():
                img_path = os.path.join(RUTA_IMAGENES, row["IMAGEN"])
                if os.path.exists(img_path):
                    st.image(Image.open(img_path), width=150)
                st.markdown(f"**ID:** {row['ID']}")
                st.markdown(f"**Nombre:** {row['NOMBRE']}")
                st.markdown(f"**Tipo ID:** {row['TIPO DE ID']}")
                st.markdown(f"**NUNC:** {row['NUNC']}")
        else:
            st.warning("⚠️ No se encontró ninguna persona con ese nombre.")

# =====================
# 🖼️ Búsqueda por Imagen
# =====================
elif opcion == "POR FOTO":
    archivo_imagen = st.file_uploader("Sube una imagen para buscar coincidencias:", type=["jpg", "jpeg", "png"])
    if archivo_imagen is not None:
        with open("temp.jpg", "wb") as f:
            f.write(archivo_imagen.getbuffer())
        try:
            encontrado = None
            for _, row in df.iterrows():
                img_path = os.path.join(RUTA_IMAGENES, row["IMAGEN"])
                if os.path.exists(img_path):
                    result = DeepFace.verify("temp.jpg", img_path, enforce_detection=False)
                    if result["verified"]:
                        encontrado = row
                        break
            if encontrado is not None:
                st.success("✅ Persona encontrada:")
                img_path = os.path.join(RUTA_IMAGENES, encontrado["IMAGEN"])
                st.image(Image.open(img_path), width=150)
                st.markdown(f"**ID:** {encontrado['ID']}")
                st.markdown(f"**Nombre:** {encontrado['NOMBRE']}")
                st.markdown(f"**Tipo ID:** {encontrado['TIPO DE ID']}")
                st.markdown(f"**NUNC:** {encontrado['NUNC']}")
            else:
                st.warning("⚠️ No se encontró ninguna coincidencia.")
        except Exception as e:
            st.error(f"❌ Error en el reconocimiento facial: {e}")




















