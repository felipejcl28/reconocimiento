import streamlit as st
import pandas as pd
import os
import unicodedata
from PIL import Image
from deepface import DeepFace
from io import BytesIO

# 📂 Rutas de archivos
RUTA_EXCEL = os.path.join(os.getcwd(), "informacion.xlsx")
RUTA_IMAGENES = os.path.join(os.getcwd(), "IMAGENES")

# 🔤 Normalizar texto
def normalizar_texto(texto: str) -> str:
    if not isinstance(texto, str):
        return ""
    texto = texto.strip().lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return texto

# 📑 Cargar datos
@st.cache_data
def cargar_datos():
    df = pd.read_excel(RUTA_EXCEL, dtype=str)
    df = df.fillna("")
    return df

df = cargar_datos()

st.title("🔎 Búsqueda de Personas")

# 📌 Opción de búsqueda
opcion = st.radio("Elige cómo deseas buscar:", ["Por ID", "Por Nombre", "Por Imagen"])

# Función para generar Excel y botón de descarga
def exportar_resultados(resultados):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        resultados.to_excel(writer, index=False, sheet_name="Resultados")
    st.download_button(
        label="📥 Descargar resultados en Excel",
        data=output.getvalue(),
        file_name="resultados_busqueda.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# =====================
# 🔍 Búsqueda por ID
# =====================
if opcion == "Por ID":
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
            exportar_resultados(resultados)
        else:
            st.warning("⚠️ No se encontró ninguna persona con ese ID.")

# =====================
# 🔍 Búsqueda por Nombre
# =====================
elif opcion == "Por Nombre":
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
            exportar_resultados(resultados)
        else:
            st.warning("⚠️ No se encontró ninguna persona con ese nombre.")

# =====================
# 🖼️ Búsqueda por Imagen
# =====================
elif opcion == "Por Imagen":
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

                # Convertir a DataFrame y exportar
                resultados = pd.DataFrame([encontrado])
                exportar_resultados(resultados)
            else:
                st.warning("⚠️ No se encontró ninguna coincidencia.")
        except Exception as e:
            st.error(f"❌ Error en el reconocimiento facial: {e}")























