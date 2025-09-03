import streamlit as st
import pandas as pd
from PIL import Image
import os
import unicodedata
from io import BytesIO
from deepface import DeepFace

# ==============================
# RUTAS DEL EXCEL E IMÁGENES
# ==============================
RUTA_EXCEL = os.path.join(os.getcwd(), "informacion.xlsx")
RUTA_IMAGENES = os.path.join(os.getcwd(), "IMAGENES")

# ==============================
# FUNCIONES AUXILIARES
# ==============================
def normalizar_texto(texto: str) -> str:
    """Convierte texto a minúsculas sin tildes ni caracteres especiales."""
    if not isinstance(texto, str):
        return ""
    texto = texto.strip().lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return texto

@st.cache_data
def cargar_datos():
    """Carga el Excel y normaliza nombres de columnas a MAYÚSCULAS."""
    try:
        df = pd.read_excel(RUTA_EXCEL)
        df.columns = [c.upper() for c in df.columns]  # ✅ columnas en mayúsculas
        return df
    except Exception as e:
        st.error(f"❌ Error cargando Excel: {e}")
        return pd.DataFrame()

def mostrar_resultados(resultados, df):
    """Muestra los resultados con imagen y datos."""
    if resultados.empty:
        st.warning("⚠️ No se encontraron coincidencias.")
    else:
        for _, row in resultados.iterrows():
            ruta_img = os.path.join(RUTA_IMAGENES, str(row["IMAGEN"]))
            if os.path.exists(ruta_img):
                st.image(ruta_img, width=150)
            else:
                st.warning(f"⚠️ Imagen no encontrada: {ruta_img}")
            st.write(f"**ID:** {row['ID']}")
            st.write(f"**Nombre:** {row['NOMBRE']}")
            st.write(f"**Tipo ID:** {row['TIPO DE ID']}")
            st.write(f"**Municipio:** {row['MUNICIPIO']}")
            st.write(f"**NUNC:** {row['NUNC']}")
            st.markdown("---")

# ==============================
# INTERFAZ STREAMLIT
# ==============================
st.title("🔎 Búsqueda de personas en Excel con reconocimiento facial")

df = cargar_datos()

if not df.empty:
    opcion = st.radio("Elige cómo buscar:", ["Por nombre", "Por ID", "Por imagen"])

    # ------------------------------
    # BÚSQUEDA POR NOMBRE (parcial)
    # ------------------------------
    if opcion == "Por nombre":
        nombre_buscar = st.text_input("Escribe el nombre (o parte del nombre) a buscar:")

        if st.button("Buscar"):
            if nombre_buscar.strip() == "":
                st.warning("⚠️ Por favor escribe un nombre.")
            else:
                df["NOMBRE_NORM"] = df["NOMBRE"].apply(normalizar_texto)
                nombre_norm = normalizar_texto(nombre_buscar)

                # ✅ búsqueda parcial
                resultados = df[df["NOMBRE_NORM"].str.contains(nombre_norm, na=False)]
                mostrar_resultados(resultados, df)

    # ------------------------------
    # BÚSQUEDA POR ID
    # ------------------------------
    elif opcion == "Por ID":
        id_buscar = st.text_input("Escribe el ID a buscar:")

        if st.button("Buscar"):
            if id_buscar.strip() == "":
                st.warning("⚠️ Por favor escribe un ID.")
            else:
                resultados = df[df["ID"].astype(str).str.contains(id_buscar.strip())]
                mostrar_resultados(resultados, df)

    # ------------------------------
    # BÚSQUEDA POR IMAGEN
    # ------------------------------
    elif opcion == "Por imagen":
        archivo = st.file_uploader("Sube una imagen para buscar coincidencias", type=["jpg", "jpeg", "png"])

        if archivo is not None:
            img_bytes = archivo.read()
            img = Image.open(BytesIO(img_bytes))
            st.image(img, caption="📷 Imagen cargada", width=200)

            if st.button("Buscar"):
                try:
                    resultados = []
                    for _, row in df.iterrows():
                        ruta_img = os.path.join(RUTA_IMAGENES, str(row["IMAGEN"]))
                        if os.path.exists(ruta_img):
                            try:
                                # Comparación con DeepFace
                                resp = DeepFace.verify(img_bytes, ruta_img, enforce_detection=False)
                                if resp["verified"]:
                                    resultados.append(row)
                            except Exception as e:
                                st.error(f"Error comparando con {ruta_img}: {e}")

                    if resultados:
                        resultados_df = pd.DataFrame(resultados)
                        mostrar_resultados(resultados_df, df)
                    else:
                        st.warning("⚠️ No se encontraron coincidencias.")

                except Exception as e:
                    st.error(f"❌ Error en la búsqueda por imagen: {e}")














