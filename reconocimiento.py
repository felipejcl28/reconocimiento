import streamlit as st
import pandas as pd
from PIL import Image
import os
import unicodedata
from io import BytesIO
from deepface import DeepFace

# ============================
# Configuración de rutas
# ============================
RUTA_EXCEL = os.path.join(os.getcwd(), "informacion.xlsx")
RUTA_IMAGENES = os.path.join(os.getcwd(), "IMAGENES")

# ============================
# Función para normalizar texto
# ============================
def normalizar_texto(texto: str) -> str:
    if not isinstance(texto, str):
        return ""
    texto = texto.strip().lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return texto

# ============================
# Cargar datos del Excel
# ============================
@st.cache_data
def cargar_datos():
    try:
        return pd.read_excel(RUTA_EXCEL)
    except Exception as e:
        st.error(f"❌ Error cargando Excel: {e}")
        return pd.DataFrame()

df = cargar_datos()

# ============================
# Interfaz de usuario
# ============================
st.title("🔎 Reconocimiento y Búsqueda de Personas")

opcion = st.radio("Elige cómo buscar:", ["Por nombre", "Por imagen"])

# ============================
# Búsqueda por nombre
# ============================
if opcion == "Por nombre":
    nombre = st.text_input("Escribe el nombre a buscar:")
    if nombre:
        nombre_norm = normalizar_texto(nombre)
        df["nombre_norm"] = df["Nombre"].apply(normalizar_texto)

        resultados = df[df["nombre_norm"].str.contains(nombre_norm, na=False)]

        if not resultados.empty:
            st.success("✅ Coincidencias encontradas:")
            st.dataframe(resultados)

            for _, row in resultados.iterrows():
                ruta_img = os.path.join(RUTA_IMAGENES, row["Imagen"])
                if os.path.exists(ruta_img):
                    st.image(ruta_img, caption=row["Nombre"], width=200)
        else:
            st.warning("⚠️ No se encontraron coincidencias.")

# ============================
# Búsqueda por imagen
# ============================
if opcion == "Por imagen":
    archivo = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

    if archivo is not None:
        imagen = Image.open(archivo)
        st.image(imagen, caption="Imagen cargada", width=250)

        # Guardar temporalmente
        img_bytes = BytesIO()
        imagen.save(img_bytes, format="PNG")
        img_path = "temp.png"
        with open(img_path, "wb") as f:
            f.write(img_bytes.getvalue())

        try:
            resultados = DeepFace.find(
                img_path=img_path,
                db_path=RUTA_IMAGENES,
                model_name="Facenet",  # ✅ más liviano que VGGFace
                enforce_detection=False
            )

            if len(resultados) > 0 and not resultados[0].empty:
                st.success("✅ Coincidencias encontradas:")

                for i, row in resultados[0].iterrows():
                    ruta_img = row["identity"]
                    st.image(ruta_img, caption=os.path.basename(ruta_img), width=200)

                    # Buscar información en el Excel
                    persona = os.path.basename(ruta_img)
                    info = df[df["Imagen"] == persona]
                    if not info.empty:
                        st.dataframe(info)

            else:
                st.warning("⚠️ No se encontraron coincidencias en la base de imágenes.")

        except Exception as e:
            st.error(f"❌ Error en el reconocimiento: {e}")













