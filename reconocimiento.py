import streamlit as st
import pandas as pd
import os
from PIL import Image
from deepface import DeepFace
from rapidfuzz import process

# ========================
# Configuración de rutas
# ========================
RUTA_EXCEL = os.path.join(os.getcwd(), "informacion.xlsx")
RUTA_IMAGENES = os.path.join(os.getcwd(), "IMAGENES")

# ========================
# Funciones auxiliares
# ========================
def cargar_datos():
    if not os.path.exists(RUTA_EXCEL):
        st.error("❌ No se encontró el archivo Excel.")
        return pd.DataFrame()
    df = pd.read_excel(RUTA_EXCEL)
    df["IMAGEN_NORM"] = df["IMAGEN"].apply(lambda x: str(x).strip().lower())
    return df

def buscar_por_nombre(df, nombre):
    nombres = df["NOMBRE"].tolist()
    mejor, score, idx = process.extractOne(nombre, nombres)
    if score > 80:  # umbral de similitud
        return df.iloc[[idx]]
    return pd.DataFrame()

def buscar_por_id(df, id_buscar):
    resultados = df[df["ID"].astype(str) == str(id_buscar)]
    return resultados

def buscar_por_imagen(df, ruta_imagen, modelo="ArcFace", detector="mtcnn", umbral=0.68):
    resultados = []
    for _, fila in df.iterrows():
        ruta_base = os.path.join(RUTA_IMAGENES, fila["IMAGEN"])
        if os.path.exists(ruta_base):
            try:
                verif = DeepFace.verify(
                    img1_path=ruta_imagen,
                    img2_path=ruta_base,
                    model_name=modelo,
                    detector_backend=detector,
                    enforce_detection=False
                )
                if verif["distance"] <= umbral:
                    resultados.append(fila)
            except Exception as e:
                st.warning(f"⚠️ Error con {fila['IMAGEN']}: {e}")
    return pd.DataFrame(resultados)

# ========================
# Interfaz Streamlit
# ========================
st.title("🔍 Búsqueda de Personas")

df = cargar_datos()
if df.empty:
    st.stop()

opcion = st.radio("Elige cómo buscar:", ["Por nombre", "Por ID", "Por imagen"])

if opcion == "Por nombre":
    nombre = st.text_input("Escribe el nombre")
    if st.button("Buscar"):
        resultados = buscar_por_nombre(df, nombre)
        if not resultados.empty:
            st.success("✅ Resultado encontrado:")
            st.dataframe(resultados)
        else:
            st.warning("⚠️ No se encontraron coincidencias.")

elif opcion == "Por ID":
    id_buscar = st.text_input("Escribe el ID")
    if st.button("Buscar"):
        resultados = buscar_por_id(df, id_buscar)
        if not resultados.empty:
            st.success("✅ Resultado encontrado:")
            st.dataframe(resultados)
        else:
            st.warning("⚠️ No se encontraron coincidencias.")

elif opcion == "Por imagen":
    archivo_imagen = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
    if archivo_imagen is not None:
        try:
            # ✅ Convertir archivo subido a imagen PIL
            img = Image.open(archivo_imagen)
            st.image(img, caption="Imagen cargada", use_container_width=True)

            # ✅ Guardar temporalmente
            temp_path = os.path.join(os.getcwd(), "temp.jpg")
            img.save(temp_path)

            if st.button("Buscar"):
                resultados = buscar_por_imagen(df, temp_path)
                if not resultados.empty:
                    st.success("✅ Resultados encontrados:")
                    st.dataframe(resultados)
                else:
                    st.warning("⚠️ No se encontraron coincidencias.")
        except Exception as e:
            st.error(f"Error al procesar la imagen: {e}")










































