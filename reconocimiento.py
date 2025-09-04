import streamlit as st
import pandas as pd
import os
from PIL import Image
from deepface import DeepFace
from rapidfuzz import process

# Rutas de archivos
RUTA_EXCEL = os.path.join(os.getcwd(), "informacion.xlsx")
RUTA_IMAGENES = os.path.join(os.getcwd(), "IMAGENES")

# Función para cargar base de datos
@st.cache_data
def cargar_datos():
    return pd.read_excel(RUTA_EXCEL)

# Función búsqueda difusa por nombre o ID
def buscar_por_texto(df, query, columna):
    opciones = df[columna].astype(str).tolist()
    coincidencia, score, idx = process.extractOne(query, opciones)
    if score > 70:  # porcentaje de similitud
        return df.iloc[[idx]]
    else:
        return pd.DataFrame()

# Función búsqueda por imagen
def buscar_por_imagen(df, imagen_path):
    resultados = []
    for _, row in df.iterrows():
        try:
            img_db = os.path.join(RUTA_IMAGENES, row["Imagen"])
            resultado = DeepFace.verify(img1_path=imagen_path, img2_path=img_db, enforce_detection=False)
            if resultado["verified"]:
                resultados.append(row)
        except Exception as e:
            print(f"Error con {row['Imagen']}: {e}")
    return pd.DataFrame(resultados)

# Interfaz Streamlit
st.title("🔎 Búsqueda de Personas")

# Cargar base
df = cargar_datos()

opcion = st.radio("Elige cómo buscar:", ["Por nombre", "Por ID", "Por imagen"])

if opcion == "Por nombre":
    nombre = st.text_input("Ingrese el nombre:")
    if st.button("Buscar"):
        resultados = buscar_por_texto(df, nombre, "Nombre")
        if not resultados.empty:
            st.success("✅ Resultados encontrados:")
            st.dataframe(resultados)
        else:
            st.warning("⚠️ No se encontraron coincidencias.")

elif opcion == "Por ID":
    id_buscar = st.text_input("Ingrese el ID:")
    if st.button("Buscar"):
        resultados = buscar_por_texto(df, id_buscar, "ID")
        if not resultados.empty:
            st.success("✅ Resultados encontrados:")
            st.dataframe(resultados)
        else:
            st.warning("⚠️ No se encontraron coincidencias.")

elif opcion == "Por imagen":
    archivo_imagen = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
    if archivo_imagen:
        # ✅ Convertir archivo a objeto PIL antes de mostrarlo
        img = Image.open(archivo_imagen)
        st.image(img, caption="Imagen cargada", use_container_width=True)

        # Guardar temporalmente para comparar en DeepFace
        temp_path = "temp.jpg"
        img.save(temp_path)

        if st.button("Buscar"):
            resultados = buscar_por_imagen(df, temp_path)
            if not resultados.empty:
                st.success("✅ Resultados encontrados:")
                st.dataframe(resultados)
            else:
                st.warning("⚠️ No se encontraron coincidencias.")








































