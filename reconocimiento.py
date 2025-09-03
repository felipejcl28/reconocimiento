import streamlit as st
import pandas as pd
from PIL import Image
from deepface import DeepFace
import os

# Rutas relativas
RUTA_EXCEL = os.path.join(os.getcwd(), "informacion.xlsx")
RUTA_IMAGENES = os.path.join(os.getcwd(), "IMAGENES")

# Cargar datos
df = pd.read_excel(RUTA_EXCEL)

st.title("Búsqueda de personas - Galería con Carrusel")

# Placeholder para resultados
resultado_slot = st.empty()

# Selección de búsqueda
opcion = st.radio("Buscar por:", ("Nombre", "Imagen"))

def mostrar_carrusel(resultados, columnas_por_fila=3):
    """Muestra resultados en filas horizontales con scroll"""
    if resultados.empty:
        resultado_slot.error("No se encontraron resultados")
        return

    # Dividir resultados en chunks de tamaño columnas_por_fila
    for i in range(0, len(resultados), columnas_por_fila):
        fila = resultados.iloc[i:i+columnas_por_fila]
        # Crear columnas para la fila actual
        cols = resultado_slot.columns(columnas_por_fila)
        for col, (_, row) in zip(cols, fila.iterrows()):
            with col:
                st.write(row.to_dict())
                try:
                    img_path = os.path.join(RUTA_IMAGENES, row['Imagen'])
                    img = Image.open(img_path)
                    st.image(img, width=200)
                except:
                    st.info("Imagen no encontrada")

if opcion == "Nombre":
    nombre_buscar = st.text_input("Ingrese el nombre completo:")
    if st.button("Buscar"):
        resultado_slot.empty()
        if nombre_buscar.strip() == "":
            resultado_slot.warning("Por favor ingresa un nombre")
        else:
            resultados = df[df["Nombre"].str.lower() == nombre_buscar.strip().lower()]
            mostrar_carrusel(resultados, columnas_por_fila=3)

else:  # Búsqueda por imagen
    uploaded_file = st.file_uploader("Sube una imagen")
    if uploaded_file:
        resultado_slot.empty()
        try:
            img = Image.open(uploaded_file)
            img.save("temp.jpg")

            coincidencias = []
            for _, row in df.iterrows():
                try:
                    persona_img_path = os.path.join(RUTA_IMAGENES, row['Imagen'])
                    res = DeepFace.verify("temp.jpg", persona_img_path, enforce_detection=False)
                    if res["verified"]:
                        coincidencias.append(row)
                except:
                    continue

            if coincidencias:
                mostrar_carrusel(pd.DataFrame(coincidencias), columnas_por_fila=3)
            else:
                resultado_slot.error("No se encontraron coincidencias")
        except Exception as e:
            resultado_slot.error(f"Ocurrió un error: {e}")









