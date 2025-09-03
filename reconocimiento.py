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

st.title("Búsqueda de personas - Galería con Carrusel (Seguro)")

# Contenedor fijo para los resultados
contenedor_resultados = st.container()

# Selección de búsqueda
opcion = st.radio("Buscar por:", ("Nombre", "Imagen"))

def mostrar_carrusel(resultados, columnas_por_fila=3):
    """Muestra resultados en filas con scroll horizontal"""
    if resultados.empty:
        st.error("No se encontraron resultados")
        return

    # Estilo CSS para scroll horizontal
    st.markdown(
        """
        <style>
        .carrusel {
            display: flex;
            overflow-x: auto;
            gap: 20px;
            padding: 10px;
        }
        .card {
            flex: 0 0 auto;
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 10px;
            background: #fafafa;
            min-width: 220px;
            max-width: 220px;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # HTML dinámico para la galería
    html = '<div class="carrusel">'
    for _, row in resultados.iterrows():
        html += '<div class="card">'
        try:
            img_path = os.path.join(RUTA_IMAGENES, row['Imagen'])
            st.image(Image.open(img_path), width=200)
        except:
            st.info("Imagen no encontrada")
        # Mostramos datos como texto
        for col in row.index:
            html += f"<p><b>{col}:</b> {row[col]}</p>"
        html += "</div>"
    html += "</div>"

    st.markdown(html, unsafe_allow_html=True)

if opcion == "Nombre":
    nombre_buscar = st.text_input("Ingrese el nombre completo:")
    if st.button("Buscar"):
        with contenedor_resultados:
            contenedor_resultados.empty()  # limpiar antes de dibujar
            if nombre_buscar.strip() == "":
                st.warning("Por favor ingresa un nombre")
            else:
                resultados = df[df["Nombre"].str.lower() == nombre_buscar.strip().lower()]
                mostrar_carrusel(resultados)

else:  # Búsqueda por imagen
    uploaded_file = st.file_uploader("Sube una imagen")
    if uploaded_file:
        with contenedor_resultados:
            contenedor_resultados.empty()
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
                    mostrar_carrusel(pd.DataFrame(coincidencias))
                else:
                    st.error("No se encontraron coincidencias")
            except Exception as e:
                st.error(f"Ocurrió un error: {e}")











