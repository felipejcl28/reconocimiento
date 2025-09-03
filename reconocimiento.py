import streamlit as st
import pandas as pd
import os
import unicodedata
from deepface import DeepFace
from PIL import Image
from io import BytesIO

# ==========================
# RUTAS
# ==========================
RUTA_EXCEL = os.path.join(os.getcwd(), "informacion.xlsx")
RUTA_IMAGENES = os.path.join(os.getcwd(), "IMAGENES")

# ==========================
# FUNCIONES AUXILIARES
# ==========================
def normalizar_texto(texto: str) -> str:
    """Normaliza texto quitando tildes y pasando a minúsculas"""
    if not isinstance(texto, str):
        return ""
    texto = texto.strip().lower()
    texto = unicodedata.normalize("NFD", texto)
    return "".join(c for c in texto if unicodedata.category(c) != "Mn")

def exportar_resultados(resultados):
    """Exporta resultados a Excel y permite descargar"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        resultados.to_excel(writer, index=False, sheet_name="Resultados")
    st.download_button(
        label="📥 Descargar resultados en Excel",
        data=output.getvalue(),
        file_name="resultados_busqueda.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ==========================
# CARGA DE DATOS
# ==========================
df = pd.read_excel(RUTA_EXCEL)
df["nombre_norm"] = df["NOMBRE"].apply(normalizar_texto)

# ==========================
# INTERFAZ STREAMLIT
# ==========================
st.title("🔎 Búsqueda de Personas")

modo_busqueda = st.radio("Elige cómo buscar:", ["Por nombre", "Por ID", "Por imagen"])

resultados = pd.DataFrame()

# ==========================
# BÚSQUEDA POR NOMBRE
# ==========================
if modo_busqueda == "Por nombre":
    nombre = st.text_input("Escribe el nombre (o parte del nombre) a buscar:")
    if st.button("Buscar", key="buscar_nombre"):
        if nombre:
            nombre_norm = normalizar_texto(nombre)
            resultados = df[df["nombre_norm"].str.contains(nombre_norm, na=False)]
        else:
            st.warning("Por favor escribe un nombre para buscar.")

# ==========================
# BÚSQUEDA POR ID
# ==========================
elif modo_busqueda == "Por ID":
    id_buscar = st.text_input("Escribe el ID a buscar:")
    if st.button("Buscar", key="buscar_id"):
        if id_buscar:
            resultados = df[df["ID"].astype(str).str.contains(id_buscar)]
        else:
            st.warning("Por favor escribe un ID para buscar.")

# ==========================
# BÚSQUEDA POR IMAGEN
# ==========================
elif modo_busqueda == "Por imagen":
    imagen_subida = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
    if imagen_subida and st.button("Buscar", key="buscar_imagen"):
        try:
            img_bytes = Image.open(imagen_subida)
            st.image(img_bytes, caption="📷 Imagen cargada", use_column_width=False)

            for _, row in df.iterrows():
                img_path = os.path.join(RUTA_IMAGENES, str(row["IMAGEN"]))
                try:
                    resultado = DeepFace.verify(img1_path=imagen_subida, img2_path=img_path, enforce_detection=False)
                    if resultado["verified"]:
                        resultados = pd.DataFrame([row])
                        break
                except Exception as e:
                    st.error(f"Error comparando con {img_path}: {e}")

        except Exception as e:
            st.error(f"Error al procesar la imagen: {e}")

# ==========================
# MOSTRAR RESULTADOS
# ==========================
if not resultados.empty:
    for _, row in resultados.iterrows():
        st.image(os.path.join(RUTA_IMAGENES, str(row["IMAGEN"])), width=150)
        st.markdown(f"**ID:** {row['ID']}")
        st.markdown(f"**Nombre:** {row['NOMBRE']}")
        st.markdown(f"**Tipo ID:** {row['TIPO DE ID']}")
        st.markdown(f"**NUNC:** {row['NUNC']}")
        st.write("---")

    # ✅ Botón para exportar a Excel
    exportar_resultados(resultados)
else:
    # Solo mostrar aviso si se hizo clic en algún botón
    if (modo_busqueda == "Por nombre" and st.session_state.get("buscar_nombre")) \
       or (modo_busqueda == "Por ID" and st.session_state.get("buscar_id")) \
       or (modo_busqueda == "Por imagen" and st.session_state.get("buscar_imagen")):
        st.warning("⚠️ No se encontraron coincidencias.")

























