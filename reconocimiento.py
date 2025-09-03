import streamlit as st
import pandas as pd
from PIL import Image
import os
import unicodedata
from io import BytesIO
from deepface import DeepFace

# =========================
# Configuración de rutas
# =========================
RUTA_EXCEL = os.path.join(os.getcwd(), "informacion.xlsx")
RUTA_IMAGENES = os.path.join(os.getcwd(), "IMAGENES")

# =========================
# Funciones auxiliares
# =========================
def normalizar_texto(texto: str) -> str:
    if not isinstance(texto, str):
        return ""
    texto = texto.strip().lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return texto

def cargar_datos():
    return pd.read_excel(RUTA_EXCEL, dtype=str)

def buscar_por_nombre(df, nombre):
    nombre_norm = normalizar_texto(nombre)
    resultados = df[df["NOMBRE_NORMALIZADO"].str.contains(nombre_norm, na=False, case=False)]
    return resultados

def buscar_por_id(df, id_persona):
    resultados = df[df["ID"].astype(str).str.contains(str(id_persona), na=False)]
    return resultados

def buscar_por_imagen(df, imagen_subida):
    img_bytes = imagen_subida.read()
    img = Image.open(BytesIO(img_bytes))

    resultados = []
    for _, row in df.iterrows():
        img_path = os.path.join(RUTA_IMAGENES, f"{row['ID']}.jpg")
        if os.path.exists(img_path):
            try:
                result = DeepFace.verify(img_bytes, img_path, enforce_detection=False)
                if result["verified"]:
                    resultados.append(row)
            except Exception as e:
                st.warning(f"Error comparando con {img_path}: {e}")
    return pd.DataFrame(resultados)

def exportar_resultados(resultados):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        resultados.to_excel(writer, index=False, sheet_name="Resultados")
    return output.getvalue()

# =========================
# Interfaz con Streamlit
# =========================
st.set_page_config(page_title="Búsqueda de Personas", layout="wide")

st.markdown("<h1 style='text-align: center;'>🔍 Búsqueda de Personas</h1>", unsafe_allow_html=True)

# Cargar datos
df = cargar_datos()
df["NOMBRE_NORMALIZADO"] = df["NOMBRE"].apply(normalizar_texto)

# Selección del tipo de búsqueda
opcion = st.radio("Elige cómo buscar:", ["Por nombre", "Por ID", "Por imagen"])

resultados = pd.DataFrame()

if opcion == "Por nombre":
    nombre = st.text_input("Escribe el nombre (o parte del nombre) a buscar:")
    if st.button("Buscar"):
        resultados = buscar_por_nombre(df, nombre)

elif opcion == "Por ID":
    id_persona = st.text_input("Escribe el número de ID a buscar:")
    if st.button("Buscar"):
        resultados = buscar_por_id(df, id_persona)

elif opcion == "Por imagen":
    imagen_subida = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
    if imagen_subida is not None:
        st.image(imagen_subida, caption="📷 Imagen cargada", use_container_width=True)  # ✅ actualizado
        if st.button("Buscar"):
            resultados = buscar_por_imagen(df, imagen_subida)

# Mostrar resultados
if not resultados.empty:
    for _, row in resultados.iterrows():
        st.image(os.path.join(RUTA_IMAGENES, f"{row['ID']}.jpg"), width=150)
        st.markdown(f"**ID:** {row['ID']}")
        st.markdown(f"**Nombre:** {row['NOMBRE']}")
        st.markdown(f"**Tipo ID:** {row['TIPO_DE_ID']}") 
        st.markdown(f"**NUNC:** {row['NUNC']}")
        st.markdown("---")

    # Botón para descargar resultados
    excel_bytes = exportar_resultados(resultados)
    st.download_button(
        label="⬇️ Descargar resultados en Excel",
        data=excel_bytes,
        file_name="resultados.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )



























