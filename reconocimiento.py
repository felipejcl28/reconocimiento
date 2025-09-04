import streamlit as st
import pandas as pd
import unicodedata
import os
from deepface import DeepFace
from PIL import Image
from io import BytesIO

# ==============================
# CONFIGURACIÓN DE RUTAS
# ==============================
RUTA_EXCEL = os.path.join(os.getcwd(), "informacion.xlsx")
RUTA_IMAGENES = os.path.join(os.getcwd(), "IMAGENES")

# ==============================
# FUNCIONES AUXILIARES
# ==============================
def normalizar_texto(texto: str) -> str:
    """Normaliza texto: minúsculas, sin tildes ni espacios extra"""
    if not isinstance(texto, str):
        return ""
    texto = texto.strip().lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return texto

def cargar_datos():
    """Carga el Excel y normaliza nombres de columnas"""
    df = pd.read_excel(RUTA_EXCEL, dtype=str)
    df = df.rename(columns=lambda x: x.strip().upper().replace(" ", "_"))
    return df

def exportar_resultados(resultados):
    """Exporta resultados encontrados a Excel descargable"""
    output = BytesIO()
    df_resultados = pd.DataFrame(resultados)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_resultados.to_excel(writer, index=False, sheet_name="Resultados")
    output.seek(0)
    return output

# ==============================
# INTERFAZ STREAMLIT
# ==============================
st.set_page_config(page_title="Búsqueda de Personas", layout="centered")
st.title("🔍 Búsqueda de Personas")

# Cargar datos
df = cargar_datos()
df["NOMBRE_NORM"] = df["NOMBRE"].apply(normalizar_texto)

# Control de búsqueda
if "busqueda_realizada" not in st.session_state:
    st.session_state.busqueda_realizada = False

# Opciones de búsqueda
opcion = st.radio("Elige cómo buscar:", ["Por nombre", "Por ID", "Por imagen"])

resultados = pd.DataFrame()

# ==============================
# BÚSQUEDA POR NOMBRE
# ==============================
if opcion == "Por nombre":
    nombre = st.text_input("Escribe el nombre (o parte del nombre) a buscar:")
    if st.button("Buscar", key="buscar_nombre"):
        nombre_norm = normalizar_texto(nombre)
        resultados = df[df["NOMBRE_NORM"].str.contains(nombre_norm, na=False)]
        st.session_state.busqueda_realizada = True

# ==============================
# BÚSQUEDA POR ID
# ==============================
elif opcion == "Por ID":
    id_buscar = st.text_input("Escribe el ID a buscar:")
    if st.button("Buscar", key="buscar_id"):
        resultados = df[df["ID"] == id_buscar]
        st.session_state.busqueda_realizada = True

# ==============================
# BÚSQUEDA POR IMAGEN (Top 3 matches con distancia)
# ==============================
elif opcion == "Por imagen":
    imagen_subida = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

    # Checkbox para controlar enforce_detection
    usar_deteccion = st.checkbox("🔍 Usar detección estricta de rostro (enforce_detection)", value=True)

    if st.button("Buscar", key="buscar_imagen") and imagen_subida:
        img_temp = os.path.join("temp.jpg")
        with open(img_temp, "wb") as f:
            f.write(imagen_subida.getbuffer())

        try:
            resultados_busqueda = DeepFace.find(
                img_path=img_temp,
                db_path=RUTA_IMAGENES,
                enforce_detection=usar_deteccion,
                model_name="VGG-Face",
                detector_backend="mtcnn"
            )

            if not resultados_busqueda[0].empty:
                # Tomamos los 3 mejores resultados
                top_matches = resultados_busqueda[0].head(3)
                resultados = pd.DataFrame()

                for _, match in top_matches.iterrows():
                    img_encontrada = os.path.basename(match["identity"])
                    distancia = match["distance"]

                    # Buscar en el Excel
                    fila = df[df["IMAGEN"] == img_encontrada].copy()
                    if not fila.empty:
                        fila["DISTANCIA"] = round(distancia, 4)  # añadimos distancia al DataFrame
                        resultados = pd.concat([resultados, fila])
            else:
                resultados = pd.DataFrame()

        except Exception as e:
            st.error(f"Error en la búsqueda por imagen: {e}")
            resultados = pd.DataFrame()

        st.session_state.busqueda_realizada = True

# ==============================
# MOSTRAR RESULTADOS
# ==============================
if not resultados.empty:
    st.subheader("Resultados encontrados (Top 3):")

    lista_resultados = []
    for _, row in resultados.iterrows():
        col1, col2 = st.columns([1, 2])

        with col1:
            img_path = os.path.join(RUTA_IMAGENES, row["IMAGEN"])
            if os.path.exists(img_path):
                st.image(img_path, width=180, caption=row["NOMBRE"])

        with col2:
            st.markdown(f"**🆔 ID:** {row['ID']}")
            st.markdown(f"**👤 Nombre:** {row['NOMBRE']}")
            st.markdown(f"**📄 Tipo ID:** {row['TIPO_DE_ID']}")
            st.markdown(f"**🔑 NUNC:** {row['NUNC']}")
            st.markdown(f"**📊 Distancia de similitud:** {row['DISTANCIA']}")

        st.markdown("---")
        lista_resultados.append(row.to_dict())

    # Descargar Excel
    output = exportar_resultados(lista_resultados)
    st.download_button(
        label="📥 Descargar resultados en Excel",
        data=output,
        file_name="resultados.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

elif st.session_state.busqueda_realizada:
    st.warning("⚠️ No se encontraron resultados.")






































