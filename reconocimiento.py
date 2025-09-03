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
st.set_page_config(page_title="🔍 Búsqueda de Personas", layout="wide")

# CSS personalizado
st.markdown(
    """
    <style>
    .titulo {
        text-align: center;
        font-size: 36px;
        color: #2C3E50;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .tarjeta {
        padding: 15px;
        border-radius: 12px;
        background-color: #F8F9F9;
        margin-bottom: 15px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
    }
    .tarjeta p {
        margin: 5px 0;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Título
st.markdown('<p class="titulo">🔍 Búsqueda de Personas</p>', unsafe_allow_html=True)
st.write("---")

# Cargar datos
df = cargar_datos()
df["NOMBRE_NORM"] = df["NOMBRE"].apply(normalizar_texto)

# Control de búsqueda
if "busqueda_realizada" not in st.session_state:
    st.session_state.busqueda_realizada = False

# Opciones de búsqueda
st.subheader("📌 Selecciona el modo de búsqueda")
opcion = st.radio("", ["Por nombre", "Por ID", "Por imagen"])

resultados = pd.DataFrame()

# ==============================
# BÚSQUEDA POR NOMBRE
# ==============================
if opcion == "Por nombre":
    nombre = st.text_input("✍️ Escribe el nombre (o parte del nombre) a buscar:")
    if st.button("Buscar", key="buscar_nombre"):
        nombre_norm = normalizar_texto(nombre)
        resultados = df[df["NOMBRE_NORM"].str.contains(nombre_norm, na=False)]
        st.session_state.busqueda_realizada = True

# ==============================
# BÚSQUEDA POR ID
# ==============================
elif opcion == "Por ID":
    id_buscar = st.text_input("🆔 Escribe el ID a buscar:")
    if st.button("Buscar", key="buscar_id"):
        resultados = df[df["ID"] == id_buscar]
        st.session_state.busqueda_realizada = True

# ==============================
# BÚSQUEDA POR IMAGEN
# ==============================
elif opcion == "Por imagen":
    imagen_subida = st.file_uploader("📷 Sube una imagen", type=["jpg", "jpeg", "png"])
    if st.button("Buscar", key="buscar_imagen") and imagen_subida:
        img_temp = os.path.join("temp.jpg")
        with open(img_temp, "wb") as f:
            f.write(imagen_subida.getbuffer())

        encontrado = False
        for _, row in df.iterrows():
            img_path = os.path.join(RUTA_IMAGENES, row["IMAGEN"])
            try:
                result = DeepFace.verify(img1_path=img_temp, img2_path=img_path, enforce_detection=False)
                if result["verified"]:
                    resultados = pd.DataFrame([row])
                    encontrado = True
                    break
            except Exception as e:
                st.error(f"Error comparando con {img_path}: {e}")

        st.session_state.busqueda_realizada = True
        if not encontrado:
            resultados = pd.DataFrame()  # vacío para mostrar advertencia

# ==============================
# MOSTRAR RESULTADOS (con estilo)
# ==============================
if not resultados.empty:
    st.subheader("✅ Resultados encontrados")

    lista_resultados = []
    for _, row in resultados.iterrows():
        with st.container():
            col1, col2 = st.columns([1, 2])
            with col1:
                img_path = os.path.join(RUTA_IMAGENES, row["IMAGEN"])
                if os.path.exists(img_path):
                    st.image(img_path, width=200)
            with col2:
                st.markdown(
                    f"""
                    <div class="tarjeta">
                        <p><b>ID:</b> {row['ID']}</p>
                        <p><b>Nombre:</b> {row['NOMBRE']}</p>
                        <p><b>Tipo ID:</b> {row['TIPO_DE_ID']}</p>
                        <p><b>NUNC:</b> {row['NUNC']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
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































