import streamlit as st
import pandas as pd
import os
from PIL import Image
from deepface import DeepFace
import unicodedata
import io

# Configuración inicial
st.set_page_config(page_title="Búsqueda de Personas", page_icon="🔍", layout="wide")

# ===================== ESTILOS =====================
st.markdown(
    """
    <style>
    /* Banner fijo */
    .banner {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        background: linear-gradient(90deg, #2C3E50, #34495E);
        color: white;
        text-align: center;
        padding: 15px 0;
        font-size: 28px;
        font-weight: bold;
        z-index: 1000;
        box-shadow: 0px 4px 8px rgba(0,0,0,0.2);
    }

    /* Espaciador para que el contenido no quede oculto detrás del banner */
    .espaciador {
        margin-top: 80px;
    }

    /* Contenedor centrado */
    .contenedor {
        max-width: 900px;
        margin: auto;
    }

    /* Tarjetas para mostrar resultados */
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

# ===================== BANNER =====================
st.markdown('<div class="banner">🔍 Búsqueda de Personas</div>', unsafe_allow_html=True)
st.markdown('<div class="espaciador"></div>', unsafe_allow_html=True)
st.markdown('<div class="contenedor">', unsafe_allow_html=True)

# ===================== RUTAS =====================
RUTA_EXCEL = os.path.join(os.getcwd(), "informacion.xlsx")
RUTA_IMAGENES = os.path.join(os.getcwd(), "IMAGENES")

# ===================== FUNCIONES =====================
def normalizar_texto(texto: str) -> str:
    if not isinstance(texto, str):
        return ""
    texto = texto.strip().lower()
    texto = unicodedata.normalize("NFD", texto)
    return "".join(c for c in texto if unicodedata.category(c) != "Mn")

# ===================== CARGAR EXCEL =====================
@st.cache_data
def cargar_datos():
    return pd.read_excel(RUTA_EXCEL)

df = cargar_datos()

# ===================== OPCIONES DE BÚSQUEDA =====================
opcion = st.radio("Selecciona el método de búsqueda:", ["Por Nombre", "Por ID", "Por Imagen"])

resultados = pd.DataFrame()

# ----------------- BÚSQUEDA POR NOMBRE -----------------
if opcion == "Por Nombre":
    nombre_busqueda = st.text_input("Ingresa el nombre (o parte del nombre):")
    if st.button("Buscar"):
        if nombre_busqueda.strip() == "":
            st.warning("⚠️ Por favor ingresa un nombre válido.")
        else:
            nombre_normalizado = normalizar_texto(nombre_busqueda)
            df["Nombre_normalizado"] = df["Nombre"].apply(normalizar_texto)
            resultados = df[df["Nombre_normalizado"].str.contains(nombre_normalizado, na=False, case=False)]

# ----------------- BÚSQUEDA POR ID -----------------
elif opcion == "Por ID":
    id_busqueda = st.text_input("Ingresa el ID:")
    if st.button("Buscar"):
        if id_busqueda.strip() == "":
            st.warning("⚠️ Por favor ingresa un ID válido.")
        else:
            resultados = df[df["ID"].astype(str) == id_busqueda.strip()]

# ----------------- BÚSQUEDA POR IMAGEN -----------------
elif opcion == "Por Imagen":
    imagen_cargada = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
    if imagen_cargada is not None:
        if st.button("Buscar"):
            try:
                imagen_bytes = Image.open(imagen_cargada)
                rutas_imagenes = [os.path.join(RUTA_IMAGENES, img) for img in os.listdir(RUTA_IMAGENES)]
                coincidencias = []
                for ruta in rutas_imagenes:
                    try:
                        resultado = DeepFace.verify(img1_path=imagen_cargada, img2_path=ruta, enforce_detection=False)
                        if resultado["verified"]:
                            nombre_imagen = os.path.basename(ruta)
                            fila = df[df["Imagen"] == nombre_imagen]
                            if not fila.empty:
                                coincidencias.append(fila)
                    except Exception:
                        pass
                if coincidencias:
                    resultados = pd.concat(coincidencias)
                else:
                    st.error("❌ No se encontraron coincidencias con la imagen subida.")
            except Exception as e:
                st.error(f"Error al procesar la imagen: {e}")

# ===================== MOSTRAR RESULTADOS =====================
if not resultados.empty:
    for _, row in resultados.iterrows():
        st.markdown(
            f"""
            <div class="tarjeta">
                <p><b>ID:</b> {row['ID']}</p>
                <p><b>Nombre:</b> {row['Nombre']}</p>
                <p><b>NUNC:</b> {row.get('NUNC', 'N/A')}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Botón para descargar resultados
    buffer = io.BytesIO()
    resultados.to_excel(buffer, index=False)
    buffer.seek(0)
    st.download_button(
        label="📥 Descargar resultados en Excel",
        data=buffer,
        file_name="resultados.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
elif opcion != "Por Imagen":  # Evitar mostrar error doble en búsqueda por imagen
    if st.button("Verificar búsqueda vacía", disabled=True):  # Solo truco para evitar ejecución automática
        pass

# ===================== CIERRE CONTENEDOR =====================
st.markdown('</div>', unsafe_allow_html=True)
































