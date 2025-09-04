import streamlit as st
import pandas as pd
import unicodedata
import os
from deepface import DeepFace
from PIL import Image
from io import BytesIO
import cv2

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

def upscale_image(img_path):
    """Mejora imágenes pequeñas usando superresolución de OpenCV"""
    try:
        img = cv2.imread(img_path)
        if img is not None and (img.shape[0] < 200 or img.shape[1] < 200):
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            sr.readModel("EDSR_x2.pb")  # necesitas descargar este modelo antes
            sr.setModel("edsr", 2)
            img = sr.upsample(img)
            cv2.imwrite(img_path, img)
    except Exception:
        pass  # si falla, simplemente seguimos con la original

# ==============================
# INTERFAZ STREAMLIT
# ==============================
st.set_page_config(page_title="Búsqueda de Personas", layout="wide")
st.title("🔍 Búsqueda de Personas")

# Cargar datos
df = cargar_datos()
df["NOMBRE_NORM"] = df["NOMBRE"].apply(normalizar_texto)

# Panel lateral
st.sidebar.header("⚙️ Configuración de búsqueda")

modelo = st.sidebar.selectbox(
    "Modelo de reconocimiento",
    ["VGG-Face", "Facenet", "Facenet512", "ArcFace", "Dlib", "SFace"],
    index=3
)

umbral = st.sidebar.slider(
    "Umbral máximo de distancia (menor es más parecido)",
    0.0, 1.0, 0.68, 0.01
)

detector = st.sidebar.selectbox(
    "Detector de rostro",
    ["mtcnn", "opencv", "retinaface", "mediapipe", "dlib", "auto"],
    index=0
)

usar_deteccion = st.sidebar.checkbox("🔍 Detección estricta (enforce_detection)", value=True)
retry_no_face = st.sidebar.checkbox("🔄 Reintentar sin detección si falla", value=True)
usar_upscale = st.sidebar.checkbox("📈 Mejorar imágenes pequeñas (upscale)", value=True)

# ==============================
# CONTROL DE BÚSQUEDA
# ==============================
if "busqueda_realizada" not in st.session_state:
    st.session_state.busqueda_realizada = False

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
# BÚSQUEDA POR IMAGEN
# ==============================
elif opcion == "Por imagen":
    imagen_subida = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
    if st.button("Buscar", key="buscar_imagen") and imagen_subida:
        img_temp = os.path.join("temp.jpg")
        with open(img_temp, "wb") as f:
            f.write(imagen_subida.getbuffer())

        if usar_upscale:
            upscale_image(img_temp)

        try:
            resultados_busqueda = DeepFace.find(
                img_path=img_temp,
                db_path=RUTA_IMAGENES,
                enforce_detection=usar_deteccion,
                model_name=modelo,
                detector_backend=detector if detector != "auto" else "mtcnn"
            )

            # ✅ Corrección del error "ambiguous DataFrame"
            if isinstance(resultados_busqueda, list) and len(resultados_busqueda) > 0 and not resultados_busqueda[0].empty:
                top_matches = (
                    resultados_busqueda[0]
                    .sort_values("distance")
                    .query("distance <= @umbral")
                    .head(3)
                )
                resultados = pd.DataFrame()

                for _, match in top_matches.iterrows():
                    img_encontrada = os.path.basename(match["identity"])
                    distancia = match["distance"]

                    fila = df[df["IMAGEN"] == img_encontrada].copy()
                    if not fila.empty:
                        fila["DISTANCIA"] = round(distancia, 4)
                        resultados = pd.concat([resultados, fila])

            else:
                resultados = pd.DataFrame()

        except Exception as e:
            if retry_no_face and "Face could not be detected" in str(e):
                st.warning("⚠️ No se detectó rostro con detección estricta. Reintentando sin detección...")
                try:
                    resultados_busqueda = DeepFace.find(
                        img_path=img_temp,
                        db_path=RUTA_IMAGENES,
                        enforce_detection=False,
                        model_name=modelo,
                        detector_backend=detector if detector != "auto" else "mtcnn"
                    )
                    if isinstance(resultados_busqueda, list) and len(resultados_busqueda) > 0 and not resultados_busqueda[0].empty:
                        top_matches = resultados_busqueda[0].sort_values("distance").query("distance <= @umbral").head(3)
                        resultados = pd.DataFrame()
                        for _, match in top_matches.iterrows():
                            img_encontrada = os.path.basename(match["identity"])
                            distancia = match["distance"]
                            fila = df[df["IMAGEN"] == img_encontrada].copy()
                            if not fila.empty:
                                fila["DISTANCIA"] = round(distancia, 4)
                                resultados = pd.concat([resultados, fila])
                except Exception as e2:
                    st.error(f"Error en la búsqueda sin detección: {e2}")
            else:
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

    output = exportar_resultados(lista_resultados)
    st.download_button(
        label="📥 Descargar resultados en Excel",
        data=output,
        file_name="resultados.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
elif st.session_state.busqueda_realizada:
    st.warning("⚠️ No se encontraron resultados.")




































