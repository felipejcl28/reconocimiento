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
    # Normalizamos nombre de archivo de imagen para hacer el join
    if "IMAGEN" in df.columns:
        df["IMAGEN_NORM"] = df["IMAGEN"].apply(lambda p: os.path.basename(str(p)).lower())
    else:
        df["IMAGEN"] = ""
        df["IMAGEN_NORM"] = ""
    return df

def exportar_resultados(resultados):
    """Exporta resultados encontrados a Excel descargable"""
    output = BytesIO()
    df_resultados = pd.DataFrame(resultados)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_resultados.to_excel(writer, index=False, sheet_name="Resultados")
    output.seek(0)
    return output

def detectar_con_backends(img_path: str, backends: list, estricto: bool):
    """Intenta detectar rostro con varios backends usando extract_faces.
    Retorna (backend_exitoso | None, faces | []).
    """
    for det in backends:
        try:
            faces = DeepFace.extract_faces(
                img_path=img_path,
                detector_backend=det,
                enforce_detection=estricto
            )
            if faces and len(faces) > 0:
                return det, faces
        except Exception:
            # Probamos el siguiente backend
            pass
    return None, []

def asegurar_tamano_minimo(path_in: str) -> str:
    """Si la imagen es muy pequeña, la reescala para ayudar a la detección."""
    try:
        with Image.open(path_in) as im:
            w, h = im.size
            # Consideramos pequeño si el lado mayor < 300 px
            if max(w, h) < 300:
                factor = 300 / max(w, h)
                new_size = (max(1, int(w * factor)), max(1, int(h * factor)))
                im = im.resize(new_size, Image.LANCZOS)
                path_out = os.path.join(os.path.dirname(path_in), "temp_upscaled.jpg")
                im.save(path_out, quality=95)
                return path_out
    except Exception:
        pass
    return path_in

# ==============================
# INTERFAZ STREAMLIT
# ==============================
st.set_page_config(page_title="Búsqueda de Personas", layout="centered")
st.title("🔍 Búsqueda de Personas")

# Cargar datos
df = cargar_datos()
df["NOMBRE_NORM"] = df["NOMBRE"].apply(normalizar_texto) if "NOMBRE" in df.columns else ""

# Control de búsqueda
if "busqueda_realizada" not in st.session_state:
    st.session_state.busqueda_realizada = False

# Opciones de búsqueda
opcion = st.radio("Elige cómo buscar:", ["Por nombre", "Por ID", "Por imagen"])

# ======== Controles comunes a imagen ========
model_name = st.sidebar.selectbox(
    "Modelo de reconocimiento",
    ["VGG-Face", "Facenet512", "ArcFace"],
    index=0
)
# Umbrales sugeridos (aprox.)
UMBRAL_SUG = {"VGG-Face": 0.4, "Facenet512": 0.3, "ArcFace": 0.68}
umbral = st.sidebar.slider(
    "Umbral máximo de distancia (menor es más parecido)",
    min_value=0.0, max_value=1.0, value=UMBRAL_SUG.get(model_name, 0.4), step=0.01
)

detector_opcion = st.sidebar.selectbox(
    "Detector de rostro",
    ["Auto (mtcnn→opencv→mediapipe)", "mtcnn", "opencv", "mediapipe"],
    index=0
)

usar_deteccion = st.sidebar.checkbox("🔍 Detección estricta (enforce_detection)", value=True)
fallback_auto = st.sidebar.checkbox("Reintentar sin detección si falla", value=True)
upscale_auto = st.sidebar.checkbox("Mejorar imágenes pequeñas (upscale)", value=True)

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
        resultados = df[df["ID"] == id_buscar] if "ID" in df.columns else pd.DataFrame()
        st.session_state.busqueda_realizada = True

# ==============================
# BÚSQUEDA POR IMAGEN (robusta)
# ==============================
elif opcion == "Por imagen":
    imagen_subida = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

    if st.button("Buscar", key="buscar_imagen") and imagen_subida:
        # Guardar imagen temporal
        img_temp = os.path.join("temp.jpg")
        with open(img_temp, "wb") as f:
            f.write(imagen_subida.getbuffer())

        # Upscale opcional si es pequeña
        img_a_usar = asegurar_tamano_minimo(img_temp) if upscale_auto else img_temp
        if img_a_usar != img_temp:
            st.info("La imagen era pequeña. Se reescaló automáticamente para mejorar la detección.")

        # Selección de backends a probar
        if detector_opcion.startswith("Auto"):
            backends = ["mtcnn", "opencv", "mediapipe"]
        else:
            backends = [detector_opcion]

        try:
            # 1) Intento de detección previa para evitar error de DeepFace.find
            backend_exitoso, faces = detectar_con_backends(
                img_path=img_a_usar,
                backends=backends,
                estricto=usar_deteccion
            )

            enforce_final = usar_deteccion
            backend_final = backend_exitoso if backend_exitoso else backends[0]

            if usar_deteccion and backend_exitoso is None:
                if fallback_auto:
                    st.warning("No se detectó rostro con detección estricta. Reintentando sin detección…")
                    enforce_final = False
                else:
                    st.error("No se detectó rostro. Desactiva la detección estricta o activa el fallback automático.")
                    resultados = pd.DataFrame()
                    st.session_state.busqueda_realizada = True
                    # Mostrar aviso y salir de la rama imagen
                # no return en Streamlit, continuamos

            # 2) Buscar en la base
            resultados_busqueda = DeepFace.find(
                img_path=img_a_usar,
                db_path=RUTA_IMAGENES,
                enforce_detection=enforce_final,
                model_name=model_name,
                detector_backend=backend_final
            )

            if resultados_busqueda and not resultados_busqueda[0].empty:
                # Ordenados por distancia ascendente, tomamos top 3 y filtramos por umbral
                top = resultados_busqueda[0].sort_values("distance").query("distance <= @umbral").head(3)
                resultados = pd.DataFrame()
                for _, match in top.iterrows():
                    img_encontrada = os.path.basename(str(match["identity"])).lower()
                    distancia = float(match["distance"])
                    fila = df[df["IMAGEN_NORM"] == img_encontrada].copy()
                    if not fila.empty:
                        fila["DISTANCIA"] = round(distancia, 4)
                        resultados = pd.concat([resultados, fila], ignore_index=True)
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
    st.subheader("Resultados encontrados (Top 3 por distancia):")

    lista_resultados = []
    for _, row in resultados.iterrows():
        col1, col2 = st.columns([1, 2])

        with col1:
            img_path = os.path.join(RUTA_IMAGENES, row.get("IMAGEN", ""))
            if os.path.exists(img_path):
                st.image(img_path, width=180, caption=row.get("NOMBRE", ""))

        with col2:
            st.markdown(f"**🆔 ID:** {row.get('ID','')}")
            st.markdown(f"**👤 Nombre:** {row.get('NOMBRE','')}")
            st.markdown(f"**📄 Tipo ID:** {row.get('TIPO_DE_ID','')}")
            st.markdown(f"**🔑 NUNC:** {row.get('NUNC','')}")
            st.markdown(f"**📊 Distancia:** {row.get('DISTANCIA','')}")

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







































