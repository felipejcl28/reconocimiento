import os
import streamlit as st
import pandas as pd
from PIL import Image
from deepface import DeepFace
from rapidfuzz import process

# ==============================
# CONFIGURACIÓN DE RUTAS
# ==============================
RUTA_EXCEL = os.path.join(os.getcwd(), "informacion.xlsx")
RUTA_IMAGENES = os.path.join(os.getcwd(), "IMAGENES")

# ==============================
# FUNCIONES AUXILIARES
# ==============================
def cargar_datos():
    if not os.path.exists(RUTA_EXCEL):
        st.error("❌ No se encontró el archivo Excel")
        return pd.DataFrame()
    df = pd.read_excel(RUTA_EXCEL)
    df.columns = df.columns.str.strip().str.upper()
    return df

def mostrar_info_persona(row):
    st.write(f"**ID:** {row['ID']}")
    st.write(f"**Nombre:** {row['NOMBRE']}")
    st.write(f"**Tipo de ID:** {row['TIPO DE ID']}")
    st.write(f"**Municipio:** {row['MUNICIPIO']}")
    st.write(f"**NUNC:** {row['NUNC']}")
    
    ruta_img = os.path.join(RUTA_IMAGENES, str(row["IMAGEN"]))
    if os.path.exists(ruta_img):
        st.image(ruta_img, caption=row["NOMBRE"], use_container_width=True)
    else:
        st.warning("⚠️ Imagen no encontrada en carpeta IMAGENES.")

# ==============================
# APP STREAMLIT
# ==============================
st.title("🔍 Búsqueda de Personas")

df = cargar_datos()
if df.empty:
    st.stop()

opcion = st.radio("Elige cómo buscar:", ["Por nombre", "Por ID", "Por imagen"])

# ==============================
# BÚSQUEDA POR NOMBRE
# ==============================
if opcion == "Por nombre":
    nombre = st.text_input("Ingrese el nombre:")
    if st.button("Buscar"):
        if nombre.strip() == "":
            st.warning("⚠️ Ingrese un nombre.")
        else:
            # --- Coincidencia parcial
            resultados = df[df["NOMBRE"].str.contains(nombre.strip(), case=False, na=False)]

            # --- Si no encuentra, probar fuzzy
            if resultados.empty:
                candidatos = df["NOMBRE"].tolist()
                mejor, score, _ = process.extractOne(nombre, candidatos, score_cutoff=70)
                if mejor:
                    st.info(f"🔎 Coincidencia aproximada encontrada: **{mejor}** (similitud {score:.1f}%)")
                    resultados = df[df["NOMBRE"].str.contains(mejor, case=False, na=False)]

            if not resultados.empty:
                st.success(f"✅ Se encontraron {len(resultados)} resultados")
                for _, row in resultados.iterrows():
                    mostrar_info_persona(row)
            else:
                st.error("❌ No se encontró ninguna persona con ese nombre.")

# ==============================
# BÚSQUEDA POR ID
# ==============================
elif opcion == "Por ID":
    id_buscar = st.text_input("Ingrese el ID:")
    if st.button("Buscar"):
        resultados = df[df["ID"].astype(str) == id_buscar.strip()]
        if not resultados.empty:
            st.success("✅ Persona encontrada")
            for _, row in resultados.iterrows():
                mostrar_info_persona(row)
        else:
            st.error("❌ No se encontró ninguna persona con ese ID.")

# ==============================
# BÚSQUEDA POR IMAGEN
# ==============================
elif opcion == "Por imagen":
    archivo = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
    if archivo is not None:
        img = Image.open(archivo)
        st.image(img, caption="Imagen cargada", use_container_width=True)

        if st.button("Buscar"):
            encontrado = False
            for _, row in df.iterrows():
                ruta_img = os.path.join(RUTA_IMAGENES, str(row["IMAGEN"]))
                if not os.path.exists(ruta_img):
                    continue
                try:
                    resultado = DeepFace.verify(
                        img_path=archivo,
                        db_path=ruta_img,
                        model_name="ArcFace",
                        detector_backend="mtcnn",
                        enforce_detection=False,
                        distance_metric="euclidean_l2"
                    )
                    if resultado["verified"]:
                        st.success("✅ Persona encontrada")
                        mostrar_info_persona(row)
                        encontrado = True
                        break
                except Exception as e:
                    st.warning(f"Error con {row['IMAGEN']}: {e}")

            if not encontrado:
                st.error("❌ No se encontró coincidencia por imagen.")







































