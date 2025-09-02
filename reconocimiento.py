import os
import streamlit as st
import pandas as pd
from PIL import Image
import unicodedata
from io import BytesIO
from deepface import DeepFace

# ---------------- CONFIG ----------------
RUTA_EXCEL = os.path.join(os.getcwd(), "informacion.xlsx")
RUTA_IMAGENES = os.path.join(os.getcwd(), "IMAGENES")
TEMP_DIR = os.path.join(os.getcwd(), "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# ---------------- FUNCIONES ----------------
def normalizar_texto(texto: str) -> str:
    if not isinstance(texto, str):
        return ""
    texto = texto.strip().lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return texto

def exportar_excel(df: pd.DataFrame) -> BytesIO:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

# ---------------- CARGA DE DATOS ----------------
if not os.path.exists(RUTA_EXCEL):
    st.error(f"❌ No se encontró el archivo Excel en {RUTA_EXCEL}")
    st.stop()

df = pd.read_excel(RUTA_EXCEL)
df["NOMBRE_NORM"] = df["NOMBRE"].apply(normalizar_texto)
df["ID_NORM"] = df["ID"].astype(str).apply(normalizar_texto)
df["IMAGEN"] = df["IMAGEN"].astype(str).str.strip().str.lower()

# ---------------- INTERFAZ ----------------
st.set_page_config(page_title="Consulta Personas", page_icon="🔎", layout="centered")
st.title("🔎 CONSULTA PERSONAS")

modo_busqueda = st.radio("Selecciona modo de búsqueda:", ["Por texto", "Por imagen"])

# ---------------- BÚSQUEDA POR TEXTO ----------------
if modo_busqueda == "Por texto":
    criterio = st.selectbox("Buscar por:", ["NOMBRE", "ID"])
    query = st.text_input(f"Ingrese {criterio}:")

    if st.button("Buscar"):
        query_norm = normalizar_texto(query)

        if criterio == "NOMBRE":
            resultados = df[df["NOMBRE_NORM"].str.contains(query_norm, na=False)]
        else:
            resultados = df[df["ID_NORM"].str.contains(query_norm, na=False)]

        if resultados.empty:
            st.warning("⚠️ No se encontraron resultados")
        else:
            st.success(f"✅ {len(resultados)} resultado(s) encontrado(s)")

            for _, row in resultados.iterrows():
                with st.expander(f"{row['NOMBRE']} - {row['ID']}"):
                    cols = st.columns([1, 2])
                    foto_path = os.path.join(RUTA_IMAGENES, row.get("IMAGEN", ""))
                    if os.path.exists(foto_path) and row.get("IMAGEN"):
                        cols[0].image(Image.open(foto_path), width=250)
                    else:
                        cols[0].write(f"⚠️ No se encontró la imagen: {row.get('IMAGEN','')}")
                    cols[1].markdown(f"""
                        <div style="background:#f9f9f9;padding:10px;border-radius:10px;">
                        <p><b>👤 Nombre:</b> {row.get("NOMBRE", "")}</p>
                        <p><b>🆔 ID:</b> {row.get("ID", "")}</p>
                        <p><b>🏙 Municipio:</b> {row.get("MUNICIPIO ", "")}</p>
                        <p><b>🔢 NUNC:</b> {row.get("NUNC", "")}</p>
                        </div>
                    """, unsafe_allow_html=True)

            # Exportar Excel
            resultados_export = resultados.drop(columns=["NOMBRE_NORM", "ID_NORM"], errors="ignore")
            excel_data = exportar_excel(resultados_export)
            st.download_button(
                label="⬇️ Descargar resultados",
                data=excel_data,
                file_name="resultados.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

# ---------------- BÚSQUEDA POR IMAGEN ----------------
else:
    uploaded_file = st.file_uploader("Sube una imagen para buscar coincidencias", type=["jpg", "jpeg", "png"])
    if uploaded_file and st.button("Buscar"):
        st.info("🔎 Procesando imagen...")
        img_path = os.path.join(TEMP_DIR, uploaded_file.name)
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            result = DeepFace.find(img_path=img_path, db_path=RUTA_IMAGENES, enforce_detection=False)
            if result.empty:
                st.warning("⚠️ No se encontraron coincidencias")
            else:
                st.success(f"✅ {len(result)} coincidencia(s) encontrada(s)")
                for idx, row in result.iterrows():
                    with st.expander(f"Imagen: {os.path.basename(row['identity'])}"):
                        cols = st.columns([1,2])
                        foto_path = os.path.join(RUTA_IMAGENES, os.path.basename(row["identity"]))
                        if os.path.exists(foto_path):
                            cols[0].image(Image.open(foto_path), width=250)
                        cols[1].markdown(f"""
                            <div style="background:#f9f9f9;padding:10px;border-radius:10px;">
                            <p><b>Ruta imagen:</b> {row['identity']}</p>
                            <p><b>Distancia:</b> {row['VGG-Face_cosine']:.4f}</p>
                            </div>
                        """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Error al procesar la imagen: {e}")


