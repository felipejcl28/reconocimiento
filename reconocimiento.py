import os
import streamlit as st
import pandas as pd
from PIL import Image
import unicodedata
from io import BytesIO

# ✅ Solo si quieres búsqueda por imagen
from deepface import DeepFace
import cv2
import numpy as np

# ---------------- CONFIG ----------------
RUTA_EXCEL = os.path.join(os.getcwd(), "informacion.xlsx")
RUTA_IMAGENES = os.path.join(os.getcwd(), "IMAGENES")

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

def encontrar_por_imagen(img_path, df, ruta_imagenes):
    """
    Busca coincidencias de la imagen en la carpeta usando DeepFace.
    Devuelve DataFrame filtrado.
    """
    coincidencias = []
    try:
        for _, row in df.iterrows():
            img_ref_path = os.path.join(ruta_imagenes, row.get("IMAGEN",""))
            if not os.path.exists(img_ref_path):
                continue
            result = DeepFace.verify(img_path, img_ref_path, enforce_detection=False)
            if result["verified"]:
                coincidencias.append(row)
    except Exception as e:
        st.error(f"Error en búsqueda por imagen: {e}")
    if coincidencias:
        return pd.DataFrame(coincidencias)
    else:
        return pd.DataFrame()  # vacío si no hay coincidencias

# ---------------- CARGA DE DATOS ----------------
if not os.path.exists(RUTA_EXCEL):
    st.error(f"❌ No se encontró el archivo Excel en {RUTA_EXCEL}")
    st.stop()

df = pd.read_excel(RUTA_EXCEL)
df["NOMBRE_NORM"] = df["NOMBRE"].apply(normalizar_texto)
df["ID_NORM"] = df["ID"].astype(str).apply(normalizar_texto)
df["IMAGEN"] = df["IMAGEN"].astype(str).str.strip()

# ---------------- INTERFAZ ----------------
st.set_page_config(page_title="Consulta Personas", page_icon="🔎", layout="centered")
st.title("🔎 CONSULTA PERSONAS")

# ---------------- SELECCIÓN DE BÚSQUEDA ----------------
modo = st.radio("Modo de búsqueda:", ["Por texto", "Por imagen"])

if modo == "Por texto":
    criterio = st.selectbox("Buscar por:", ["NOMBRE", "ID"])
    query = st.text_input(f"Ingrese {criterio}:")
    
    if st.button("Buscar texto"):
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
                    col1, col2 = st.columns([1,2])
                    # Imagen
                    foto_path = os.path.join(RUTA_IMAGENES, row.get("IMAGEN",""))
                    if os.path.exists(foto_path) and row.get("IMAGEN"):
                        col1.image(Image.open(foto_path), width=250)
                    else:
                        col1.write(f"⚠️ No se encontró la imagen: {row.get('IMAGEN','')}")
                    # Información
                    col2.markdown(f"""
                        <div style="background:#f9f9f9;padding:10px;border-radius:10px;">
                        <p><b>👤 Nombre:</b> {row.get("NOMBRE","")}</p>
                        <p><b>🆔 ID:</b> {row.get("ID","")}</p>
                        <p><b>🏙 Municipio:</b> {row.get("MUNICIPIO ","")}</p>
                        <p><b>🔢 NUNC:</b> {row.get("NUNC","")}</p>
                        </div>
                    """, unsafe_allow_html=True)
            # Exportar
            resultados_export = resultados.drop(columns=["NOMBRE_NORM","ID_NORM"], errors="ignore")
            excel_data = exportar_excel(resultados_export)
            st.download_button(
                label="⬇️ Descargar resultados",
                data=excel_data,
                file_name="resultados.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

else:  # Por imagen
    uploaded_file = st.file_uploader("Sube la imagen para buscar coincidencias", type=["jpg","jpeg","png"])
    if uploaded_file is not None:
        # Guardamos temporalmente
        img_temp_path = os.path.join("temp_img.jpg")
        with open(img_temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("Buscar imagen"):
            resultados = encontrar_por_imagen(img_temp_path, df, RUTA_IMAGENES)
            if resultados.empty:
                st.warning("⚠️ No se encontraron coincidencias")
            else:
                st.success(f"✅ {len(resultados)} coincidencia(s) encontrada(s)")
                for _, row in resultados.iterrows():
                    with st.expander(f"{row['NOMBRE']} - {row['ID']}"):
                        col1, col2 = st.columns([1,2])
                        foto_path = os.path.join(RUTA_IMAGENES, row.get("IMAGEN",""))
                        if os.path.exists(foto_path) and row.get("IMAGEN"):
                            col1.image(Image.open(foto_path), width=250)
                        else:
                            col1.write(f"⚠️ No se encontró la imagen: {row.get('IMAGEN','')}")
                        col2.markdown(f"""
                            <div style="background:#f9f9f9;padding:10px;border-radius:10px;">
                            <p><b>👤 Nombre:</b> {row.get("NOMBRE","")}</p>
                            <p><b>🆔 ID:</b> {row.get("ID","")}</p>
                            <p><b>🏙 Municipio:</b> {row.get("MUNICIPIO ","")}</p>
                            <p><b>🔢 NUNC:</b> {row.get("NUNC","")}</p>
                            </div>
                        """, unsafe_allow_html=True)
                # Exportar
                resultados_export = resultados.drop(columns=["NOMBRE_NORM","ID_NORM"], errors="ignore")
                excel_data = exportar_excel(resultados_export)
                st.download_button(
                    label="⬇️ Descargar resultados",
                    data=excel_data,
                    file_name="resultados_imagen.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        # Limpiar imagen temporal
        if os.path.exists(img_temp_path):
            os.remove(img_temp_path)







