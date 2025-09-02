import os
import streamlit as st
import pandas as pd
from PIL import Image
import unicodedata
from io import BytesIO

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

# ---------------- BÚSQUEDA POR TEXTO ----------------
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

        # Mostrar resultados con expander (estable y seguro)
        for _, row in resultados.iterrows():
            with st.expander(f"{row['NOMBRE']} - {row['ID']}"):
                col1, col2 = st.columns([1,2])

                # Columna 1: Imagen
                foto_path = os.path.join(RUTA_IMAGENES, row.get("IMAGEN",""))
                if os.path.exists(foto_path) and row.get("IMAGEN"):
                    col1.image(Image.open(foto_path), width=250)
                else:
                    col1.write(f"⚠️ No se encontró la imagen: {row.get('IMAGEN','')}")

                # Columna 2: Información
                col2.markdown(f"""
                    <div style="background:#f9f9f9;padding:10px;border-radius:10px;">
                    <p><b>👤 Nombre:</b> {row.get("NOMBRE","")}</p>
                    <p><b>🆔 ID:</b> {row.get("ID","")}</p>
                    <p><b>🏙 Municipio:</b> {row.get("MUNICIPIO ","")}</p>
                    <p><b>🔢 NUNC:</b> {row.get("NUNC","")}</p>
                    </div>
                """, unsafe_allow_html=True)

        # Exportar resultados a Excel
        resultados_export = resultados.drop(columns=["NOMBRE_NORM","ID_NORM"], errors="ignore")
        excel_data = exportar_excel(resultados_export)
        st.download_button(
            label="⬇️ Descargar resultados",
            data=excel_data,
            file_name="resultados.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )






