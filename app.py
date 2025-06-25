import re
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Chatbot-csv", page_icon="🌸", layout="centered")

# 🌟 CSS bonito
st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #fddde6, #ffe6f0); font-family: 'Comic Sans MS'; color: #a64d79; }
    h1 { color: #d63384; font-size: 3rem; text-align: center; margin-bottom: 1rem; }
    .stTextInput input { border-radius: 10px; border: 2px solid #d147a3; padding: 10px; }
    .stButton button {
        background: linear-gradient(90deg, #ff66b2, #ff99cc);
        border: none; border-radius: 20px;
        color: white; font-weight: bold; padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌸 Chatbot-csv 🌸")
archivo = st.file_uploader("📁 Sube tu archivo CSV", type="csv")

if archivo:
    df = pd.read_csv(archivo)
    with st.expander("📊 Ver datos del CSV"):
        st.dataframe(df, use_container_width=True)

    textos = df.astype(str).agg(' | '.join, axis=1).tolist()

    @st.cache_resource
    def cargar_modelo():
        return SentenceTransformer('all-MiniLM-L6-v2')

    modelo = cargar_modelo()

    @st.cache_resource
    def vectorizar(textos):
        return modelo.encode(textos, convert_to_tensor=True)

    embeddings = vectorizar(textos)

    pregunta = st.text_input("💬 Haz cualquier pregunta sobre los datos:")

    if pregunta:
        pregunta_lower = pregunta.lower()
        columnas = df.columns.str.lower().tolist()
        col_map = {col.lower(): col for col in df.columns}

        if "cuantos datos" in pregunta_lower or ("cuantas" in pregunta_lower and "filas" in pregunta_lower) or ("cuantas" in pregunta_lower and "columnas" in pregunta_lower):
            st.info(f"📊 El archivo tiene **{df.shape[0]} filas** y **{df.shape[1]} columnas**.")

        nombre_col = next((c for c in columnas if any(x in c for x in ["nombre", "producto", "alimento", "comida", "bebida"])), None)
        entidad_encontrada = None

        if nombre_col:
            posibles = df[col_map[nombre_col]].dropna().astype(str).unique()
            entidad_encontrada = next((n for n in posibles if n.lower() in pregunta_lower), None)

        if entidad_encontrada:
            filtro = df[df[col_map[nombre_col]].astype(str).str.lower().str.contains(entidad_encontrada.lower())]
            if not filtro.empty:
                fila = filtro.iloc[0]
                resultado = {}

                if "calor" in pregunta_lower:
                    c = next((x for x in columnas if "calor" in x), None)
                    if c: resultado["Calorías"] = fila[col_map[c]]

                if "vegetarian" in pregunta_lower:
                    c = next((x for x in columnas if "vegetariano" in x), None)
                    if c: resultado["¿Es vegetariano?"] = fila[col_map[c]]

                if "tipo" in pregunta_lower:
                    c = next((x for x in columnas if "tipo" in x), None)
                    if c: resultado["Tipo"] = fila[col_map[c]]

                if "precio" in pregunta_lower or "cuesta" in pregunta_lower:
                    c = next((x for x in columnas if "precio" in x or "costo" in x), None)
                    if c: resultado["Precio"] = fila[col_map[c]]

                if "stock" in pregunta_lower or "cantidad" in pregunta_lower:
                    c = next((x for x in columnas if "stock" in x or "cantidad" in x), None)
                    if c: resultado["Stock"] = fila[col_map[c]]

                if resultado:
                    st.success(f"📌 Datos de {entidad_encontrada}:")
                    for clave, valor in resultado.items():
                        st.markdown(f"**📌 {clave}:** {valor}")
                else:
                    st.warning("⚠️ No se encontró un dato específico en tu pregunta.")
            else:
                st.warning("❌ No encontré ese elemento en el archivo.")

        elif any(p in pregunta_lower for p in ["mayor", "más alto", "más caro", "máximo", "menor", "menos", "mínimo"]):
            cols_num = df.select_dtypes(include=["int", "float"]).columns
            for col in cols_num:
                if col.lower() in pregunta_lower:
                    val = df[col].min() if any(w in pregunta_lower for w in ["menor", "mínimo", "menos"]) else df[col].max()
                    fila = df[df[col] == val]
                    st.success(f"🔢 Valor {'mínimo' if 'menor' in pregunta_lower else 'máximo'} en {col}: {val}")
                    st.dataframe(fila)
                    break
            else:
                st.warning("⚠️ No encontré columna numérica relevante.")

        elif "fecha" in pregunta_lower:
            col_fecha = next((c for c in columnas if "fecha" in c or "ingreso" in c), None)
            if col_fecha:
                df[col_map[col_fecha]] = pd.to_datetime(df[col_map[col_fecha]], errors="coerce")
                val = df[col_map[col_fecha]].min() if "antigua" in pregunta_lower or "primera" in pregunta_lower else df[col_map[col_fecha]].max()
                fila = df[df[col_map[col_fecha]] == val]
                st.success(f"📅 Fecha {'más antigua' if 'antigua' in pregunta_lower else 'más reciente'}: {val.date()}")
                st.dataframe(fila)
            else:
                st.warning("⚠️ No encontré columna con fecha.")

        else:
            emb = modelo.encode(pregunta, convert_to_tensor=True)
            sims = cosine_similarity(emb.reshape(1, -1), embeddings)[0]
            idx = np.argmax(sims)
            score = sims[idx]

            if score > 0.45:
                st.success("🔍 Resultado basado en contexto:")
                st.dataframe(df.iloc[[idx]])
                st.caption(f"Similitud: {score:.2f}")
            else:
                st.warning("❌ No encontré nada relevante para esa pregunta.")
