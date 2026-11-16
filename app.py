import time

import pandas as pd
import streamlit as st
import torch
from transformers import pipeline

# ======================= Configuración =======================
MODELO_TEXTO = "Qwen/Qwen2.5-0.5B-Instruct"  # modelo pequeño (~1 GB en bfloat16), cabe en la nube gratuita
HAY_GPU = torch.cuda.is_available()
NUM_HILOS_CPU = 2           # Community Cloud ofrece 2 núcleos; evita saturarlos
MAX_FILAS_COMPLETAS = 30    # si la tabla tiene más filas, al modelo solo se le envía una muestra
MAX_COLUMNAS = 15           # columnas máximas que se describen al modelo
MAX_MB_CSV = 5              # tamaño máximo del archivo CSV
MENSAJES_MEMORIA = 6        # cuántos mensajes previos recuerda el chatbot
MAX_TOKENS_RESPUESTA = 300  # longitud máxima de cada respuesta (más tokens = más lento en CPU)
MAX_CARACTERES_PREGUNTA = 1000

st.set_page_config(page_title="Chatbot Hugging Face", page_icon="🤖", layout="wide")

if not HAY_GPU:
    torch.set_num_threads(NUM_HILOS_CPU)


# ======================= Carga del modelo =======================
# Streamlit vuelve a ejecutar todo el script en cada interacción.
# @st.cache_resource hace que el modelo se cargue UNA sola vez y se comparta entre visitantes.
@st.cache_resource(show_spinner="Cargando el modelo (solo la primera vez, puede tardar un par de minutos)...")
def cargar_modelo_texto():
    return pipeline(
        "text-generation",
        model=MODELO_TEXTO,
        dtype=torch.bfloat16,  # la mitad de memoria que float32
        device=0 if HAY_GPU else -1,
    )


# ======================= Funciones auxiliares =======================
def leer_csv(archivo):
    """Lee un CSV detectando automáticamente el separador (, o ;) y la codificación."""
    if archivo.size > MAX_MB_CSV * 1024 * 1024:
        raise ValueError(f"El archivo supera el máximo de {MAX_MB_CSV} MB.")
    for codificacion in ("utf-8", "latin-1"):
        try:
            archivo.seek(0)
            return pd.read_csv(archivo, sep=None, engine="python", encoding=codificacion)
        except UnicodeDecodeError:
            continue
    raise ValueError("No se pudo leer el archivo CSV.")


def describir_tabla(df):
    """Convierte la tabla en texto que el modelo de lenguaje pueda leer."""
    total_filas, total_columnas = df.shape
    vista = df.iloc[:, :MAX_COLUMNAS]  # los modelos pequeños se pierden con demasiadas columnas
    partes = [f"La tabla tiene {total_filas} filas y {total_columnas} columnas."]
    if total_columnas > MAX_COLUMNAS:
        partes.append(f"Solo se muestran las primeras {MAX_COLUMNAS} columnas.")
    partes.append("Columnas (tipo de dato):\n" + "\n".join(f"- {c} ({t})" for c, t in vista.dtypes.astype(str).items()))
    if total_filas <= MAX_FILAS_COMPLETAS:
        partes.append("Datos completos (CSV):\n" + vista.to_csv(index=False))
    else:
        partes.append(
            f"Primeras {MAX_FILAS_COMPLETAS} filas de {total_filas} (CSV):\n"
            + vista.head(MAX_FILAS_COMPLETAS).to_csv(index=False)
        )
    # Los modelos pequeños se equivocan haciendo cuentas: les damos los cálculos hechos con pandas
    numericas = vista.select_dtypes("number")
    if not numericas.empty:
        partes.append("Suma total de cada columna numérica (calculada con pandas):\n" + numericas.sum().round(2).to_string())
    partes.append("Estadísticas descriptivas (calculadas con pandas):\n" + vista.describe(include="all").round(2).to_string())
    return "\n\n".join(partes)


def responder_texto(historial, df=None):
    """Genera la respuesta del chatbot con el modelo de texto de Hugging Face."""
    chat = cargar_modelo_texto()

    sistema = "Eres un asistente útil y amable. Responde siempre en español, de forma clara y concisa."
    if df is not None:
        sistema += (
            "\n\nEl usuario subió una tabla de datos. Responde sus preguntas usando ÚNICAMENTE "
            "la información de esta tabla. Si la respuesta no está en la tabla, dilo.\n\n"
            + describir_tabla(df)
        )

    previos = [
        {"role": m["rol"], "content": m["contenido"]}
        for m in historial
        if m["tipo"] == "texto"
    ][-MENSAJES_MEMORIA:]

    salida = chat(
        [{"role": "system", "content": sistema}] + previos,
        max_new_tokens=MAX_TOKENS_RESPUESTA,
        do_sample=True,
        temperature=0.2 if df is not None else 0.7,  # más preciso cuando responde sobre datos
        repetition_penalty=1.05,
    )
    return salida[0]["generated_text"][-1]["content"]


# ======================= Estado de la conversación =======================
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

# Cargar el modelo al abrir la app (así el primer visitante ve el aviso de carga y no una pantalla en blanco)
cargar_modelo_texto()


# ======================= Barra lateral =======================
with st.sidebar:
    st.header("⚙️ Opciones")

    st.subheader("📊 Tabla de datos")
    archivo = st.file_uploader(f"Sube un archivo CSV (máx. {MAX_MB_CSV} MB)", type=["csv"])
    df = None
    if archivo is not None:
        try:
            df = leer_csv(archivo)
            st.success(f"Tabla cargada: {df.shape[0]} filas × {df.shape[1]} columnas")
        except Exception as e:
            st.error(f"No se pudo leer el CSV: {e}")
    usar_tabla = st.checkbox("Responder usando la tabla", value=True, disabled=df is None)

    st.divider()
    if st.button("🗑️ Borrar conversación"):
        st.session_state.mensajes = []
        st.rerun()
    st.caption(f"Modelo: `{MODELO_TEXTO}`  \nDispositivo: {'GPU' if HAY_GPU else 'CPU'}")


# ======================= Página principal =======================
st.title("🤖 Chatbot con modelos de Hugging Face")
st.caption(
    "Chatea o haz preguntas sobre una tabla CSV. "
    "Usa un modelo pequeño en CPU: responde en pocos segundos, pero puede equivocarse. "
    "No ingreses datos personales ni confidenciales."
)

if df is not None:
    with st.expander("👀 Vista previa de la tabla"):
        st.dataframe(df)

# Mostrar el historial de la conversación
for m in st.session_state.mensajes:
    with st.chat_message(m["rol"]):
        st.markdown(m["contenido"])
        if m.get("info"):
            st.caption(m["info"])

# Entrada del usuario
prompt = st.chat_input(
    "Escribe tu mensaje o una pregunta sobre la tabla...",
    max_chars=MAX_CARACTERES_PREGUNTA,
)

if prompt:
    st.session_state.mensajes.append({"rol": "user", "tipo": "texto", "contenido": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        inicio = time.time()
        tabla = df if (df is not None and usar_tabla) else None
        try:
            with st.spinner("Pensando..."):
                respuesta = responder_texto(st.session_state.mensajes, tabla)
            info = f"{MODELO_TEXTO}{' · usando la tabla' if tabla is not None else ''} · ⏱️ {time.time() - inicio:.1f} s"
            st.session_state.mensajes.append({"rol": "assistant", "tipo": "texto", "contenido": respuesta, "info": info})
        except Exception as e:
            st.session_state.mensajes.append({
                "rol": "assistant",
                "tipo": "aviso",
                "contenido": f"⚠️ No pude generar la respuesta ({type(e).__name__}). Intenta de nuevo en unos segundos.",
            })
    st.rerun()
