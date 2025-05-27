import os
import streamlit as st
import base64
from openai import OpenAI
import openai
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# --- Función para codificar imagen a base64 ---
def encode_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
            return encoded_image
    except FileNotFoundError:
        return "Error: Imagen no encontrada."

# --- Configuración de la app Streamlit ---
st.set_page_config(page_title='Tablero Inteligente')
st.title('Tablero Inteligente')

with st.sidebar:
    st.subheader("Acerca de:")
    st.subheader("Esta aplicación reconoce una placa vehicular dibujada y valida el acceso.")

st.subheader("Dibuja la placa en el panel y presiona el botón para analizarla")

# --- Canvas para dibujo ---
drawing_mode = "freedraw"
stroke_width = st.sidebar.slider('Selecciona el ancho de línea', 1, 30, 5)
stroke_color = "#000000"
bg_color = '#FFFFFF'

canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=300,
    width=400,
    drawing_mode=drawing_mode,
    key="canvas",
)

# --- Entrada de API Key ---
ke = st.text_input('Ingresa tu clave de API de OpenAI', type='password')
os.environ['OPENAI_API_KEY'] = ke
api_key = os.environ['OPENAI_API_KEY']

# --- Inicializar cliente OpenAI ---
client = OpenAI(api_key=api_key)

# --- Botón de análisis ---
analyze_button = st.button("Analiza la imagen", type="secondary")

if canvas_result.image_data is not None and api_key and analyze_button:
    with st.spinner("Analizando imagen..."):

        # Guardar la imagen dibujada
        input_numpy_array = np.array(canvas_result.image_data)
        input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA')
        input_image_rgb = input_image.convert("RGB")
        input_image_rgb.save('img.png')

        # Codificar en base64
        base64_image = encode_image_to_base64("img.png")

        # Prompt para OpenAI
        prompt_text = "Extrae y transcribe la placa vehicular (números y letras) que aparece en esta imagen."

        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                            },
                        ],
                    }
                ],
                max_tokens=100,
            )

            if response.choices[0].message.content:
                resultado = response.choices[0].message.content.upper().strip()

                st.markdown("**Texto detectado:**")
                st.code(resultado)

                # Validar contra placas autorizadas
                placas_autorizadas = ["CKN 364", "MXL 931"]
                if any(placa in resultado for placa in placas_autorizadas):
                    st.success(f"✅ ACCESO PERMITIDO: Placa reconocida: {resultado}")
                else:
                    st.error(f"⛔ ACCESO DENEGADO: Placa no autorizada. Detectado: {resultado}")

        except Exception as e:
            st.error(f"Error con la API de OpenAI: {e}")
else:
    if not api_key:
        st.warning("⚠️ Por favor ingresa tu clave de API.")
