import os
import streamlit as st
import base64
from openai import OpenAI
import openai
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pytesseract
from streamlit_drawable_canvas import st_canvas

# --- Función para codificar imagen a base64 ---
def encode_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
            return encoded_image
    except FileNotFoundError:
        return "Error: La imagen no se encontró en la ruta especificada."

# --- Configuración de la app Streamlit ---
st.set_page_config(page_title='Tablero Inteligente')
st.title('Tablero Inteligente')

with st.sidebar:
    st.subheader("Acerca de:")
    st.subheader("Esta aplicación reconoce una placa vehicular dibujada y valida el acceso.")

st.subheader("Dibuja la placa en el panel y presiona el botón para analizarla")

# --- Configuración del canvas para dibujo ---
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

# --- Entrada de clave de API ---
ke = st.text_input('Ingresa tu clave de API de OpenAI', type='password')
os.environ['OPENAI_API_KEY'] = ke
api_key = os.environ['OPENAI_API_KEY']

# --- Inicializar cliente OpenAI ---
client = OpenAI(api_key=api_key)

# --- Botón para procesar imagen ---
analyze_button = st.button("Analiza la imagen", type="secondary")

if canvas_result.image_data is not None and api_key and analyze_button:
    with st.spinner("Analizando ..."):
        # Convertir imagen de canvas a formato PIL
        input_numpy_array = np.array(canvas_result.image_data)
        input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA')
        input_image_rgb = input_image.convert("RGB")
        input_image_rgb.save('img.png')

        # --- OCR para detectar texto ---
        ocr_text = pytesseract.image_to_string(input_image_rgb)
        ocr_text = ocr_text.strip().upper().replace("\n", " ").replace("  ", " ")
        
        # --- Placas autorizadas ---
        placas_autorizadas = ["CKN 364", "MXL 931"]

        # --- Verificar acceso ---
        if any(placa in ocr_text for placa in placas_autorizadas):
            st.success(f"✅ ACCESO PERMITIDO: Placa reconocida: {ocr_text}")
        else:
            st.error(f"⛔ ACCESO DENEGADO: Placa no autorizada. Detectado: {ocr_text}")

        # --- Mostrar descripción con GPT (opcional) ---
        base64_image = encode_image_to_base64("img.png")
        prompt_text = "Describe en español brevemente el contenido de la imagen"

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
                max_tokens=500,
            )

            if response.choices[0].message.content:
                st.markdown("**Descripción generada por GPT:**")
                st.write(response.choices[0].message.content)

        except Exception as e:
            st.error(f"Ocurrió un error al usar la API de OpenAI: {e}")

else:
    if not api_key:
        st.warning("⚠️ Por favor ingresa tu API key.")
