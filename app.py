import streamlit as st
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os, io, re, json, time, wave, requests
from datetime import datetime
import joblib, pandas as pd

# ===== VOZ (grabación en navegador) =====
from streamlit_mic_recorder import mic_recorder
from gtts import gTTS

# ===== PDF =====
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm

# ============================================
# CONFIGURACIÓN INICIAL
# ============================================
load_dotenv()
st.set_page_config(page_title="Orientador Inteligente", page_icon="🧭", layout="wide")

WEBHOOK_URL = "http://localhost:5678/webhook/orientador"

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    try:
        api_key = st.secrets["GEMINI_API_KEY"]  # Secrets de Streamlit Cloud
    except Exception:
        api_key = None
if not api_key:
    st.error("❌ No se encontró la API Key. Verifica el archivo .env")
    st.stop()

try:
    with open("system_prompt.txt", "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read()
except FileNotFoundError:
    st.error("❌ No se encontró system_prompt.txt")
    st.stop()

# ============================================
# CONEXIÓN + ROTACIÓN DE MODELOS
# ============================================
MODELOS = ["gemini-3.6-flash", "gemini-3.7-flash", "gemini-2.5-flash", "gemini-3.5-flash-lite"]

@st.cache_resource
def conectar():
    cliente = genai.Client(api_key=api_key)
    err = ""
    for nombre in MODELOS:
        try:
            cliente.models.generate_content(model=nombre, contents="Di OK")
            return cliente, nombre, ""
        except Exception as e:
            err = str(e)
    return None, None, err

client, modelo_activo, error_conexion = conectar()
if client is None:
    st.error(f"❌ No se pudo conectar con ningún modelo. Error exacto: {error_conexion}")
    st.stop()

# ===== MODELO ML (clasificador de perfil) =====
RUTAS_NOMBRES = {1:"Desarrollo Web Full Stack",2:"Análisis y Visualización de Datos",3:"IA Aplicada y Automatización",4:"Cloud Computing y Ciberseguridad",5:"Contenidos Digitales",6:"Marketing Digital",7:"E-Commerce y Emprendimiento",8:"Digital X",9:"Tech",10:"Profundización 4.0",11:"Bootcamp IA Aplicada",12:"Formación Google Cloud",13:"Cursos Senatic",14:"Licencias Platzi",15:"RPA con PIX Robotics",16:"Medellín Next",17:"Gen N-Proyector",18:"Bootcamp HubSpot CRM",19:"IA Sin Límites",20:"Medellinglish",21:"Fondos Sapiencia"}
try:
    modelo_ml = joblib.load("modelo_rutas_orientador.pkl")
except Exception:
    modelo_ml = None

def predecir_ml(interes, nivel, objetivo, perfil):
    d = pd.DataFrame([{"edad":25,"interes_principal":interes,"nivel_conocimiento":nivel,"objetivo":objetivo,
                       "tiempo_disponible":"Poco (66-92h)","experiencia_programacion":"Ninguna",
                       "perfil_actual":perfil,"modalidad_preferida":"Híbrida"}])
    d = pd.get_dummies(d).reindex(columns=modelo_ml.feature_names_in_, fill_value=False)
    return int(modelo_ml.predict(d)[0])

if "messages" not in st.session_state:
    st.session_state.messages = []

def llamar_modelo(contents, config=None):
    preferido = st.session_state.get("modelo_ok", modelo_activo)
    orden = [preferido] + [m for m in MODELOS if m != preferido]
    ultimo_error = ""
    for i, modelo in enumerate(orden):
        try:
            resp = client.models.generate_content(model=modelo, contents=contents, config=config)
            st.session_state["modelo_ok"] = modelo
            return resp.text
        except Exception as e:
            msg = str(e)
            ultimo_error = msg
            es_429 = any(k in msg.upper() for k in ["429", "RESOURCE_EXHAUSTED", "QUOTA", "LIMIT", "LÍMITE"])
            es_404 = any(k in msg.upper() for k in ["404", "NOT_FOUND"])
            if es_404:
                continue
            if es_429 and i < len(orden) - 1:
                st.info(f"⏳ {modelo} llegó a su límite; rotando a otro modelo…")
                continue
            if es_429:
                st.warning("⏳ Todos los modelos al límite. Esperando 60 s…")
                time.sleep(60)
                try:
                    resp = client.models.generate_content(model=modelo, contents=contents, config=config)
                    st.session_state["modelo_ok"] = modelo
                    return resp.text
                except Exception as e2:
                    ultimo_error = str(e2)
                    continue
            raise
    raise Exception(f"No se obtuvo respuesta. Error: {ultimo_error}")

# ============================================
# VOZ DE ENTRADA: TRANSCRIPCIÓN CON GEMINI (acepta webm)
# ============================================
def transcribir_audio(audio_bytes):
    mimes = ["audio/webm", "audio/ogg", "audio/mp4", "audio/wav"]
    for mime in mimes:
        try:
            resp = client.models.generate_content(
                model=st.session_state.get("modelo_ok", modelo_activo),
                contents=[
                    types.Part.from_bytes(data=audio_bytes, mime_type=mime),
                    "Transcribe literalmente lo que se dice en este audio. Devuelve únicamente el texto transcrito, sin comentarios."
                ]
            )
            texto = resp.text.strip()
            if texto:
                return texto
        except Exception:
            continue
    return None

# ============================================
# VOZ DE SALIDA: GEMINI TTS (natural) + gTTS (respaldo)
# ============================================
MODELOS_TTS = ["gemini-2.5-flash-preview-tts", "gemini-2.5-flash-tts"]
VOZ_ELEGIDA = "Kore"  # Alternativas: Leda, Puck, Zephyr, Charon, Orus

def limpiar_para_voz(texto, max_chars):
    limpio = re.sub(r'[*#`>|]', '', texto)
    limpio = re.sub(r'\n+', '. ', limpio).strip()
    if len(limpio) > max_chars:
        limpio = limpio[:max_chars] + ". Para más detalles, lee el texto en pantalla."
    return limpio

def texto_a_voz_gemini(texto):
    limpio = limpiar_para_voz(texto, 1000)
    for modelo_tts in MODELOS_TTS:
        try:
            response = client.models.generate_content(
                model=modelo_tts,
                contents=limpio,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=VOZ_ELEGIDA)
                        )
                    )
                )
            )
            audio_pcm = response.candidates[0].content.parts[0].inline_data.data
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(audio_pcm)
            buf.seek(0)
            return buf, "wav"
        except Exception:
            continue
    return None, None

def texto_a_voz_gtts(texto):
    limpio = limpiar_para_voz(texto, 600)
    tts = gTTS(text=limpio, lang="es")
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf, "mp3"

# ============================================
# FUNCIONES DE PDF
# ============================================
def esc(t):
    return (t or "-").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def fecha_hoy():
    MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio",
             "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
    h = datetime.now()
    return f"{h.day} de {MESES[h.month-1]} de {h.year}"

def extraer_datos_conversacion():
    historial = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages])
    pedido = (
        "A partir de esta conversación entre un usuario y el Orientador Inteligente, "
        "extrae la ruta recomendada y devuelve UN JSON con estas claves exactas: "
        "nombre_ruta, categoria, nivel, duracion, modalidad, institucion, descripcion, "
        "porque_recomienda, requisitos, siguiente_paso, url. "
        "Si algún dato no aparece, usa '-'. CONVERSACIÓN:\n" + historial
    )
    texto = llamar_modelo(
        pedido,
        config=types.GenerateContentConfig(temperature=0.3, response_mime_type="application/json")
    )
    return json.loads(texto)

def generar_pdf(datos, nombre):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=1.8*cm, bottomMargin=1.8*cm,
                            title="Guía de Ruta - Orientador Inteligente")
    styles = getSampleStyleSheet()
    azul = colors.HexColor('#1F4E79')
    azul_claro = colors.HexColor('#2E75B6')
    st_title = ParagraphStyle('t', parent=styles['Title'], textColor=azul, fontSize=20, spaceAfter=4)
    st_sub = ParagraphStyle('s', parent=styles['Normal'], textColor=azul_claro, fontSize=11, spaceAfter=10)
    st_h2 = ParagraphStyle('h', parent=styles['Heading2'], textColor=azul, fontSize=13, spaceBefore=10, spaceAfter=4)
    st_norm = ParagraphStyle('n', parent=styles['Normal'], fontSize=10.5, leading=15, spaceAfter=6)
    st_small = ParagraphStyle('x', parent=styles['Normal'], fontSize=8.5, textColor=colors.grey, leading=11)

    story = [
        Paragraph("ORIENTADOR INTELIGENTE", st_title),
        Paragraph("Sistema de orientación y recomendación de rutas | Ecosistema de Medellín", st_sub),
        HRFlowable(width="100%", thickness=1.2, color=azul_claro),
        Spacer(1, 0.4*cm),
        Paragraph(f"¡Hola, {esc(nombre)}!", st_h2),
        Paragraph(f"Medellín, {fecha_hoy()}", st_norm),
        Paragraph("Con base en nuestra conversación, preparamos para ti esta guía sencilla con la ruta que te recomendamos explorar. Léela con calma y da el siguiente paso cuando estés listo o lista.", st_norm),
        Spacer(1, 0.3*cm),
        Paragraph("TU RUTA RECOMENDADA", st_h2),
        Paragraph(f"<b>{esc(datos.get('nombre_ruta'))}</b>", ParagraphStyle('r', parent=st_norm, fontSize=13, textColor=azul_claro)),
        Paragraph(f"<b>Categoría:</b> {esc(datos.get('categoria'))}  |  <b>Nivel:</b> {esc(datos.get('nivel'))}", st_norm),
        Paragraph(f"<b>Duración:</b> {esc(datos.get('duracion'))}  |  <b>Modalidad:</b> {esc(datos.get('modalidad'))}", st_norm),
        Paragraph(f"<b>Institución:</b> {esc(datos.get('institucion'))}", st_norm),
        Paragraph("¿De qué se trata?", st_h2),
        Paragraph(esc(datos.get('descripcion')), st_norm),
        Paragraph("¿Por qué te la recomendamos?", st_h2),
        Paragraph(esc(datos.get('porque_recomienda')), st_norm),
        Paragraph("Requisitos principales", st_h2),
        Paragraph(esc(datos.get('requisitos')), st_norm),
        Paragraph("Tu siguiente paso", st_h2),
        Paragraph(esc(datos.get('siguiente_paso')), st_norm),
    ]
    url = datos.get('url', '')
    if url and url != '-':
        story += [Paragraph("Dónde consultar la información oficial", st_h2),
                  Paragraph(f'<a href="{url}" color="#2E75B6">{url}</a>', st_norm)]
    story += [Spacer(1, 0.6*cm),
              HRFlowable(width="100%", thickness=0.6, color=colors.lightgrey),
              Spacer(1, 0.2*cm),
              Paragraph("Este documento es una orientación generada por el Orientador Inteligente como proyecto académico. No representa oficialmente a ninguna institución. Verifica requisitos, fechas y disponibilidad en los canales oficiales de cada programa.", st_small)]
    doc.build(story)
    return buffer.getvalue()

def sanitizar(nombre):
    return re.sub(r'[^A-Za-z0-9ÁÉÍÓÚáéíóúÑñ ]', '', nombre).strip().replace(' ', '_') or "Usuario"

# ============================================
# SIDEBAR (incluye PDF para no interrumpir el chat)
# ============================================
with st.sidebar:
    st.title("🧭 Orientador Inteligente")
    st.success(f"✅ Modelo activo: {st.session_state.get('modelo_ok', modelo_activo)}")
    st.markdown("---")
    st.markdown("### 🎙️ Modo voz")
    voz_activada = st.toggle("🔊 Que el agente hable", value=True)
    motor_voz = st.selectbox("Motor de voz", ["Gemini TTS (natural)", "gTTS (clásica)"])
    auto_play = st.toggle("▶️ Reproducción automática", value=True)
    st.markdown("Presiona el micrófono, habla y suéltalo:")
    audio = mic_recorder(key="mic")
    st.markdown("---")
    st.markdown("### 📄 Guía personalizada")
    nombre_usuario = st.text_input("Tu nombre", "")
    correo_usuario = st.text_input("Tu correo (para enviarte la guía)", "")
    st.markdown("---")
    st.markdown("### 🤖 Clasificador ML")
    if modelo_ml is not None:
        ml_interes = st.selectbox("Interés", ["IA","Datos","Desarrollo Web","Programación","Marketing","Diseño","Emprendimiento","Ciberseguridad","Cloud","Blockchain","Automatización","Certificaciones","Idiomas","Becas"])
        ml_nivel = st.selectbox("Nivel", ["Ninguno","Básico","Intermedio","Avanzado"])
        ml_objetivo = st.selectbox("Objetivo", ["Empleo","Emprendimiento","Interés personal","Cambio de carrera","Mejorar perfil","Certificación","Escalar startup"])
        ml_perfil = st.selectbox("Perfil", ["Estudiante","Profesional","Directivo/Líder","Emprendedor","Desempleado"])
        if st.button("🎯 Clasificar mi perfil", use_container_width=True):
            st.session_state["ml_ruta"] = predecir_ml(ml_interes, ml_nivel, ml_objetivo, ml_perfil)
            st.success(f"Ruta ML: {st.session_state['ml_ruta']} — {RUTAS_NOMBRES.get(st.session_state['ml_ruta'])}")
    else:
        st.caption("Modelo ML no cargado (falta el .pkl)")


    if st.button("📥 Generar mi guía en PDF", use_container_width=True):
        if len(st.session_state.messages) == 0:
            st.warning("Primero conversa con el Orientador para que pueda recomendarte una ruta.")
        else:
            with st.spinner("Preparando tu guía personalizada..."):
                try:
                    datos = extraer_datos_conversacion()

                    # ===== AUTOMATIZACIÓN n8n =====
                    try:
                        r = requests.post(WEBHOOK_URL, json={
                            "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "nombre": nombre_usuario or "Estudiante",
                            "correo": correo_usuario,
                            "categoria": datos.get("categoria", "-"),
                            "nivel": datos.get("nivel", "-"),
                            "ruta_recomendada": datos.get("nombre_ruta", "-"),
                            "url_ruta": datos.get("url", "-"),
                            "siguiente_paso": datos.get("siguiente_paso", "-"),
                        }, timeout=10)
                        if r.status_code == 200:
                            st.success("⚙️ Registro enviado a n8n")
                        else:
                            st.warning(f"⚙️ n8n respondió código {r.status_code}")
                    except Exception:
                        st.warning("⚙️ n8n no está activo (la app sigue funcionando)")

                    pdf = generar_pdf(datos, nombre_usuario or "Estudiante")
                    st.download_button(
                        "⬇️ Descargar mi guía",
                        data=pdf,
                        file_name=f"Orientador_Inteligente_{sanitizar(nombre_usuario or 'Estudiante')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    st.success("✅ ¡Tu guía está lista!")
                except Exception as e:
                    st.error(f"❌ Error al generar el PDF: {e}")

    st.markdown("---")
    if st.button("🔄 Nueva conversación", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption("Proyecto académico — Fundamentos de IA | Agosto 2026")

# ============================================
# CHAT PRINCIPAL (limpio, sin interrupciones)
# ============================================
st.title("🧭 Orientador Inteligente")
# ===== VIDEO DE PRESENTACIÓN (requisito 4.3) =====
with st.expander("🎬 Conoce al Orientador Inteligente (video 30 s)"):
    st.video("orientador_promo.mp4")
st.markdown("*Te ayudo a encontrar una ruta de formación en el ecosistema de Medellín. Escribe o háblame.*")
st.markdown("---")

chat_container = st.container(height=560)
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if len(st.session_state.messages) == 0:
        with st.chat_message("assistant"):
            st.markdown("¡Hola! 👋 Soy el **Orientador Inteligente**. Cuéntame: **¿qué te gustaría aprender, mejorar o conseguir?** (también puedes hablarme con el micrófono 🎙️)")

# ===== CAPTURA DE ENTRADA (voz o texto) =====
prompt = None
if audio and audio.get("bytes"):
    firma = len(audio["bytes"])
    if st.session_state.get("last_audio_sig") != firma:
        st.session_state.last_audio_sig = firma
        texto_voz = transcribir_audio(audio["bytes"])
        if texto_voz:
            prompt = texto_voz
        else:
            st.warning("🎙️ No te entendí bien, intenta de nuevo.")

texto_escrito = st.chat_input("Escribe tu mensaje o usa el micrófono...")
if texto_escrito:
    prompt = texto_escrito

# ===== PROCESAR CONVERSACIÓN =====
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)
    with chat_container:
        with st.chat_message("assistant"):
            try:
                contents = [
                    types.Content(role="user" if m["role"] == "user" else "model",
                                  parts=[types.Part(text=m["content"])])
                    for m in st.session_state.messages
                ]
                respuesta = llamar_modelo(
                    contents,
                        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT + (
                        f"\n\nCONTEXTO DEL MODELO ML: El clasificador sugiere como hipótesis inicial la Ruta {st.session_state['ml_ruta']} ({RUTAS_NOMBRES.get(st.session_state['ml_ruta'],'')}). Contrástala con lo que el usuario te cuente y explica tu recomendación final."
                        if st.session_state.get("ml_ruta") else ""), temperature=0.7)
                )
                st.markdown(respuesta)
                if voz_activada:
                    if motor_voz.startswith("Gemini"):
                        buf, fmt = texto_a_voz_gemini(respuesta)
                        if buf is None:
                            buf, fmt = texto_a_voz_gtts(respuesta)
                    else:
                        buf, fmt = texto_a_voz_gtts(respuesta)
                    try:
                        st.audio(buf, format=f"audio/{fmt}", autoplay=auto_play)
                    except TypeError:
                        # Streamlit antiguo sin autoplay: muestra el player normal
                        st.audio(buf, format=f"audio/{fmt}")
                st.session_state.messages.append({"role": "assistant", "content": respuesta})
            except Exception as e:
                st.error(f"❌ Error: {e}")