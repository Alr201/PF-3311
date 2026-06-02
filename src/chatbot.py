"""
Agente Virtual con RAG - Streamlit + ChromaDB + Gemini
=======================================================
Uso:
    pip install streamlit chromadb google-genai python-dotenv
    pip install faster-whisper edge-tts streamlit-audiorecorder
    streamlit run chatbot.py
"""

import os
import io
import base64
import tempfile
import re
import asyncio
import websockets
import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types
import streamlit as st
import edge_tts
from faster_whisper import WhisperModel


# ─── Configuración ────────────────────────────────────────────────────────────

load_dotenv()

CHROMA_PATH      = "./chroma_db"
COLLECTION_NAME  = "rag_presentaciones"
EMBED_MODEL      = "models/gemini-embedding-001"
CHAT_MODEL       = "gemini-2.5-flash"
TOP_K            = 5
MIN_SIMILARITY   = 0.4
WHISPER_MODEL    = "small"

# Voces disponibles en español:
# es-CR-JuanNeural    → Costa Rica, masculino
# es-CR-MariaNeural   → Costa Rica, femenino
# es-MX-DaliaNeural   → México, femenino (muy natural)
# es-MX-JorgeNeural   → México, masculino
# es-ES-ElviraNeural  → España, femenino
VOICE = "es-CR-JuanNeural"

SYSTEM_PROMPT = """Eres un agente virtual educativo con embodiment de la Universidad de Costa Rica (UCR), especializado en el material de los módulos de virtualidad y educación en línea.

Tu comportamiento:
1. Respondé SIEMPRE basándote primero en el contexto recuperado del material de los módulos.
2. Si el contexto es suficiente, respondé con esa información.
3. Si el contexto no es suficiente o no es relevante, podés usar tu conocimiento general pero avisalo claramente.
4. Respondé en español, de forma clara y amigable, pero trata de ser sereno y relajado para mantener un flujo de conversación más natural.
5. En ocasiones, tu respuesta va a darse en forma de TTS y tu input va a venir de STT, por lo que la naturalidad y fluidez de la conversación es vital.
6. Si te preguntan algo fuera del ámbito educativo/UCR, podés responder brevemente pero redirigí la conversación al material del curso.

Formato de respuesta cuando usás el RAG:
- Respondé la pregunta directamente
- Al final de tu respuesta agregá: "📚 Fuente: [Hipervínculo de la presentación]" para cada fuente usada
"""

# ─── Inicialización cacheada ──────────────────────────────────────────────────

@st.cache_resource
def init_clients():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("❌ Falta GEMINI_API_KEY en el archivo .env")
        st.stop()
    genai_client = genai.Client(api_key=api_key)
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    return genai_client, collection

@st.cache_resource
def init_whisper():
    """Carga el modelo Whisper una sola vez (puede tardar la primera vez)."""
    with st.spinner(f"Cargando modelo de voz ({WHISPER_MODEL})..."):
        return WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")

# ─── STT: Audio → Texto ───────────────────────────────────────────────────────

def transcribe_audio(audio_bytes: bytes, whisper_model) -> str:
    """Transcribe audio usando Whisper local."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    segments, _ = whisper_model.transcribe(
        tmp_path,
        language="es",
        beam_size=5,
        vad_filter=True,
    )
    os.unlink(tmp_path)
    return " ".join(seg.text for seg in segments).strip()

# ─── TTS: Texto → Audio (edge-tts) ───────────────────────────────────────────

def clean_for_tts(text: str) -> str:
    """Limpia markdown para que suene natural al leerlo en voz alta."""
    # Quitar formato markdown
    clean = re.sub(r"\*+|_+|#{1,6}\s?|`+", "", text)
    # Links: mantener solo el texto visible
    clean = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", clean)
    # Quitar emojis (edge-tts los lee literalmente a veces)
    clean = re.sub(r"[^\w\s\.\,\;\:\!\?\-\(\)áéíóúüñÁÉÍÓÚÜÑ]", "", clean)
    # Normalizar espacios
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean

LIPSYNC_SERVER = "ws://localhost:8765/audio"

async def _send_audio_to_lipsync(audio_bytes: bytes):
    """Envía el audio MP3 al servidor de lipsync para animar el avatar."""
    try:
        async with websockets.connect(LIPSYNC_SERVER, open_timeout=2) as ws:
            await ws.send(audio_bytes)
    except Exception:
        pass  # Si Unity no está conectado, no interrumpir el flujo

def send_to_lipsync(audio_bytes: bytes):
    """Wrapper síncrono para llamar desde Streamlit."""
    try:
        asyncio.run(_send_audio_to_lipsync(audio_bytes))
    except Exception:
        pass

async def _tts_async(text: str, voice: str) -> bytes:
    communicate = edge_tts.Communicate(text, voice)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    return buf.getvalue()

def text_to_speech(text: str, voice: str = VOICE) -> bytes:
    """Convierte texto a audio MP3 con edge-tts (voces neurales de Microsoft)."""
    clean = clean_for_tts(text)
    if not clean:
        return b""
    return asyncio.run(_tts_async(clean, voice))

def autoplay_audio(audio_bytes: bytes):
    """Inyecta HTML para reproducir audio automáticamente en Streamlit."""
    if not audio_bytes:
        return
    b64 = base64.b64encode(audio_bytes).decode()
    st.markdown(
        f'<audio autoplay controls src="data:audio/mp3;base64,{b64}"></audio>',
        unsafe_allow_html=True,
    )

# ─── RAG: recuperar contexto relevante ────────────────────────────────────────

def retrieve_context(query: str, genai_client, collection) -> tuple[str, list[dict]]:
    response = genai_client.models.embed_content(
        model=EMBED_MODEL,
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    query_embedding = response.embeddings[0].values

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    relevant = [
        (doc, meta, 1 - dist)
        for doc, meta, dist in zip(docs, metas, distances)
        if (1 - dist) >= MIN_SIMILARITY
    ]

    if not relevant:
        return "", []

    context_parts, sources = [], []
    for doc, meta, similarity in relevant:
        title = meta.get("enriched_title", meta.get("slide_name", "Sin título"))
        context_parts.append(f"[{title}]\n{doc}")
        sources.append({
            "title": title,
            "source": meta.get("source", ""),
            "similarity": similarity,
        })

    return "\n\n---\n\n".join(context_parts), sources

# ─── Chat con Gemini ──────────────────────────────────────────────────────────

def chat_with_rag(query: str, genai_client, collection) -> tuple[str, list[dict]]:
    context, sources = retrieve_context(query, genai_client, collection)

    if context:
        user_message = f"""
CONTEXTO DEL CURSO:

{context}

INSTRUCCIONES:
- Utiliza prioritariamente el contexto proporcionado.
- Continuá la conversación actual.
- No vuelvas a presentarte.
- No saludes nuevamente salvo que sea una nueva conversación.
- Si el contexto es insuficiente, indicalo claramente.

PREGUNTA DEL ESTUDIANTE: {query}"""
    else:
        user_message = f"""No se encontró contexto relevante en el material del curso.

Pregunta del estudiante: {query}"""

    response = st.session_state.chat.send_message(user_message)
    return response.text, sources

# ─── UI Streamlit ─────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Asistente Virtual UCR",
        page_icon="🎓",
        layout="wide"
    )

    # ── Layout: avatar izquierda, chat derecha ──
    col_avatar, col_chat = st.columns([1.2, 1], gap="large")

    with col_avatar:
        st.markdown("### 🎓 Asistente Virtual UCR")
        st.components.v1.iframe(
            src="http://localhost:8080",
            width=None,    # ocupa el ancho de la columna
            height=580,
            scrolling=False,
        )

    with col_chat:
        st.markdown("### 💬 Chat")
        st.caption("Escribí o usá el micrófono")

    genai_client, collection = init_clients()
    whisper_model = init_whisper()

    # Inicializar chat persistente
    if "chat" not in st.session_state:
        st.session_state.chat = genai_client.chats.create(
            model=CHAT_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.3,
                max_output_tokens=4096,
            )
        )

    # ── Sidebar ──
    with st.sidebar:
        st.header("📚 Base de conocimiento")
        st.metric("Chunks indexados", collection.count())
        st.divider()
        st.markdown("**Configuración RAG**")
        st.slider("Chunks a recuperar", 1, 10, TOP_K, key="top_k")
        st.slider("Similitud mínima", 0.0, 1.0, MIN_SIMILARITY, 0.05, key="min_sim")
        st.divider()
        st.markdown("**Voz**")
        voice = st.selectbox("Voz", [
            "es-CR-MariaNeural",
            "es-CR-JuanNeural",
            "es-MX-JorgeNeural",
            "es-MX-DaliaNeural",
            "es-ES-AlvaroNeural",
            "es-ES-ElviraNeural",
        ], key="voice")
        tts_enabled = st.toggle("🔊 Respuesta en audio", value=True)
        st.divider()
        if st.button("🗑️ Limpiar conversación"):
            st.session_state.messages = []
            del st.session_state.chat
            st.rerun()

    # Inicializar historial
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with col_chat:
        # Mostrar historial
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander("📎 Fuentes consultadas"):
                        for src in msg["sources"]:
                            st.markdown(f"- **{src['title']}** `{src['similarity']:.0%}` — [ver presentación]({src['source']})")

        # ── Input por micrófono ──
        audio_input = st.audio_input("🎤 Grabá tu pregunta")

        prompt = None

        if audio_input is not None:
            audio_key = audio_input.size
            if st.session_state.get("last_audio_key") != audio_key:
                st.session_state.last_audio_key = audio_key
                with st.spinner("Transcribiendo..."):
                    transcript = transcribe_audio(audio_input.read(), whisper_model)
                if transcript:
                    st.info(f"🗣 *{transcript}*")
                    prompt = transcript

        # ── Input por texto ──
        text_input = st.chat_input("O escribí tu pregunta aquí...")
        if text_input:
            prompt = text_input

        # ── Procesar prompt ──
        if prompt:
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("assistant"):
                with st.spinner("Buscando en el material..."):
                    response, sources = chat_with_rag(prompt, genai_client, collection)

                st.markdown(response)

                if sources:
                    with st.expander("📎 Fuentes consultadas"):
                        for src in sources:
                            st.markdown(f"- **{src['title']}** `{src['similarity']:.0%}` — [ver presentación]({src['source']})")

                if tts_enabled:
                    with st.spinner("Generando audio..."):
                        audio_bytes = text_to_speech(response, voice=st.session_state.voice)
                    autoplay_audio(audio_bytes)
                    send_to_lipsync(audio_bytes)

            st.session_state.messages.append({
                "role": "model",
                "content": response,
                "sources": sources,
            })


if __name__ == "__main__":
    main()
