"""
Agente Virtual con RAG - Streamlit + ChromaDB + Gemini
=======================================================
Uso:
    pip install streamlit chromadb google-genai python-dotenv faster-whisper edge-tts miniaudio
    streamlit run chatbot.py

Assets requeridos:
    ./assets/idle.json
    ./assets/thinking.json
    ./assets/speaking.json
"""

import os
import io
import json
import base64
import tempfile
import re
import asyncio
import time
import threading
import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types
import streamlit as st
import edge_tts
from faster_whisper import WhisperModel
import miniaudio

load_dotenv()

CHROMA_PATH     = "./chroma_db"
COLLECTION_NAME = "rag_presentaciones"
EMBED_MODEL     = "models/gemini-embedding-001"
CHAT_MODEL      = "gemini-2.5-flash"
TOP_K           = 5
MIN_SIMILARITY  = 0.4
WHISPER_MODEL   = "small"
VOICE           = "es-CR-MariaNeural"
ASSETS_DIR      = "./assets"

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

CSS = """
<style>
[data-testid="stMainBlockContainer"] {
    padding: 0 !important;
    max-width: 100% !important;
}
body, html { overflow: hidden !important; }

footer, #MainMenu, header { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }

[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child > div {
    position: fixed !important;
    top: 0; left: 0;
    width: 42vw !important;
    height: 100vh !important;
    background: var(--background-color, #0e1117);
    padding: 1.5rem 1rem 1rem 1.5rem !important;
    z-index: 10;
    border-right: 1px solid rgba(255,255,255,0.08);
    overflow: hidden;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child > div {
    position: fixed !important;
    top: 0;
    left: 42vw;
    width: 58vw !important;
    height: 100vh !important;
    display: flex !important;
    flex-direction: column !important;
    padding: 1rem 1rem 0 1rem !important;
    box-sizing: border-box !important;
    overflow: hidden !important;
}

[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child iframe {
    flex: 1 !important;
    border: none !important;
    min-height: 0 !important;
}

.input-bar-wrapper {
    flex-shrink: 0;
    padding-top: 0.5rem;
    border-top: 1px solid rgba(255,255,255,0.08);
}

[data-testid="stAudioInput"] label { display: none !important; }
[data-testid="stAudioInput"] button {
    padding: 0.25rem 0.75rem !important;
    font-size: 0.78rem !important;
    border-radius: 20px !important;
    min-height: unset !important;
}
</style>
"""

# ─── Avatar Lottie ────────────────────────────────────────────────────────────

@st.cache_resource
def load_animations() -> dict:
    animations = {}
    for state in ("idle", "thinking", "speaking"):
        path = os.path.join(ASSETS_DIR, f"{state}.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                animations[state] = json.load(f)
        except FileNotFoundError:
            animations[state] = None
    return animations

def render_avatar(state: str, animations: dict, height: int = 520):
    anim = animations.get(state) or animations.get("idle")
    if anim is None:
        st.warning(f"⚠️ No se encontró {state}.json en ./assets/")
        return
    anim_json = json.dumps(anim)
    st.components.v1.html(f"""
        <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
        <div style="display:flex;justify-content:center;align-items:center;width:100%;height:{height}px;">
            <lottie-player autoplay loop mode="normal" speed="0.3")
                style="width:100%;height:{height}px;background:transparent;"
                src='data:application/json,{anim_json}'>
            </lottie-player>
        </div>
    """, height=height)

# ─── Audio player SIN autoplay ───────────────────────────────────────────────

def get_audio_duration(mp3_bytes: bytes) -> float:
    """Calcula la duración exacta del MP3 en segundos usando miniaudio."""
    try:
        info = miniaudio.mp3_get_info(mp3_bytes)
        return info.duration
    except Exception:
        return len(mp3_bytes) / 16000.0

def audio_player(audio_bytes: bytes):
    """
    Muestra el audio SIN autoplay. El usuario hace clic en play manualmente.
    El estado speaking → idle se maneja por timer en Python (ver schedule_idle).
    """
    b64 = base64.b64encode(audio_bytes).decode()
    st.components.v1.html(f"""
        <!DOCTYPE html>
        <html>
        <body style="margin:0;padding:4px;background:transparent;">
        <audio id="tts"
               controls
               style="width:100%;height:40px;">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        </body></html>
    """, height=55)

# ─── Timer para volver a idle ─────────────────────────────────────────────────

def schedule_idle(duration_seconds: float):
    """
    Lanza un hilo que espera `duration_seconds` y luego activa la bandera
    `go_idle` en session_state. El próximo rerun de Streamlit la detecta
    y cambia el avatar a idle.

    Usamos st.session_state directamente desde el hilo; en Streamlit >= 1.27
    esto es thread-safe para escrituras simples.
    """
    def _worker():
        # Añadir un pequeño margen para que el usuario alcance a escuchar
        # el final del audio antes de que el avatar cambie de estado.
        time.sleep(duration_seconds + 0.8)
        st.session_state["go_idle"] = True

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

# ─── Inicialización ───────────────────────────────────────────────────────────

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
    with st.spinner(f"Cargando modelo de voz ({WHISPER_MODEL})..."):
        return WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")

# ─── STT ──────────────────────────────────────────────────────────────────────

def transcribe_audio(audio_bytes: bytes, whisper_model) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    segments, _ = whisper_model.transcribe(tmp_path, language="es", beam_size=5, vad_filter=True)
    os.unlink(tmp_path)
    return " ".join(seg.text for seg in segments).strip()

# ─── TTS ──────────────────────────────────────────────────────────────────────

def clean_for_tts(text: str) -> str:
    clean = re.sub(r"\*+|_+|#{1,6}\s?|`+", "", text)
    clean = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", clean)
    clean = re.sub(r"[^\w\s\.\,\;\:\!\?\-\(\)áéíóúüñÁÉÍÓÚÜÑ]", "", clean)
    return re.sub(r"\s+", " ", clean).strip()

async def _tts_async(text: str, voice: str) -> bytes:
    communicate = edge_tts.Communicate(text, voice)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    return buf.getvalue()

def text_to_speech(text: str, voice: str = VOICE) -> bytes:
    clean = clean_for_tts(text)
    if not clean:
        return b""
    return asyncio.run(_tts_async(clean, voice))

# ─── RAG ──────────────────────────────────────────────────────────────────────

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
    docs, metas, distances = results["documents"][0], results["metadatas"][0], results["distances"][0]
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
        sources.append({"title": title, "source": meta.get("source", ""), "similarity": similarity})
    return "\n\n---\n\n".join(context_parts), sources

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
        user_message = f"No se encontró contexto relevante en el material del curso.\n\nPregunta del estudiante: {query}"
    response = st.session_state.chat.send_message(user_message)
    return response.text, sources

# ─── Historial HTML ───────────────────────────────────────────────────────────

def render_messages_html(messages: list) -> str:
    rows = []
    for msg in messages:
        is_user = msg["role"] == "user"
        bg = "#1e3a5f" if is_user else "#1a1a2e"
        align = "flex-end" if is_user else "flex-start"
        icon = "🧑" if is_user else "🎓"
        content = msg["content"].replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        sources_html = ""
        if msg.get("sources"):
            links = "".join(
                f'<a href="{s["source"]}" target="_blank" style="color:#4da6ff;font-size:0.78rem;">'
                f'{s["title"]} ({s["similarity"]:.0%})</a><br>'
                for s in msg["sources"]
            )
            sources_html = f'<details style="margin-top:0.4rem;font-size:0.8rem;color:#aaa;"><summary>📎 Fuentes</summary>{links}</details>'
        rows.append(f"""
        <div style="display:flex;justify-content:{align};margin-bottom:0.75rem;">
          <div style="max-width:85%;background:{bg};border-radius:12px;padding:0.6rem 0.9rem;">
            <span style="font-size:0.75rem;color:#888;">{icon}</span>
            <div style="margin-top:0.2rem;font-size:0.9rem;line-height:1.5;">{content}</div>
            {sources_html}
          </div>
        </div>""")
    return "".join(rows)

# ─── UI ───────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Asistente Virtual UCR",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(CSS, unsafe_allow_html=True)

    genai_client, collection = init_clients()
    whisper_model = init_whisper()
    animations = load_animations()

    # ── Session state ──
    if "chat" not in st.session_state:
        st.session_state.chat = genai_client.chats.create(
            model=CHAT_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.3,
                max_output_tokens=4096,
            )
        )
    if "messages"      not in st.session_state: st.session_state.messages      = []
    if "avatar_state"  not in st.session_state: st.session_state.avatar_state  = "idle"
    if "pending_audio" not in st.session_state: st.session_state.pending_audio = None
    if "go_idle"       not in st.session_state: st.session_state.go_idle       = False

    # ── Detectar señal del timer: volver a idle ──
    if st.session_state.go_idle:
        st.session_state.go_idle       = False
        st.session_state.avatar_state  = "idle"
        # No hacemos rerun aquí; el próximo ciclo natural de Streamlit
        # ya refleja el cambio. Si querés que sea inmediato:
        st.rerun()

    # ── Layout ──
    col_avatar, col_chat = st.columns([1, 1.2], gap="large")

    with col_avatar:
        st.markdown("#### 🎓 Asistente Virtual UCR")
        render_avatar(st.session_state.avatar_state, animations, height=520)
        state_labels = {
            "idle":     "💤 Esperando",
            "thinking": "🤔 Pensando...",
            "speaking": "🗣 Hablando..."
        }
        st.caption(state_labels.get(st.session_state.avatar_state, ""))

    with col_chat:

        # ── Reproducir audio pendiente (sin autoplay) ──
        if st.session_state.pending_audio:
            audio_player(st.session_state.pending_audio)
            st.session_state.pending_audio = None

        # ── Historial ──
        history_html = f"""
        <html><head><style>
          body {{ margin:0; padding:0.5rem 0.5rem 1rem 0.5rem;
                 background:transparent; color:#fafafa;
                 font-family:-apple-system,sans-serif; overflow-x:hidden; }}
          ::-webkit-scrollbar {{ width:4px; }}
          ::-webkit-scrollbar-thumb {{ background:#444; border-radius:4px; }}
        </style></head>
        <body>
          {render_messages_html(st.session_state.messages)}
          <div id="bottom"></div>
          <script>document.getElementById('bottom').scrollIntoView();</script>
        </body></html>
        """
        st.components.v1.html(history_html, height=400, scrolling=True)

        # ── Input bar ──
        st.markdown('<div class="input-bar-wrapper">', unsafe_allow_html=True)
        audio_input = st.audio_input("🎤", label_visibility="collapsed")
        text_input  = st.chat_input("Escribí tu pregunta...")
        st.markdown('</div>', unsafe_allow_html=True)

        # ── STT ──
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

        if text_input:
            prompt = text_input

        # ── Nuevo prompt → thinking ──
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.avatar_state = "thinking"
            st.rerun()

        # ── Thinking → generar respuesta + TTS ──
        if st.session_state.avatar_state == "thinking" and \
           st.session_state.messages and \
           st.session_state.messages[-1]["role"] == "user":

            with st.spinner("Buscando en el material..."):
                response, sources = chat_with_rag(
                    st.session_state.messages[-1]["content"],
                    genai_client, collection
                )
            st.session_state.messages.append({
                "role": "model",
                "content": response,
                "sources": sources,
            })

            with st.spinner("Generando audio..."):
                audio_bytes = text_to_speech(response, voice=VOICE)

            # Calcular duración y programar retorno a idle
            if audio_bytes:
                duration = get_audio_duration(audio_bytes)
                schedule_idle(duration)

            st.session_state.pending_audio = audio_bytes
            st.session_state.avatar_state  = "speaking"
            st.rerun()


if __name__ == "__main__":
    main()