"""
Agente Virtual con RAG - Streamlit + ChromaDB + Gemini
=======================================================
Uso:
    pip install streamlit chromadb google-genai python-dotenv
    streamlit run chatbot.py
"""

import os
import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types
import streamlit as st

# ─── Configuración ────────────────────────────────────────────────────────────

load_dotenv()

CHROMA_PATH      = "./chroma_db"
COLLECTION_NAME  = "rag_presentaciones"
EMBED_MODEL      = "models/gemini-embedding-001"
CHAT_MODEL       = "gemini-2.5-flash"
TOP_K            = 5       # chunks a recuperar por query
MIN_SIMILARITY   = 0.4     # umbral mínimo de relevancia (0-1)

SYSTEM_PROMPT = """Eres un asistente virtual educativo de la Universidad de Costa Rica (UCR), especializado en el material de los módulos de virtualidad y educación en línea.

Tu comportamiento:
1. Respondé SIEMPRE basándote primero en el contexto recuperado del material de los módulos.
2. Si el contexto es suficiente, respondé con esa información y citá la fuente (nombre del slide).
3. Si el contexto no es suficiente o no es relevante, podés usar tu conocimiento general pero avisalo claramente.
4. Respondé en español, de forma clara y amigable.
5. Si te preguntan algo fuera del ámbito educativo/UCR, podés responder brevemente pero redirigí la conversación al material del curso.

Formato de respuesta cuando usás el RAG:
- Respondé la pregunta directamente
- Al final agregá: "📚 Fuente: [Hipervínuclo de la presentación]" para cada fuente usada
"""

# ─── Inicialización (cacheada para no repetir en cada rerun) ──────────────────

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

# ─── RAG: recuperar contexto relevante ────────────────────────────────────────

def retrieve_context(query: str, genai_client, collection) -> tuple[str, list[dict]]:
    """
    Embeddea la query, busca en ChromaDB y retorna:
    - context_text: string con el contexto formateado para el prompt
    - sources: lista de metadatos de los chunks recuperados
    """
    # Embed la query
    response = genai_client.models.embed_content(
        model=EMBED_MODEL,
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    query_embedding = response.embeddings[0].values

    # Buscar en ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    # Filtrar por similitud mínima
    relevant = [
        (doc, meta, 1 - dist)
        for doc, meta, dist in zip(docs, metas, distances)
        if (1 - dist) >= MIN_SIMILARITY
    ]

    if not relevant:
        return "", []

    # Formatear contexto para el prompt
    context_parts = []
    sources = []
    for doc, meta, similarity in relevant:
        title = meta.get("enriched_title", meta.get("slide_name", "Sin título"))
        context_parts.append(f"[{title}]\n{doc}")
        sources.append({
            "title": title,
            "source": meta.get("source", ""),
            "similarity": similarity,
        })

    context_text = "\n\n---\n\n".join(context_parts)
    return context_text, sources


# ─── Chat con Gemini ──────────────────────────────────────────────────────────

def chat_with_rag(query: str, history: list, genai_client, collection) -> tuple[str, list[dict]]:
    """
    Genera una respuesta usando RAG + historial de conversación.
    Retorna (respuesta, fuentes_usadas)
    """
    # Recuperar contexto relevante
    context, sources = retrieve_context(query, genai_client, collection)

    # Construir el mensaje con contexto
    if context:
        user_message = f"""Contexto recuperado del material del curso:
{context}

---
Pregunta del estudiante: {query}"""
    else:
        user_message = f"""No se encontró contexto relevante en el material del curso.

Pregunta del estudiante: {query}"""

    # Construir historial en formato Gemini
    gemini_history = []
    for msg in history:
        gemini_history.append(
            types.Content(role=msg["role"], parts=[types.Part(text=msg["content"])])
        )

    # Llamar a Gemini con historial
    chat = genai_client.chats.create(
        model=CHAT_MODEL,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.3,    # bajo para respuestas más precisas y consistentes
            max_output_tokens=1024,
        ),
        history=gemini_history,
    )

    response = chat.send_message(user_message)
    return response.text, sources


# ─── UI Streamlit ─────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Asistente Virtual UCR",
        page_icon="🎓",
        layout="centered"
    )

    st.title("🎓 Asistente Virtual UCR")
    st.caption("Consultá sobre los módulos de virtualidad y educación en línea")

    # Inicializar clientes
    genai_client, collection = init_clients()

    # Mostrar info de la base de conocimiento
    with st.sidebar:
        st.header("📚 Base de conocimiento")
        total = collection.count()
        st.metric("Chunks indexados", total)
        st.divider()
        st.markdown("**Configuración**")
        top_k = st.slider("Chunks a recuperar", 1, 10, TOP_K)
        min_sim = st.slider("Similitud mínima", 0.0, 1.0, MIN_SIMILARITY, 0.05)
        st.divider()
        if st.button("🗑️ Limpiar conversación"):
            st.session_state.messages = []
            st.rerun()

    # Inicializar historial
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Mostrar historial
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 Fuentes consultadas"):
                    for src in msg["sources"]:
                        st.markdown(f"- **{src['title']}** `{src['similarity']:.0%}` — [ver presentación]({src['source']})")

    # Input del usuario
    if prompt := st.chat_input("¿Qué querés saber sobre el material?"):
        # Mostrar mensaje del usuario
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generar respuesta
        with st.chat_message("assistant"):
            with st.spinner("Buscando en el material..."):
                # Pasar historial sin el mensaje actual
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]
                response, sources = chat_with_rag(
                    query=prompt,
                    history=history,
                    genai_client=genai_client,
                    collection=collection,
                )

            st.markdown(response)
            if sources:
                with st.expander("📎 Fuentes consultadas"):
                    for src in sources:
                        st.markdown(f"- **{src['title']}** `{src['similarity']:.0%}` — [ver presentación]({src['source']})")

        # Guardar respuesta en historial
        st.session_state.messages.append({
            "role": "model",
            "content": response,
            "sources": sources,
        })


if __name__ == "__main__":
    main()