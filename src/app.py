import streamlit as st
from llm import ask_gemini, RateLimitError

st.set_page_config(
    page_title="AVI de soporte a usuarios para Mediación Virtual",
    page_icon="🤖"
)

st.title("AVI de soporte a usuarios para Mediación Virtual")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):
        st.markdown(
            message["content"]
        )

prompt = st.chat_input(
    "Haz una pregunta..."
)

if prompt:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pensando... 🤖"):
            try:

                respuesta = ask_gemini(prompt)

            except RateLimitError:

                respuesta = (
                    "⚠️ Estoy saturado en este momento. "
                    "Intenta de nuevo en unos segundos."
                )

            except Exception:

                respuesta = (
                    "⚠️ Ocurrió un error inesperado."
                )

            st.markdown(respuesta)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": respuesta
        }
    )