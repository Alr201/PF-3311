# Agente Virtual Mediación Virtual — RAG + Avatar 3D

Agente virtual educativo desarrollado para la Universidad de Costa Rica (UCR) como parte del curso PF-3311. Combina un pipeline RAG sobre material de presentaciones Genially con un avatar 3D animado en Unity, interfaz web en Streamlit, síntesis de voz con Edge-TTS y transcripción local con Whisper.

---

## Características

- **RAG** sobre 28 presentaciones Genially (538 chunks indexados)
- **Enriquecimiento semántico** de chunks con Gemini (título, resumen, keywords)
- **Embeddings** con `gemini-embedding-001`
- **Chat** con `gemini-2.5-flash` con historial de conversación
- **STT** local con `faster-whisper` (sin API externa)
- **TTS** con `edge-tts` — voces neurales de Microsoft en español
- **Avatar "D** Diseñado en Figma y LottieFiles
- **Interfaz web** en Streamlit con avatar embebido via iframe

---

##  Estructura del proyecto

```
docs/
├── Entrega_1_Esteban_Alonso_Leon_Rodriguez.pdf
├── Guia_Procedimiento_Apartado_C.html
├── Reporte Tarea 10 - Esteban Alonso León Rodríguez
├── Entrega_2_Esteban_Alonso_Leon_Rodriguez.pdf
├── Entrega_3_Esteban_Alonso_Leon_Rodriguez.pdf
├── Entrega_3_Esteban_Alonso_Leon_Rodriguez.tex

│
src/
│
├── assets/
│   ├── greetings.json
│   ├── idle.json
│   ├── speaking.json
│   ├── thinkking.json
│
├── enrich_and_chunk.py     # Paso 1: enriquece JSONs con Gemini y genera chunks
├── index_to_chroma.py      # Paso 2: indexa chunks en ChromaDB con embeddings
├── chatbot.py              # Interfaz Streamlit: chat + STT + TTS + lipsync
├── lipsync_server.py       # Servidor WebSocket: analiza audio y anima el avatar
├── serve_unity.py          # Servidor HTTP: sirve el build WebGL de Unity
├── LipsyncClient.cs        # Script Unity C#: recibe blendshapes y anima la cara
│
├── raw_json/               # JSONs crudos extraídos de Genially (input)
├── output/
│   ├── enriched/           # JSONs enriquecidos por Gemini
│   ├── chunks/             # Chunks por presentación
│   └── all_chunks.json     # Todos los chunks combinados (input para indexado)
├── chroma_db/              # Base de datos vectorial ChromaDB (generada)
├── unity_build/            # Build WebGL de Unity (generado desde Unity)

```

---

##  Requisitos

### Python
- Python 3.10–3.14

### Dependencias Python
```bash
pip install google-genai python-dotenv
pip install chromadb
pip install streamlit
pip install faster-whisper
pip install edge-tts
pip install websockets numpy miniaudio
```

### API Keys
Crear archivo `.env` en `src/`:
```env
GEMINI_API_KEY=tu_api_key_aqui
```

---

## Setup completo (primera vez)

> ⚠️ La base de datos de Chroma ya está en el repositorio al hacer pull. Sólo es necesario seguir esta sección si se quiere correr todo desde el principio o reindexar la base de datos, de lo contrario saltar a la siguiente sección!!!

### Paso 1 — Enriquecer los JSONs con Gemini

```bash
python enrich_and_chunk.py --input ./raw_json --output ./output
```

Genera los archivos enriquecidos en `output/enriched/` y los chunks en `output/all_chunks.json`.

### Paso 2 — Indexar en ChromaDB

```bash
python index_to_chroma.py --chunks ./output/all_chunks.json --db ./chroma_db
```

Verifica el resultado:
```bash
python index_to_chroma.py --db ./chroma_db --stats
```
---

## Uso diario y video demostrativo

Enlace a video demostrativo: [https://youtu.be/UDzAVQJQBAM](https://youtu.be/UDzAVQJQBAM)

Enlace a video demostrativo Entrega Final: [https://youtu.be/Nf2opgnrNMA](https://youtu.be/Nf2opgnrNMA)

```bash
# Navegar a la carpeta src/
cd src/

# Correr el chatbot con streamlit
streamlit.exe run .\chatbot.py

```

Luego abrir [http://localhost:8501](http://localhost:8501) en el browser.

---

## 🔍 Consultas de prueba desde CLI

```bash
# Búsqueda semántica
python index_to_chroma.py --db ./chroma_db --query "¿Qué es la virtualidad?"

# Filtrar por presentación específica
python index_to_chroma.py --db ./chroma_db --query "evaluación" --filter-genially 64482d4d6e89d00013091ff7

# Más resultados
python index_to_chroma.py --db ./chroma_db --query "principios pedagógicos" --top-k 10
```


## Arquitectura

```
Usuario
  │
  ├─ Voz → faster-whisper (STT local) → texto
  │
  └─ Texto → Streamlit chatbot
                │
                ├─ embed query → gemini-embedding-001
                ├─ buscar en ChromaDB (top-K chunks)
                ├─ contexto + historial → gemini-2.5-flash
                │
                ├─ respuesta → edge-tts (TTS) → audio MP3
                │
                ├─ Avatar en 2D diseñado en Figma y LottieFiles
```

---

## Notas

- **ChromaDB**: los datos persisten en `chroma_db/`. Para reindexar desde cero, borrar esa carpeta.
- **Sin internet**: el STT (Whisper) funciona offline. El TTS (Edge-TTS) y Gemini requieren conexión.