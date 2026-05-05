# Agente Virtual Multimodal para Atención al Cliente

## Descripción

Este proyecto propone el diseño y evaluación de un **agente virtual multimodal** para atención al cliente, enfocado en mejorar la experiencia del usuario mediante:

- Interacción conversacional natural (texto y voz)
- Integración de conocimiento mediante Retrieval-Augmented Generation (RAG)
- Representación visual del agente (*embodiment*)
- Uso de modelos de lenguaje avanzados (Gemini)

El objetivo es analizar cómo la combinación de estas tecnologías impacta la **usabilidad, satisfacción del usuario y eficiencia en la resolución de consultas**, en comparación con chatbots tradicionales.

---

## Objetivos

- Diseñar un agente virtual con capacidades multimodales
- Integrar RAG con datos reales (tickets históricos, FAQs)
- Implementar interacción por voz (STT y TTS)
- Incorporar un avatar que represente al agente
- Evaluar el sistema en términos de experiencia de usuario y desempeño

---


### Componentes principales:

- **Frontend**: Interfaz web con chat + voz + avatar
- **Backend**: Orquestador del agente (manejo de contexto y lógica)
- **RAG**: Base de conocimiento + base de datos vectorial
- **LLM**: Gemini
- **Voz**: Speech-to-Text (STT) y Text-to-Speech (TTS)

---

## Estructura del repositorio
```
├── docs/
│ └── Entrega_1_Esteban_Alonso_Leon_Rodriguez.pdf
│
├── src/
│ ├── frontend/
│ ├── backend/
│ ├── rag/
│ └── voice/
│
├── data/
│ ├── raw/
│ └── processed/
│
├── tests/ # Pruebas del sistema
│
├── .env
├── requirements.txt
├── README.md
└── .gitignore
```
