"""
RAG Pipeline: Enrich + Chunk
=============================
Paso 1: Enriquece cada archivo JSON con Gemini (enriched_title, summary, keywords)
Paso 2: Genera chunks listos para embeddings

Uso:
    pip install google-generativeai
    export GEMINI_API_KEY="tu_api_key"
    python enrich_and_chunk.py --input ./rag_docs --output ./cloutput

Estructura esperada de input:
    ./raw_json/
        presentacion_01.json
        presentacion_02.json
        ...
"""

import os
import json
import time
import argparse
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
import google.genai as genai

from config import (
    GEMINI_API_KEY,
    MODEL
)

# ─── Configuración ────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

GEMINI_MODEL = MODEL  # Flash es más rápido y barato; cámbialo a gemini-1.5-pro si necesitas más calidad
MAX_WORKERS = 4                      # Requests en paralelo (ajusta según tu cuota RPM)
RETRY_ATTEMPTS = 3
RETRY_DELAY = 10                     # segundos entre reintentos

ENRICH_PROMPT = """Eres un asistente especializado en procesamiento de contenido educativo para sistemas RAG (Retrieval-Augmented Generation).

Recibirás un array JSON con diapositivas extraídas de una presentación educativa. Cada objeto tiene los campos: source, genially_id, slide_id, slide_name, title_candidate y content.

Tu tarea es enriquecer cada objeto con los siguientes campos adicionales:

- "enriched_title": Un título claro, descriptivo y conciso (máximo 10 palabras) que refleje fielmente el contenido real de la diapositiva. Si el title_candidate ya es adecuado, puedes refinarlo. Si es genérico ("Copia", "Libro", "Slide 1", "Portada", etc.), reemplázalo completamente basándote en el contenido.

- "summary": Un resumen breve del contenido (2-4 oraciones). Debe capturar la idea central de forma útil para recuperación semántica.

- "keywords": Lista de 5 a 10 palabras clave o frases cortas relevantes para búsqueda. Prioriza términos específicos del dominio, conceptos clave y entidades nombradas.

REGLAS IMPORTANTES:
1. No modifiques ningún campo existente. Solo agrega los tres campos nuevos.
2. Si el contenido de una diapositiva está vacío o es insignificante (ej: solo un número de página o texto como "Copia Copia"), usa enriched_title: "Diapositiva sin contenido relevante", summary: "" y keywords: [].
3. Responde ÚNICAMENTE con el array JSON enriquecido. Sin explicaciones, sin bloques de código markdown, sin texto adicional antes o después.
4. Mantén el mismo idioma que el contenido original.

Array JSON a procesar:
"""

# ─── Gemini: enriquecimiento ───────────────────────────────────────────────────

def enrich_file(json_path: Path, output_dir: Path) -> Optional[Path]:
    """Enriquece un archivo JSON con Gemini. Retorna el path del output o None si falla."""
    out_path = output_dir / "enriched" / json_path.name

    # Si ya fue procesado, skip
    if out_path.exists():
        log.info(f"  ⏭  Ya procesado: {json_path.name}")
        return out_path

    with open(json_path, "r", encoding="utf-8") as f:
        slides = json.load(f)

    log.info(f"  🔄  Procesando: {json_path.name} ({len(slides)} slides)")

    prompt = ENRICH_PROMPT + json.dumps(slides, ensure_ascii=False, indent=2)

    client = genai.Client(
        api_key=GEMINI_API_KEY
    )   

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt
            )
            raw = response.text.strip()

            # Limpiar posibles bloques markdown que Gemini a veces agrega
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw.rsplit("```", 1)[0]

            enriched = json.loads(raw)

            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(enriched, f, ensure_ascii=False, indent=2)

            log.info(f"  ✅  Guardado: {out_path.name}")
            return out_path

        except json.JSONDecodeError as e:
            log.warning(f"  ⚠️  JSON inválido en intento {attempt}: {e}")
        except Exception as e:
            log.warning(f"  ⚠️  Error en intento {attempt}: {e}")

        if attempt < RETRY_ATTEMPTS:
            log.info(f"     Reintentando en {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)

    log.error(f"  ❌  Falló después de {RETRY_ATTEMPTS} intentos: {json_path.name}")
    return None


def run_enrichment(input_dir: Path, output_dir: Path):
    """Procesa todos los JSON en paralelo con rate limiting."""
    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        log.error(f"No se encontraron archivos JSON en {input_dir}")
        return

    log.info(f"\n{'='*50}")
    log.info(f"📂  Archivos encontrados: {len(json_files)}")
    log.info(f"🤖  Modelo: {GEMINI_MODEL}")
    log.info(f"⚡  Workers paralelos: {MAX_WORKERS}")
    log.info(f"{'='*50}\n")

    results = {"ok": [], "fail": []}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(enrich_file, f, output_dir): f
            for f in json_files
        }
        for future in as_completed(futures):
            src = futures[future]
            result = future.result()
            if result:
                results["ok"].append(src.name)
            else:
                results["fail"].append(src.name)

    log.info(f"\n{'='*50}")
    log.info(f"✅  Exitosos: {len(results['ok'])}/{len(json_files)}")
    if results["fail"]:
        log.warning(f"❌  Fallidos: {results['fail']}")
    log.info(f"{'='*50}\n")


# ─── Chunking ─────────────────────────────────────────────────────────────────

def build_chunk_text(slide: dict) -> str:
    """
    Construye el texto del chunk combinando campos enriquecidos + contenido original.
    Esta es la cadena que se va a embedear.
    """
    parts = []

    title = slide.get("enriched_title") or slide.get("title_candidate", "")
    if title:
        parts.append(f"Título: {title}")

    summary = slide.get("summary", "")
    if summary:
        parts.append(f"Resumen: {summary}")

    content = slide.get("content", "")
    if content:
        parts.append(content)

    keywords = slide.get("keywords", [])
    if keywords:
        parts.append(f"Palabras clave: {', '.join(keywords)}")

    return "\n\n".join(parts)


def chunk_file(enriched_path: Path, output_dir: Path):
    """Convierte un JSON enriquecido en chunks listos para embeddings."""
    with open(enriched_path, "r", encoding="utf-8") as f:
        slides = json.load(f)

    chunks = []
    for slide in slides:
        # Saltar diapositivas vacías
        if slide.get("enriched_title") == "Diapositiva sin contenido relevante":
            continue
        if not slide.get("content", "").strip():
            continue

        chunk = {
            # ── Texto a embedear ──────────────────────────────────────
            "chunk_text": build_chunk_text(slide),

            # ── Metadatos para filtrado y display ────────────────────
            "metadata": {
                "source": slide.get("source", ""),
                "genially_id": slide.get("genially_id", ""),
                "slide_id": slide.get("slide_id", ""),
                "slide_name": slide.get("slide_name", ""),
                "enriched_title": slide.get("enriched_title", ""),
                "summary": slide.get("summary", ""),
                "keywords": slide.get("keywords", []),
            }
        }
        chunks.append(chunk)

    out_path = output_dir / "chunks" / enriched_path.name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    log.info(f"  📦  Chunks generados: {len(chunks)} → {out_path.name}")
    return chunks


def run_chunking(output_dir: Path):
    """Genera chunks de todos los archivos enriquecidos."""
    enriched_dir = output_dir / "enriched"
    json_files = sorted(enriched_dir.glob("*.json"))

    if not json_files:
        log.error(f"No hay archivos enriquecidos en {enriched_dir}. Corre primero el enriquecimiento.")
        return

    log.info(f"\n{'='*50}")
    log.info(f"📦  Generando chunks de {len(json_files)} archivos...")
    log.info(f"{'='*50}\n")

    all_chunks = []
    for f in json_files:
        chunks = chunk_file(f, output_dir)
        all_chunks.extend(chunks)

    # También guardar un único archivo combinado con todos los chunks
    combined_path = output_dir / "all_chunks.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    log.info(f"\n✅  Total chunks: {len(all_chunks)}")
    log.info(f"💾  Archivo combinado: {combined_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    global GEMINI_MODEL, MAX_WORKERS
    parser = argparse.ArgumentParser(description="RAG Pipeline: Enrich + Chunk")
    parser.add_argument("--input", required=True, help="Carpeta con los JSON crudos")
    parser.add_argument("--output", required=True, help="Carpeta de output")
    parser.add_argument(
        "--step",
        choices=["enrich", "chunk", "all"],
        default="all",
        help="Paso a ejecutar: 'enrich', 'chunk', o 'all' (default)"
    )
    parser.add_argument("--model", default=GEMINI_MODEL, help=f"Modelo de Gemini (default: {GEMINI_MODEL})")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help=f"Workers paralelos (default: {MAX_WORKERS})")
    args = parser.parse_args()

    # Configurar Gemini
    api_key = GEMINI_API_KEY
    if not api_key:
        raise EnvironmentError("Falta la variable de entorno GEMINI_API_KEY")

    GEMINI_MODEL = args.model
    MAX_WORKERS = args.workers

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.step in ("enrich", "all"):
        run_enrichment(input_dir, output_dir)

    if args.step in ("chunk", "all"):
        run_chunking(output_dir)


if __name__ == "__main__":
    main()