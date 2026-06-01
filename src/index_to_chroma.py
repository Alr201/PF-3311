"""
RAG Pipeline: Indexar chunks en ChromaDB con Google Embeddings
===============================================================
Requiere haber corrido primero enrich_and_chunk.py

Uso:
    pip install chromadb google-generativeai
    export GEMINI_API_KEY="tu_api_key"

    # Indexar todos los chunks
    python index_to_chroma.py --chunks ./output/all_chunks.json --db ./chroma_db

    # Consultar (varios modos)
    python index_to_chroma.py --db ./chroma_db --query "¿Qué es la virtualidad?"
    python index_to_chroma.py --db ./chroma_db --query "principios pedagógicos" --filter-genially 64482d4d6e89d00013091ff7
    python index_to_chroma.py --db ./chroma_db --query "evaluación" --top-k 10
"""

import os
from dotenv import load_dotenv
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Optional
import chromadb
from chromadb.config import Settings
from google import genai
from google.genai import types

# ─── Configuración ────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

COLLECTION_NAME = "rag_presentaciones"
EMBED_MODEL     = "models/gemini-embedding-001"
EMBED_BATCH     = 20      # Google permite hasta 100, pero 20 es seguro para textos largos
EMBED_DELAY     = 1.0     # segundos entre batches (evita rate limiting)

load_dotenv()

# Después de load_dotenv(), agregá esto:
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise EnvironmentError("Falta la variable de entorno GEMINI_API_KEY")
genai_client = genai.Client(api_key=api_key)

# ─── Embeddings ───────────────────────────────────────────────────────────────

def embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """
    Genera embeddings con Google gemini-embedding-001.
    task_type:
      - "RETRIEVAL_DOCUMENT"  → para indexar chunks
      - "RETRIEVAL_QUERY"     → para consultas
    """
    all_embeddings = []
    for i, text in enumerate(texts):
        if i % 10 == 0:
            log.info(f"  Embedding {i+1}/{len(texts)}...")

        response = genai_client.models.embed_content(
            model=EMBED_MODEL,
            contents=text,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        all_embeddings.append(response.embeddings[0].values)

        time.sleep(0.1)

    return all_embeddings


# ─── Indexado ─────────────────────────────────────────────────────────────────

def index_chunks(chunks_path: Path, db_path: Path):
    """Carga all_chunks.json, genera embeddings y los guarda en ChromaDB."""

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    log.info(f"\n{'='*50}")
    log.info(f"📦  Total chunks a indexar: {len(chunks)}")
    log.info(f"🗄️   ChromaDB path: {db_path}")
    log.info(f"{'='*50}\n")

    # Inicializar ChromaDB (persiste en disco)
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}   # distancia coseno es mejor para texto
    )

    # Ver si ya hay datos indexados
    existing = collection.count()
    if existing > 0:
        log.info(f"⚠️  La colección ya tiene {existing} chunks. Saltando los que ya existen...")

    # Preparar datos
    texts      = [c["chunk_text"] for c in chunks]
    ids        = [f"{c['metadata']['genially_id']}_{c['metadata']['slide_id']}" for c in chunks]
    metadatas  = [c["metadata"] for c in chunks]

    # ChromaDB no acepta listas en metadata → convertir keywords a string
    for m in metadatas:
        if isinstance(m.get("keywords"), list):
            m["keywords"] = ", ".join(m["keywords"])

    # Filtrar los que ya están indexados
    existing_ids = set(collection.get(ids=ids)["ids"]) if existing > 0 else set()
    new_indices  = [i for i, id_ in enumerate(ids) if id_ not in existing_ids]

    if not new_indices:
        log.info("✅  Todos los chunks ya estaban indexados.")
        return

    log.info(f"🆕  Chunks nuevos a indexar: {len(new_indices)}")

    new_texts     = [texts[i]     for i in new_indices]
    new_ids       = [ids[i]       for i in new_indices]
    new_metadatas = [metadatas[i] for i in new_indices]

    # Generar embeddings
    log.info(f"\n🔢  Generando embeddings con {EMBED_MODEL}...")
    embeddings = embed_texts(new_texts, task_type="RETRIEVAL_DOCUMENT")

    # Insertar en ChromaDB
    log.info(f"\n💾  Insertando en ChromaDB...")
    collection.add(
        ids=new_ids,
        embeddings=embeddings,
        documents=new_texts,
        metadatas=new_metadatas,
    )

    log.info(f"\n✅  Indexado completo. Total en colección: {collection.count()}")


# ─── Consultas ────────────────────────────────────────────────────────────────

def query_collection(
    db_path: Path,
    query_text: str,
    top_k: int = 5,
    filter_genially: Optional[str] = None,
    filter_keyword: Optional[str] = None,
):
    """
    Consulta la colección ChromaDB.

    Modos:
    - Solo semántico: sin filtros
    - Filtrar por presentación: --filter-genially <genially_id>
    - Filtrar por keyword exacto: --filter-keyword <palabra>
    """
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_collection(name=COLLECTION_NAME)

    log.info(f"\n🔍  Query: '{query_text}'")
    log.info(f"    Top-K: {top_k}")

    # Construir filtros (where clause de ChromaDB)
    where = None
    if filter_genially and filter_keyword:
        where = {
            "$and": [
                {"genially_id": {"$eq": filter_genially}},
                {"keywords": {"$contains": filter_keyword}},
            ]
        }
    elif filter_genially:
        where = {"genially_id": {"$eq": filter_genially}}
        log.info(f"    Filtro presentación: {filter_genially}")
    elif filter_keyword:
        where = {"keywords": {"$contains": filter_keyword}}
        log.info(f"    Filtro keyword: {filter_keyword}")

    # Generar embedding de la query
    query_embedding = embed_texts([query_text], task_type="RETRIEVAL_QUERY")[0]

    # Consultar
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    # Mostrar resultados
    print(f"\n{'='*60}")
    print(f"Resultados para: \"{query_text}\"")
    print(f"{'='*60}")

    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), 1):
        similarity = 1 - dist   # coseno: distancia → similitud
        print(f"\n[{i}] Similitud: {similarity:.3f}")
        print(f"    Título:  {meta.get('enriched_title', 'N/A')}")
        print(f"    Fuente:  {meta.get('source', 'N/A')}")
        print(f"    Slide:   {meta.get('slide_name', 'N/A')}")
        print(f"    Summary: {meta.get('summary', '')[:150]}...")
        print(f"    Keywords:{meta.get('keywords', '')}")
        print(f"    Chunk preview: {doc[:200]}...")

    return results


# ─── Utilidades ───────────────────────────────────────────────────────────────

def show_stats(db_path: Path):
    """Muestra estadísticas de la colección."""
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_collection(name=COLLECTION_NAME)

    total = collection.count()
    sample = collection.get(limit=5, include=["metadatas"])

    print(f"\n📊  Estadísticas de la colección '{COLLECTION_NAME}'")
    print(f"    Total chunks: {total}")
    print(f"\n    Muestra de metadatos:")
    for m in sample["metadatas"]:
        print(f"      - [{m.get('genially_id', '')[:8]}...] {m.get('enriched_title', 'N/A')}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ChromaDB indexer + query para RAG")
    parser.add_argument("--db", required=True, help="Carpeta donde se guarda/lee ChromaDB")

    # Modo indexado
    parser.add_argument("--chunks", help="Path al all_chunks.json generado por enrich_and_chunk.py")

    # Modo consulta
    parser.add_argument("--query", help="Texto a consultar")
    parser.add_argument("--top-k", type=int, default=5, help="Cantidad de resultados (default: 5)")
    parser.add_argument("--filter-genially", help="Filtrar por genially_id específico")
    parser.add_argument("--filter-keyword", help="Filtrar chunks que contengan este keyword")

    # Stats
    parser.add_argument("--stats", action="store_true", help="Mostrar estadísticas de la colección")

    args = parser.parse_args()

    db_path = Path(args.db)

    if args.chunks:
        index_chunks(Path(args.chunks), db_path)

    if args.query:
        query_collection(
            db_path=db_path,
            query_text=args.query,
            top_k=args.top_k,
            filter_genially=args.filter_genially,
            filter_keyword=args.filter_keyword,
        )

    if args.stats:
        show_stats(db_path)

    if not args.chunks and not args.query and not args.stats:
        parser.print_help()


if __name__ == "__main__":
    main()