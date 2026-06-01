import json
import re
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import time
import random
import whisper
import yt_dlp


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

URL_FILE = "genially_urls.txt"

RAW_DIR = Path("output/raw")
STRUCTURED_DIR = Path("output/structured")
RAG_DIR = Path("output/rag_docs")
TRANSCRIPT_DIR = Path("output/transcripts")
AUDIO_DIR = Path("output/audio")

AUDIO_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)
STRUCTURED_DIR.mkdir(parents=True, exist_ok=True)
RAG_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# RUIDO GENIALLY
# --------------------------------------------------

NOISE_EXACT = {
    "volver",
    "siguiente",
    "anterior",
    "inicio",
    "empezar",
    "enter",
    "cerrar",
    "continuar",
    "next",
    "back",
    "home",
    "play",
    "pause"
}

NOISE_PATTERNS = [

    r"^hagamos clic.*",
    r"^haz clic.*",
    r"^hacer clic.*",

    r"^conozcamos.*",
    r"^veamos.*",
    r"^avancemos.*",

    r"^para comenzar.*",
    r"^posicion[aá]ndonos.*",

    r"^haga clic.*",
    r"^haz click.*",

    r"^clic en.*",
    r"^click en.*",

    r"^reproducir video.*",
    r"^reproducir el video.*",

    r"^siguiente$",
    r"^anterior$",
    r"^volver$",
]


BAD_SLIDE_NAMES = {
    "copia",
    "copy",
    "libro",
    "shape",
    "texto",
    "text",
    "imagen",
    "image",
    "grupo",
    "group",
    "rectangulo",
    "rectangle"
}


# --------------------------------------------------
# UTILIDADES
# --------------------------------------------------

def get_genially_id(url):

    match = re.search(
        r"view\.genially\.com/([a-zA-Z0-9]+)",
        url
    )

    if not match:
        raise ValueError(
            f"No pude obtener ID de {url}"
        )

    return match.group(1)


def clean_html(html):

    if not html:
        return ""

    text = BeautifulSoup(
        html,
        "html.parser"
    ).get_text(
        " ",
        strip=True
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def clean_content(text):

    if not text:
        return ""

    fragments = re.split(
        r"\n|\.\s+",
        text
    )

    cleaned = []

    for fragment in fragments:

        fragment = fragment.strip()

        if not fragment:
            continue

        lower = fragment.lower()

        if lower in NOISE_EXACT:
            continue

        skip = False

        for pattern in NOISE_PATTERNS:

            if re.match(
                pattern,
                lower,
                flags=re.IGNORECASE
            ):
                skip = True
                break

        if skip:
            continue

        if len(fragment) <= 2:
            continue

        cleaned.append(fragment)

    result = ". ".join(cleaned)

    result = re.sub(
        r"\s+",
        " ",
        result
    )

    return result.strip()


def clean_slide_name(name):

    if not name:
        return ""

    lower = name.lower()

    for bad in BAD_SLIDE_NAMES:

        if bad in lower:
            return ""

    return name.strip()


def get_candidate_title(slide):

    if slide["slide_name"]:

        return slide["slide_name"]

    if slide["texts"]:

        first_text = slide["texts"][0]

        words = first_text.split()

        return " ".join(words[:8])

    return "Sin título"


def get_video_id(source):

    patterns = [
        r"youtu\.be/([A-Za-z0-9_-]{11})",
        r"v=([A-Za-z0-9_-]{11})",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            source
        )

        if match:
            return match.group(1)

    return None

def download_audio(video_id):

    audio_file = (
        AUDIO_DIR /
        f"{video_id}.mp3"
    )

    if audio_file.exists():

        print(
            f"📁 Audio cache {video_id}"
        )

        return str(audio_file)

    url = (
        f"https://www.youtube.com/watch?v={video_id}"
    )

    print(
        f"⬇ Descargando audio {video_id}"
    )

    ydl_opts = {
        "format": "bestaudio/best",

        "outtmpl":
            str(
                AUDIO_DIR /
                f"{video_id}.%(ext)s"
            ),

        "quiet": True,

        "noplaylist": True,

        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ]
    }

    with yt_dlp.YoutubeDL(
        ydl_opts
    ) as ydl:

        ydl.download([url])

    return str(audio_file)

def get_transcript(video_id):

    transcript_file = (
        TRANSCRIPT_DIR /
        f"{video_id}.txt"
    )

    # ------------------
    # CACHE
    # ------------------

    if transcript_file.exists():

        print(
            f"📁 Transcript cache {video_id}"
        )

        return transcript_file.read_text(
            encoding="utf-8"
        )

    try:

        audio_path = download_audio(
            video_id
        )

        print(
            f"🎙 Transcribiendo {video_id}"
        )

        result = (
            WHISPER_MODEL.transcribe(
                audio_path,
                language="es"
            )
        )

        text = clean_content(
            result["text"]
        )

        transcript_file.write_text(
            text,
            encoding="utf-8"
        )

        return text

    except Exception as e:

        print(
            f"⚠ Error transcribiendo "
            f"{video_id}: {e}"
        )

        return ""

# --------------------------------------------------
# PROCESAMIENTO
# --------------------------------------------------

def process_genially(url):

    genially_id = get_genially_id(url)

    print(
        f"\nProcesando {genially_id}"
    )

    api_url = (
        f"https://view.genially.com/api/view/{genially_id}"
    )

    data = requests.get(api_url).json()

    # ----------------------------------------
    # RAW JSON
    # ----------------------------------------

    raw_file = (
        RAW_DIR /
        f"{genially_id}.json"
    )

    with open(
        raw_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )

    # ----------------------------------------
    # Slides
    # ----------------------------------------

    slides = {}

    for slide in data["Slides"]:

        slides[slide["Id"]] = {

            "slide_id": slide["Id"],

            "slide_name":
                clean_slide_name(
                    slide["Name"]
                ),

            "texts": [],

            "videos": []
        }

    # ----------------------------------------
    # Textos
    # ----------------------------------------

    for text in data["Texts"]:

        slide_id = text["IdSlide"]

        if slide_id not in slides:
            continue

        content = clean_html(
            text.get(
                "TextMessage",
                ""
            )
        )

        content = clean_content(
            content
        )

        if content:

            slides[slide_id][
                "texts"
            ].append(
                content
            )

    # ----------------------------------------
    # Videos
    # ----------------------------------------

    for video in data["Videos"]:

        slide_id = video["IdSlide"]

        if slide_id not in slides:
            continue

        source = video.get(
            "Source",
            ""
        )

        video_id = get_video_id(
            source
        )

        if not video_id:
            continue

        transcript = get_transcript(
            video_id
        )

        slides[slide_id][
            "videos"
        ].append(
            {
                "video_id":
                    video_id,

                "source":
                    source,

                "transcript":
                    transcript
            }
        )

    # ----------------------------------------
    # Eliminar slides vacías
    # ----------------------------------------

    structured = []

    for slide in slides.values():

        if (
            not slide["texts"]
            and
            not slide["videos"]
        ):
            continue

        slide[
            "title_candidate"
        ] = get_candidate_title(
            slide
        )

        structured.append(
            slide
        )

    # ----------------------------------------
    # JSON estructurado
    # ----------------------------------------

    structured_file = (
        STRUCTURED_DIR /
        f"{genially_id}_structured.json"
    )

    with open(
        structured_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            structured,
            f,
            ensure_ascii=False,
            indent=2
        )

    # ----------------------------------------
    # Documentos RAG
    # ----------------------------------------

    rag_documents = []

    for slide in structured:

        text_parts = []

        text_parts.append(
            f"TITULO_CANDIDATO: "
            f"{slide['title_candidate']}"
        )

        if slide["texts"]:

            text_parts.append(
                "\nCONTENIDO:\n"
            )

            text_parts.append(
                "\n".join(
                    slide["texts"]
                )
            )

        for video in slide["videos"]:

            if video[
                "transcript"
            ]:

                text_parts.append(
                    "\nTRANSCRIPCION:\n"
                )

                text_parts.append(
                    video[
                        "transcript"
                    ]
                )

        document = "\n".join(
            text_parts
        )

        rag_documents.append(
            {
                "source": url,

                "genially_id":
                    genially_id,

                "slide_id":
                    slide["slide_id"],

                "slide_name":
                    slide["slide_name"],

                "title_candidate":
                    slide[
                        "title_candidate"
                    ],

                "content":
                    document
            }
        )

    rag_file = (
        RAG_DIR /
        f"{genially_id}_rag.json"
    )

    with open(
        rag_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            rag_documents,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"✓ {len(rag_documents)} documentos generados"
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

with open(
    URL_FILE,
    encoding="utf-8"
) as f:

    urls = [
        line.strip()
        for line in f
        if line.strip()
    ]

print("Cargando modelo Whisper...")

WHISPER_MODEL = whisper.load_model(
    "small"
)

for url in urls:

    try:

        process_genially(
            url
        )

    except Exception as e:

        print(
            f"\nERROR en {url}"
        )

        print(e)

print(
    "\nProceso finalizado."
)