from google import genai
from google.genai.errors import ClientError

from config import (
    GEMINI_API_KEY,
    MODEL
)

client = genai.Client(
    api_key=GEMINI_API_KEY
)

class RateLimitError(Exception):
    pass


def ask_gemini(prompt: str):

    try:

        response = client.models.generate_content(
            model=MODEL,
            contents=prompt
        )

        return response.text


    except ClientError as e:

        # Convertir a string seguro
        error_str = str(e)

        # Detectar rate limit
        if "RESOURCE_EXHAUSTED" in error_str or "429" in error_str:

            raise RateLimitError(
                "RATE_LIMIT_EXCEEDED"
            )

        # Otros errores
        raise Exception(
            f"Gemini error: {e}"
        )