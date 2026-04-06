import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

personality = f"""
Quiero que actues como un amigo cercano, utiliza lenguaje relajado y amigable.
Trate de responder de forma natural, como en una conversación real. Puedes 
bromear ligeramente y tratar de ser empático.
"""

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-3-flash-preview", system_instruction=personality)

chat = model.start_chat()

print("Chat iniciado, escriba 'q' para salir\n")

while True:
    user_input = input("Usuario: ")
    if user_input.lower() == "q":
        break

    response = chat.send_message(user_input)

    print("AmigoGPT:", response.text)
