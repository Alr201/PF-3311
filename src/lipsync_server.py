"""
Lipsync WebSocket Server
========================
Corre en paralelo con Streamlit. Recibe audio MP3, analiza la amplitud
frame por frame y transmite valores de blendshapes a Unity vía WebSocket.

Instalación:
    pip install websockets numpy pydub

Uso:
    python lipsync_server.py
    (correr antes de iniciar Streamlit)
"""

import asyncio
import json
import io
import numpy as np
import miniaudio
import websockets

HOST = "localhost"
PORT = 8765

# Clientes Unity conectados
connected_clients: set = set()

# ─── Análisis de audio ────────────────────────────────────────────────────────

def analyze_amplitude(mp3_bytes: bytes, fps: int = 30) -> list[dict]:
    """
    Decodifica MP3 con miniaudio y analiza amplitud frame a frame a {fps} FPS.
    Retorna lista de dicts con valores de blendshapes por frame.
    """
    decoded = miniaudio.decode(
        mp3_bytes,
        output_format=miniaudio.SampleFormat.FLOAT32,
        nchannels=1,        # mono
        sample_rate=16000,  # downsample para ser más rápido
    )
    samples = np.frombuffer(decoded.samples, dtype=np.float32)
    sample_rate = decoded.sample_rate
    frame_size = int(sample_rate / fps)
    frames = []

    for i in range(0, len(samples) - frame_size, frame_size):
        chunk = samples[i:i + frame_size]
        amplitude = float(np.sqrt(np.mean(chunk ** 2)))  # RMS
        frames.append(map_amplitude_to_blendshapes(amplitude))

    return frames


def map_amplitude_to_blendshapes(amplitude: float) -> dict:
    """
    Mapea amplitud RMS a valores de blendshapes de boca.
    Lipsync simulado: abre/cierra boca con variación natural.
    """
    # Normalizar amplitud a 0-1 (clamp)
    a = min(amplitude * 4.0, 1.0)   # factor 4 para que sea más sensible

    # Umbral de silencio
    if a < 0.05:
        return {
            "FCL_MTH_Close":   1.0,
            "FCL_MTH_Neutral": 0.0,
            "FCL_MTH_A":       0.0,
            "FCL_MTH_I":       0.0,
            "FCL_MTH_U":       0.0,
            "FCL_MTH_E":       0.0,
            "FCL_MTH_O":       0.0,
            "FCL_MTH_Large":   0.0,
            "FCL_MTH_Small":   0.0,
        }

    # Apertura base de la boca proporcional a la amplitud
    open_amount = a

    # Variación: alternar entre vocales simuladas para dar vida
    # En lipsync real esto vendría del fonema, acá lo simulamos con ruido suave
    noise = float(np.random.uniform(0, 0.3))

    return {
        "FCL_MTH_Close":   max(0.0, 1.0 - open_amount),
        "FCL_MTH_Neutral": 0.0,
        "FCL_MTH_A":       open_amount * 0.8 + noise * 0.2,
        "FCL_MTH_I":       open_amount * 0.3,
        "FCL_MTH_U":       open_amount * 0.2,
        "FCL_MTH_E":       open_amount * 0.3,
        "FCL_MTH_O":       open_amount * 0.5,
        "FCL_MTH_Large":   open_amount * 0.6,
        "FCL_MTH_Small":   max(0.0, 0.3 - open_amount * 0.3),
    }


# ─── WebSocket handlers ───────────────────────────────────────────────────────

async def handle_unity(websocket):
    """Maneja la conexión de Unity (recibe comandos, envía blendshapes)."""
    connected_clients.add(websocket)
    print(f"[+] Unity conectado ({len(connected_clients)} clientes)")
    try:
        async for message in websocket:
            # Unity puede enviar "ping" para mantener vivo el socket
            if message == "ping":
                await websocket.send("pong")
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        connected_clients.discard(websocket)
        print(f"[-] Unity desconectado ({len(connected_clients)} clientes)")


async def handle_streamlit(websocket):
    """
    Recibe audio MP3 desde Streamlit, analiza y broadcast a Unity.
    Streamlit se conecta a ws://localhost:8765/audio
    """
    print("[~] Streamlit enviando audio...")
    try:
        mp3_bytes = await websocket.recv()  # recibe el MP3 completo
        frames = analyze_amplitude(mp3_bytes)
        print(f"[~] {len(frames)} frames generados a 30fps")

        # Broadcast frame por frame a todos los clientes Unity
        if connected_clients:
            for i, frame in enumerate(frames):
                msg = json.dumps({"frame": i, "blendshapes": frame})
                await asyncio.gather(
                    *[client.send(msg) for client in connected_clients],
                    return_exceptions=True,
                )
                await asyncio.sleep(1 / 30)  # 30 FPS

            # Al terminar, mandar cierre de boca
            close_msg = json.dumps({"frame": -1, "blendshapes": map_amplitude_to_blendshapes(0.0)})
            await asyncio.gather(
                *[client.send(close_msg) for client in connected_clients],
                return_exceptions=True,
            )
        else:
            print("[!] No hay clientes Unity conectados")

    except Exception as e:
        print(f"[!] Error procesando audio: {e}")


async def router(websocket):
    """Enruta conexiones según el path."""
    path = websocket.request.path
    if path == "/audio":
        await handle_streamlit(websocket)
    else:
        await handle_unity(websocket)


async def main():
    print(f"🎙️  Lipsync server corriendo en ws://{HOST}:{PORT}")
    print(f"    Unity conecta a:    ws://{HOST}:{PORT}/")
    print(f"    Streamlit envía a:  ws://{HOST}:{PORT}/audio")
    async with websockets.serve(router, HOST, PORT):
        await asyncio.Future()  # correr indefinidamente


if __name__ == "__main__":
    asyncio.run(main())
