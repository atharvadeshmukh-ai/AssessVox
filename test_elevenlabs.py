# /// script
# dependencies = ["websockets", "sounddevice", "python-dotenv", "numpy"]
# ///
"""
ElevenLabs scribe_v2_realtime — live microphone test.

Connects to the same WebSocket endpoint the bot uses, streams your mic audio,
and prints interim + final transcripts in real time.

Run:
    uv run test_elevenlabs.py

Press Ctrl+C to stop.
"""

import asyncio
import base64
import json
import os
import queue
import threading

import numpy as np
import sounddevice as sd
import websockets
from dotenv import load_dotenv

load_dotenv(override=True)

API_KEY     = os.getenv("ELEVENLABS_API_KEY", "")
SAMPLE_RATE = 16000
CHANNELS    = 1
CHUNK_MS    = 100                        # send 100 ms chunks to ElevenLabs
CHUNK_FRAMES = int(SAMPLE_RATE * CHUNK_MS / 1000)

WS_URL = (
    "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
    "?model_id=scribe_v2_realtime"
    f"&audio_format=pcm_{SAMPLE_RATE}"
    "&commit_strategy=vad"               # ElevenLabs detects speech/silence itself
)


def _mic_thread(audio_q: queue.Queue, stop_event: threading.Event):
    """Capture mic audio and push raw PCM bytes into the queue."""
    def callback(indata, frames, time, status):
        if status:
            print(f"[mic] {status}")
        pcm = (indata[:, 0] * 32767).astype(np.int16).tobytes()
        audio_q.put(pcm)

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=CHUNK_FRAMES,
        callback=callback,
    ):
        stop_event.wait()


async def run():
    if not API_KEY:
        print("ERROR: ELEVENLABS_API_KEY not set in .env")
        return

    print(f"API key  : {API_KEY[:8]}…{API_KEY[-4:]}")
    print(f"Endpoint : scribe_v2_realtime  (same as the bot)")
    print(f"Strategy : ElevenLabs VAD  (auto detects when you stop speaking)")
    print()
    print("Connecting…")

    headers = {"xi-api-key": API_KEY}
    audio_q: queue.Queue = queue.Queue()
    stop_event = threading.Event()

    mic = threading.Thread(target=_mic_thread, args=(audio_q, stop_event), daemon=True)
    mic.start()

    try:
        async with websockets.connect(WS_URL, additional_headers=headers) as ws:
            # Wait for session_started
            raw = await asyncio.wait_for(ws.recv(), timeout=8.0)
            msg = json.loads(raw)
            mtype = msg.get("message_type", "")

            if mtype == "unaccepted_terms":
                print("✗  ElevenLabs STT terms not accepted.")
                print("   → Go to elevenlabs.io and accept the Speech-to-Text API terms.")
                return
            elif mtype != "session_started":
                print(f"✗  Unexpected first message: {msg}")
                return

            print("✓  Connected — speak now (Ctrl+C to quit)\n")

            async def send_audio():
                loop = asyncio.get_event_loop()
                while True:
                    try:
                        pcm = await loop.run_in_executor(
                            None, lambda: audio_q.get(timeout=0.5)
                        )
                        payload = {
                            "message_type": "input_audio_chunk",
                            "audio_base_64": base64.b64encode(pcm).decode(),
                            "commit": False,
                            "sample_rate": SAMPLE_RATE,
                        }
                        await ws.send(json.dumps(payload))
                    except queue.Empty:
                        continue

            async def recv_transcripts():
                async for raw in ws:
                    msg = json.loads(raw)
                    mtype = msg.get("message_type", "")
                    if mtype == "partial_transcript":
                        text = msg.get("text", "").strip()
                        if text:
                            print(f"\r  [interim]  {text:<80}", end="", flush=True)
                    elif mtype in ("committed_transcript", "committed_transcript_with_timestamps"):
                        text = msg.get("text", "").strip()
                        print(f"\r  [FINAL]    {text:<80}")
                    elif "error" in mtype:
                        print(f"\n✗  Server error ({mtype}): {msg.get('error', msg)}")

            await asyncio.gather(send_audio(), recv_transcripts())

    except asyncio.TimeoutError:
        print("✗  Timed out waiting for session_started")
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"\n✗  Connection closed: {e}")
    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        stop_event.set()


if __name__ == "__main__":
    asyncio.run(run())
