"""
Token server for the interview app.
Generates LiveKit access tokens for frontend participants.

Run with:
    uv run uvicorn token_server:app --port 8080 --reload
"""

import os
import time
import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from livekit.api import AccessToken, VideoGrants

load_dotenv(".env.local")

app = FastAPI(title="Interview Agent Token Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Tighten this in production
    allow_methods=["GET"],
    allow_headers=["*"],
)

LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
LIVEKIT_URL = os.getenv("LIVEKIT_URL")


@app.get("/token")
def get_token(
    room: str = Query(default="interview-room", description="LiveKit room name"),
    username: str = Query(default="candidate", description="Participant identity"),
):
    """
    Returns a LiveKit access token and the server URL.
    The frontend uses these to connect to the interview room.
    """
    if not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
        raise HTTPException(status_code=500, detail="LiveKit credentials not configured.")

    token = (
        AccessToken(api_key=LIVEKIT_API_KEY, api_secret=LIVEKIT_API_SECRET)
        .with_identity(username)
        .with_name(username)
        .with_grants(
            VideoGrants(
                room_join=True,
                room=room,
                can_publish=True,       # Candidate publishes camera + mic
                can_subscribe=True,     # Candidate receives agent audio
            )
        )
        .with_ttl(datetime.timedelta(hours=1))   # 1-hour token expiry
    )

    return {
        "token": token.to_jwt(),
        "serverUrl": LIVEKIT_URL,
        "room": room,
        "identity": username,
    }


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": int(time.time())}
