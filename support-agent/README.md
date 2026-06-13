# AI Project Review Agent

A live project review agent powered by **LiveKit Agents** + **Gemini 3.1 Flash Live**. Acts as an AI teacher that conducts a real-time camera-enabled project review session with students.

## What it does

- Student joins via browser (camera + mic)
- Agent sees the student through their camera
- Greets them, compliments what they're wearing, then conducts a structured project review
- Asks the student to display and explain their project
- Formulates targeted questions based on what it sees and hears
- Requests live demos of specific modules
- Ends with honest feedback and reflection prompts

## Architecture

```
Browser (React + LiveKit)  ←→  LiveKit Cloud  ←→  Python Agent (Gemini 3.1 Live)
         ↑
  Token Server (FastAPI)
```

## Prerequisites

- **Python 3.11+** with [uv](https://docs.astral.sh/uv/getting-started/installation/) installed
- **Node.js 18+** with npm
- A [LiveKit Cloud](https://cloud.livekit.io) account (free tier works) — you need a project URL, API key, and API secret
- A [Gemini API key](https://aistudio.google.com/apikey) with access to Gemini Live models

## Setup

### 1. Clone & install Python deps

```bash
git clone <repo-url>
cd support-agent
uv sync
```

### 2. Install frontend deps

```bash
cd interview_app && npm install && cd ..
```

### 3. Configure environment

```bash
cp .env.example .env.local
```

Open `.env.local` and fill in:

```env
GEMINI_API_KEY=your_gemini_key_here

LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=APIxxxxxxxxxxxxxxx
LIVEKIT_API_SECRET=your_secret_here
```

> **Where to get these:**
> - Gemini key → [Google AI Studio](https://aistudio.google.com/apikey)
> - LiveKit credentials → [LiveKit Cloud dashboard](https://cloud.livekit.io) → your project → Settings → Keys

### 4. Run everything (3 terminals)

**Terminal 1 — Token server:**
```bash
uv run uvicorn token_server:app --port 8080 --reload
```

**Terminal 2 — LiveKit agent:**
```bash
uv run python agent.py dev
```

**Terminal 3 — Frontend:**
```bash
cd interview_app && npm run dev
```

Open [http://localhost:5173](http://localhost:5173), enter your name, and click **Join**.

> Make sure to **allow camera and microphone** when the browser asks.
> Say **"Hello"** once you're in — the agent waits for you to speak first.

## Project structure

```
support-agent/
├── agent.py           # LiveKit agent — Gemini 3.1 Flash Live, VAD, video input
├── prompts.py         # Teacher persona & project review instructions
├── token_server.py    # FastAPI server issuing LiveKit JWT tokens
├── tools.py           # (placeholder for future tools)
├── pyproject.toml     # Python dependencies
├── interview_app/     # React/TypeScript frontend
│   └── src/
│       ├── App.tsx
│       ├── PreJoin.tsx       # Name + room entry screen
│       └── InterviewRoom.tsx # Live session UI (camera, agent panel, controls)
└── .env.example       # Environment variable template
```

## Tech stack

- [LiveKit Agents](https://docs.livekit.io/agents/) v1.5+
- [Gemini 3.1 Flash Live](https://deepmind.google/technologies/gemini/) — realtime multimodal (audio + video)
- [Silero VAD](https://github.com/snakers4/silero-vad) — voice activity detection
- React + TypeScript + Tailwind CSS
- FastAPI token server
