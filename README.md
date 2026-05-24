# ☽ Oniromancer — Dream Intelligence Platform

**An AI-powered longitudinal dream intelligence platform** combining multilingual symbolic retrieval, Jungian/Islamic archetypal profiling, voice input, a live analytics dashboard, and personalised unconscious mapping.

## Quick Start (3 commands)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure — add your free Groq key
cp .env.example .env  # then edit .env

# 3. Run
uvicorn app.main:app --reload --port 8000
```

Then open **http://localhost:8000** — landing page, app, and dashboard all served automatically.

## Pages

| URL | Description |
|---|---|
| http://localhost:8000 | Landing page |
| http://localhost:8000/app | Dream chat interface |
| http://localhost:8000/dashboard | Analytics dashboard |
| http://localhost:8000/docs | API documentation |

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| /api/chat | POST | Main dream interpretation |
| /api/symbol | POST | Symbol oracle (cross-traditional) |
| /api/analyse | POST | Unconscious map from session |
| /api/pattern | POST | Cross-session pattern engine |
| /api/journal | POST | Literary journal entry generator |
| /api/report | POST | Full PDF unconscious report |
| /api/transcribe | POST | Voice → text (Whisper) |
| /api/lunar | GET | Current moon phase |
| /api/dashboard | GET | All analytics data |
| /api/health | GET | Server status |

## Free Services Used

- **Groq** — llama-3.3-70b-versatile + whisper-large-v3 (free tier, no billing)
- **sentence-transformers** — paraphrase-multilingual-mpnet-base-v2 (local)
- **ChromaDB** — local vector store (local)
- **SQLite** — dream log and metadata (local)
- **ReportLab** — PDF generation (local)

## Languages

English · Français · العربية · درجة تونسية (including phonetic: 7lam, wlah, mta3, 9al…)
