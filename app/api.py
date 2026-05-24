from datetime import datetime, timezone
from typing import List, Optional
from collections import Counter

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from groq import Client

import logging
import re
import os
import json
import math
import sqlite3
import io
import tempfile

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
router = APIRouter()

# ══════════════════════════════════════════════════════════════
#  MODELS
# ══════════════════════════════════════════════════════════════
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    question: str
    history: List[Message] = []
    lang: str = "auto"
    session_id: Optional[str] = None

class SymbolRequest(BaseModel):
    symbol: str
    lang: str = "auto"

class AnalyseRequest(BaseModel):
    dreams: List[str]
    lang: str = "auto"

class PatternRequest(BaseModel):
    sessions: List[List[str]]
    lang: str = "auto"

class JournalRequest(BaseModel):
    dream: str
    interpretation: str
    lang: str = "auto"

class ReportRequest(BaseModel):
    session_ids: List[str] = []
    lang: str = "auto"

# ══════════════════════════════════════════════════════════════
#  LANGUAGE DETECTION
# ══════════════════════════════════════════════════════════════
ARABIC_RE    = re.compile(r'[\u0600-\u06FF]')
FRENCH_WORDS = {
    # pronouns
    "je","tu","il","elle","nous","vous","ils","elles","ce","ca","ça",
    # common verbs & conjugations
    "est","était","sont","étaient","avoir","être","suis","as","avez","ont",
    "dure","dur","fais","fait","faire","voir","dit","dire","aller","vais","vas",
    "reve","rêve","rêvé","reve","songe","dormi","réveillé","senti","sentais",
    "estait","c'était","c'etait","cest","c'est","cetait",
    # articles & prepositions
    "le","la","les","un","une","des","du","de","en","dans","avec","sur","sous",
    "pour","par","mais","ou","et","donc","car","que","qui","quoi","dont",
    # common adjectives & adverbs
    "très","trop","bien","mal","vrai","faux","grand","petit","beau","belle",
    "difficile","facile","bizarre","étrange","peur","calme","seul","seule",
    # dream-specific
    "nuit","rêve","cauchemar","songe","mort","eau","feu","maison","voler","tomber",
    "chat","chien","serpent","homme","femme","enfant","maman","papa",
    # short words that unmistakably signal French
    "mon","ma","mes","ton","ta","tes","son","sa","ses","leur","leurs",
    "moi","toi","lui","eux","nos","vos","cet","cette","ces","au","aux",
    # contractions split by apostrophe
    "estait","cest","cest","cetait","cétait","etait","était","jai","jai",
    "suis","nest","nai","cela","ceci"
}
DERIJA_WORDS = {
    "bech","wlah","mta3","mteaa","reit","reyt","cheft","chouft","7lam",
    "7lem","lilt","nhar","9al","gal","wesh","ash","barcha","behi","mazal",
    "taw","enti","ena","baba","ommi","khti","khoya","famma","barra","jit","mchit",
    "haka","heka","kima","kifeh","3andi","nta","nti","houma","fama","barra",
    # dream-specific Derija words
    "bera7","hlmt","bory","walit","3asfoura","taswira","7lem","hlima","khyal"
}

def detect_lang(text: str) -> str:
    if not text: return "en"
    if ARABIC_RE.search(text): return "ar"

    # Normalize: lowercase, remove punctuation but keep letters and spaces
    normalized = text.lower()
    # Replace apostrophes with space so "c'était" → "c était" → ["c","était"]
    normalized = normalized.replace("'", " ").replace("'", " ").replace("`", " ")
    words = set(re.findall(r'[a-z\u00e0-\u00ff\d]+', normalized))

    if words & DERIJA_WORDS: return "derija"
    if words & FRENCH_WORDS: return "fr"

    # Extra check: French accent characters are a strong signal
    if re.search(r'[àâäéèêëîïôùûüçœæ]', text.lower()): return "fr"

    return "en"

def clean(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

# ══════════════════════════════════════════════════════════════
#  LUNAR PHASE
# ══════════════════════════════════════════════════════════════
LUNAR_CYCLE = 29.53058867

def lunar_phase() -> dict:
    known_new = datetime(2000, 1, 6, 18, 14, tzinfo=timezone.utc)
    now       = datetime.now(timezone.utc)
    age       = (now - known_new).total_seconds() / 86400.0 % LUNAR_CYCLE
    pct       = age / LUNAR_CYCLE

    if   pct < 0.025: phase,emoji = "New Moon","🌑"
    elif pct < 0.25:  phase,emoji = "Waxing Crescent","🌒"
    elif pct < 0.275: phase,emoji = "First Quarter","🌓"
    elif pct < 0.475: phase,emoji = "Waxing Gibbous","🌔"
    elif pct < 0.525: phase,emoji = "Full Moon","🌕"
    elif pct < 0.725: phase,emoji = "Waning Gibbous","🌖"
    elif pct < 0.775: phase,emoji = "Last Quarter","🌗"
    elif pct < 0.975: phase,emoji = "Waning Crescent","🌘"
    else:             phase,emoji = "New Moon","🌑"

    meanings = {
        "New Moon":        "A threshold night. Dreams carry seeds of the unformed.",
        "Waxing Crescent": "Desire accelerates. Dreams surface yearning and unfinished intentions.",
        "First Quarter":   "Tension and decision. The dream-mind dramatises obstacles and crossroads.",
        "Waxing Gibbous":  "Refinement. The unconscious is close to naming something it has circled for weeks.",
        "Full Moon":       "Peak luminosity. Dreams are vivid, archetypal, emotionally saturated.",
        "Waning Gibbous":  "The tide recedes. Dreams are rich with aftermath — what was gained, what was lost.",
        "Last Quarter":    "A reckoning. The unconscious surfaces what has run its course.",
        "Waning Crescent": "The dark before renewal. Dreams are shadowed, liminal, deeply honest.",
    }
    return {
        "phase": phase, "emoji": emoji,
        "meaning": meanings[phase],
        "age_days": round(age, 1),
        "illumination_pct": round(abs(math.cos(math.pi * age / (LUNAR_CYCLE/2))) * 100, 1),
    }

# ══════════════════════════════════════════════════════════════
#  KNOWLEDGE CORPUS  — CSV + PDFs (same as original working app)
#  Using same embedding model as the original: multi-qa-MiniLM-L6-cos-v1
# ══════════════════════════════════════════════════════════════
logger.info("Loading embeddings model…")

def load_pdf(path: str) -> list:
    docs = []
    try:
        raw  = PyPDFLoader(path).load()
        good = [d for d in raw if len(d.page_content.strip()) > 60]
        if len(good) / max(len(raw), 1) > 0.25:
            logger.info(f"[pdf] {path}: {len(good)} pages via PyPDF")
            return good
    except Exception as e:
        logger.warning(f"PyPDF failed {path}: {e}")
    try:
        from pdfminer.high_level import extract_pages
        from pdfminer.layout import LTTextContainer
        for i, layout in enumerate(extract_pages(path)):
            t = clean(" ".join(el.get_text() for el in layout if isinstance(el, LTTextContainer)))
            if len(t) > 60:
                docs.append(Document(page_content=t, metadata={"source": "knowledge", "page": i}))
        logger.info(f"[pdf] {path}: {len(docs)} pages via pdfminer")
    except Exception as e:
        logger.error(f"pdfminer failed {path}: {e}")
    return docs

# Same embedding model as the original working app
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
)

PERSIST_DIR = "MyDB_v8"

if os.path.exists(PERSIST_DIR):
    logger.info("Loading persisted vector store (CSV + PDFs)…")
    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
else:
    logger.info("Building corpus from CSV + PDFs…")

    # CSV — dream symbol interpretations
    csv_path = "app/documents/dreams_interpretations.csv"
    csv_docs = []
    if os.path.exists(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path)
        csv_docs = [
            Document(
                page_content=clean(str(r["Interpretation"])),
                metadata={"source": "csv", "symbol": str(r["Dream Symbol"])}
            )
            for _, r in df.iterrows()
            if pd.notna(r["Interpretation"]) and pd.notna(r["Dream Symbol"])
        ]
        logger.info(f"CSV docs: {len(csv_docs)}")
    else:
        logger.warning("CSV not found — continuing with PDFs only")

    # PDFs — Freud + Ibn Sirine
    pdf_docs = load_pdf("app/documents/freud.pdf") + load_pdf("app/documents/IbnSirine.pdf")

    all_raw = csv_docs + pdf_docs
    logger.info(f"Total raw docs: {len(all_raw)}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=40)
    chunks   = [
        Document(page_content=c, metadata=doc.metadata)
        for doc in all_raw
        for c in splitter.split_text(doc.page_content)
    ]
    logger.info(f"Total chunks: {len(chunks)}")

    if chunks:
        # ChromaDB has a max batch size — insert in batches of 5000
        BATCH_SIZE = 5000
        if len(chunks) <= BATCH_SIZE:
            vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_DIR)
        else:
            logger.info(f"Inserting {len(chunks)} chunks in batches of {BATCH_SIZE}…")
            vectorstore = Chroma.from_documents(chunks[:BATCH_SIZE], embeddings, persist_directory=PERSIST_DIR)
            for i in range(BATCH_SIZE, len(chunks), BATCH_SIZE):
                batch = chunks[i:i+BATCH_SIZE]
                vectorstore.add_documents(batch)
                logger.info(f"Added batch up to chunk {i+len(batch)}")
    else:
        logger.error("No chunks built — vector store will be empty")
        vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
logger.info("Vector store ready.")

# ══════════════════════════════════════════════════════════════
#  DUAL LLM: GROQ (default) + GEMINI (Derija only)
# ══════════════════════════════════════════════════════════════
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY: raise ValueError("GROQ_API_KEY not set in .env")
groq = Client(api_key=GROQ_API_KEY)

_gemini_client = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    google_api_key=os.environ.get("GEMINI_API_KEY"),
    temperature=0.7,
)

def llm(messages: list, temperature=0.65, max_tokens=900, lang="en") -> str:
    """Route to Gemini for Derija, Groq for everything else."""
    if lang == "derija":
        lc_messages = []
        for m in messages:
            if m["role"] == "system":
                lc_messages.append(SystemMessage(content=m["content"]))
            elif m["role"] == "user":
                lc_messages.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant":
                lc_messages.append(AIMessage(content=m["content"]))
        result = _gemini_client.invoke(lc_messages).content
        if isinstance(result, list):
            result = "".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in result)
        return result.strip()
    else:
        r = groq.chat.completions.create(
            model="llama-3.3-70b-versatile", messages=messages,
            temperature=temperature, max_tokens=max_tokens, top_p=0.9)
        return r.choices[0].message.content.strip()

def llm_json(messages: list, lang="en") -> dict:
    raw = llm(messages, temperature=0.1, max_tokens=256, lang=lang)
    raw = re.sub(r"```json|```", "", raw).strip()
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        try: return json.loads(m.group())
        except: pass
    return {}

# ══════════════════════════════════════════════════════════════
#  ✦ CORE PROMPT — HUMAN THERAPIST EDITION
#
#  Responds like a real person: warm when greeted, direct when
#  interpreting, short and precise always. No poetry, no moon,
#  no Islamic citations. Just clean psychological insight in
#  plain conversational language.
# ══════════════════════════════════════════════════════════════
CORE_PROMPT = """You are Oniromancer, a perceptive dream interpreter with real psychological depth.

LANGUAGE — THIS IS THE MOST IMPORTANT RULE:
Reply ONLY in the language the user wrote in. No exceptions. Ever.
Arabic → Arabic. French → French. Darija → Darija. English → English.
If unsure, match the script or dominant words.
NEVER say "I'll respond in English" — just respond in their language.

YOUR PERSONALITY:
You are warm, direct, and psychologically sharp. You talk like a real person, not an AI assistant.
You do NOT use filler phrases like "It's like you're reaching out for a hand to hold" or "walking through a desert".
You are specific. Every sentence references something the person actually said or dreamed.

HOW TO INTERPRET DREAMS:
Focus on the emotional truth — what feeling is this dream staging, and why NOW?
Don't explain symbols generically. "Iron walls = feeling trapped" is lazy. Go deeper.
Connect to what the person has shared about their life in this conversation.
Use "you" directly. Make it feel personal and observed, not textbook.

FORMAT:
2-4 paragraphs of flowing prose. No headers, no bullets, no numbered lists.
Only ask ONE question at the end — and only when it genuinely deepens the conversation.
Don't ask a question after every single response. Sometimes just interpret.

WHEN THE USER SHARES PERSONAL PAIN (grief, loneliness, loss):
Acknowledge it briefly and naturally, then connect it back to the dream.
Don't become a general therapist. You interpret dreams — that's your lane.
If the dream connects to their pain, say how. That's more useful than generic comfort.

CRISIS MESSAGES ("i want to kill myself", "i want to die"):
Take it seriously but stay calm. Say something like:
"That's important to say. Are you safe right now? If things feel really dark, please reach out to someone you trust or a crisis line — you don't have to carry this alone."
Then gently return to the conversation. Don't shut down completely. Don't refuse to talk.
If they say they were joking, accept it simply and move on without lecturing them.

SCOPE:
You are a dream interpreter. When conversations drift too far from dreams, gently bring it back.
But do it naturally — not with "I'm only here to talk about dreams."

WHEN GREETED:
Reply warmly in 1-2 sentences in their language. Ask them to share their dream. Nothing more."""

LANG_PROMPTS = {
    "fr":     "Réponds uniquement en français.",
    "ar":     "أجب فقط باللغة العربية.",
    "derija": "جاوب فقط بالدّارجة التونسية.",
    "en":     "Respond only in English.",
}

CLASSIFY_PROMPT = """Classify this dream. Return ONLY valid JSON, no other text:
{"dream_type":"one of [prophetic,compensatory,trauma,wish-fulfilment,shadow,liminal,mundane]","emotion":"one of [dread,longing,confusion,revelation,grief,ecstasy,rage,tenderness,void]","core_symbol":"the single most potent image (2-4 words max)","intensity":integer_1_to_5}"""

# ══════════════════════════════════════════════════════════════
#  SQLITE
# ══════════════════════════════════════════════════════════════
DB_PATH = "bi.db"

def init_db():
    with sqlite3.connect(DB_PATH) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session TEXT, role TEXT, content TEXT, lang TEXT, ts TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS dream_meta (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session TEXT, dream_type TEXT, emotion TEXT,
            core_symbol TEXT, intensity INTEGER, moon_phase TEXT, ts TEXT)""")
        c.commit()

init_db()

def log_exchange(sid, q, a, lang):
    ts = datetime.utcnow().isoformat()
    with sqlite3.connect(DB_PATH) as c:
        c.execute("INSERT INTO sessions VALUES(NULL,?,?,?,?,?)", (sid,"user",q,lang,ts))
        c.execute("INSERT INTO sessions VALUES(NULL,?,?,?,?,?)", (sid,"assistant",a,lang,ts))
        c.commit()

def log_meta(sid, meta, moon_phase):
    with sqlite3.connect(DB_PATH) as c:
        c.execute("INSERT INTO dream_meta VALUES(NULL,?,?,?,?,?,?,?)",
            (sid, meta.get("dream_type",""), meta.get("emotion",""),
             meta.get("core_symbol",""), meta.get("intensity",0),
             moon_phase, datetime.utcnow().isoformat()))
        c.commit()

# ══════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════

@router.post("/chat")
async def chat(req: ChatRequest):
    try:
        question = (req.question or "").strip()
        if not question:
            raise HTTPException(400, "Empty question")

        logger.info(f"/chat  lang={req.lang}  q_len={len(question)}  history={len(req.history)}")

        moon = lunar_phase()

        lang = req.lang if req.lang != "auto" else detect_lang(question)

        # If current message is ambiguous (detected as English) but history shows
        # a different language, use that — handles short replies like "oui", "non", "c'est dur"
        if lang == "en" and req.history:
            history_text = " ".join(m.content for m in req.history if m.role == "user")
            h_lang = detect_lang(history_text)
            if h_lang != "en":
                lang = h_lang

        lang_label = lang  # kept for logging only

        # Retrieve relevant knowledge (skip for pure greetings)
        greetings = {"hey","hi","hello","hola","salut","salam","sup","yo","bonjour",
                     "marhba","ahla","marhaba","coucou","bonsoir","wesh","wesh wesh",
                     "bonsoir","slt","bjr","cva","ça va","ca va"}
        is_greeting = len(question.split()) <= 4 and question.lower().strip(".,!?") in greetings

        context = ""
        if not is_greeting:
            try:
                relevant = retriever.invoke(question)
            except Exception:
                relevant = retriever.get_relevant_documents(question)
            context = "\n\n".join(d.page_content for d in relevant)

        # Build message array — language rule FIRST, short and strong
        msgs = [
            {"role": "system", "content": CORE_PROMPT},
            {"role": "system", "content": LANG_PROMPTS.get(lang, "Respond only in English.")},
        ]

        if context.strip():
            msgs.append({"role": "system", "content":
                f"Use the following as PRIMARY interpretive material. Base your psychological reading strongly on these sources — they contain real symbolic, psychoanalytic and oneiric knowledge:\n\n{context}"})

        # Conversation history
        for m in req.history[-16:]:
            msgs.append({"role": m.role, "content": m.content})

        msgs.append({"role": "user", "content": question})

        answer = llm(msgs, temperature=0.45, max_tokens=700, lang=lang)

        # Parallel classification
        meta = llm_json([
            {"role": "system", "content": CLASSIFY_PROMPT},
            {"role": "user",   "content": question}
        ], lang=lang)

        sid = req.session_id or (req.history[0].content[:40] if req.history else question[:40])
        try:
            log_exchange(sid, question, answer, lang)
            if meta: log_meta(sid, meta, moon["phase"])
        except Exception as e:
            logger.warning(f"DB write: {e}")

        return jsonable_encoder({
            "response":    answer,
            "lang":        lang,
            "moon":        moon,
            "dream_type":  meta.get("dream_type", ""),
            "emotion":     meta.get("emotion", ""),
            "core_symbol": meta.get("core_symbol", ""),
            "intensity":   meta.get("intensity", 0),
        })

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(500, detail=str(e))


@router.post("/symbol")
async def symbol_oracle(req: SymbolRequest):
    try:
        lang = req.lang if req.lang != "auto" else detect_lang(req.symbol)
        try:
            relevant = retriever.invoke(req.symbol)
        except Exception:
            relevant = retriever.get_relevant_documents(req.symbol)
        context = "\n\n".join(d.page_content for d in relevant[:6])

        answer = llm([
            {"role": "system", "content":
                f"""You are a precise symbol analyst. For the given dream symbol, give a direct expert reading:
— Its core psychological meaning (specific drives or complexes it represents)
— What it typically signals about the person's current inner state
— One non-obvious insight most people miss

80-120 words maximum. Conversational, direct. No filler, no "this could represent".
Reply in language: {lang}.

Reference knowledge:\n\n{context}"""},
            {"role": "user", "content": f'Symbol: "{req.symbol}"'}
        ], temperature=0.55, max_tokens=300, lang=lang)

        return {"symbol": req.symbol, "reading": answer, "lang": lang}
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@router.post("/analyse")
async def analyse(req: AnalyseRequest):
    try:
        if not req.dreams: raise HTTPException(400, "No dreams provided")

        # Detect language from ALL dreams combined, not just first
        all_text = " ".join(req.dreams)
        lang = req.lang if req.lang != "auto" else detect_lang(all_text)
        if lang == "en":
            h_lang = detect_lang(all_text)
            if h_lang != "en": lang = h_lang

        block = "\n\n---\n\n".join(f"Dream {i+1}: {d}" for i, d in enumerate(req.dreams))

        answer = llm([
            {"role": "system", "content": LANG_PROMPTS.get(lang, "Respond only in English.")},
            {"role": "system", "content":
            f"""You are a perceptive dream analyst. Read these {len(req.dreams)} dream(s) from the same person:

{block}

Write a flowing psychological profile (200-300 words) that:
- Identifies the 2-3 most recurring emotional themes or images — be specific to what actually appears in these dreams
- Names what the person seems to be emotionally processing right now
- Points to the shadow material — what is being avoided or showing up in disguised form
- Ends with one sharp insight that goes straight to the emotional core

Do NOT use numbered lists or headers. Pure flowing prose.
Talk directly to the person using "you" and "your".
No generic symbolism — every sentence must be specific to these actual dreams."""}
        ], temperature=0.45, max_tokens=600, lang=lang)

        return {"profile": answer, "dream_count": len(req.dreams), "lang": lang}
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@router.post("/pattern")
async def pattern(req: PatternRequest):
    try:
        all_dreams = [d for s in req.sessions for d in s]
        if not all_dreams: raise HTTPException(400, "No dreams provided")
        lang = req.lang if req.lang != "auto" else detect_lang(" ".join(all_dreams))
        block = "\n\n---\n\n".join(
            f"Session {i+1}, Dream {j+1}: {d}"
            for i, s in enumerate(req.sessions) for j, d in enumerate(s))

        answer = llm([
            {"role": "system", "content": LANG_PROMPTS.get(lang, "Respond only in English.")},
            {"role": "system", "content":
            f"""Read these dreams from {len(req.sessions)} different sessions by the same person:

{block}

Find the patterns this person cannot see themselves. Write 250-350 words of direct flowing prose:
- What symbol or emotional theme keeps returning across sessions, even in different forms
- How the emotional state has shifted over time
- The one central unresolved tension driving all of it
- What this person most needs to hear right now

Talk directly to them. No headers. No lists. Specific to these actual dreams."""}
        ], temperature=0.45, max_tokens=650, lang=lang)

        return {"patterns": answer, "session_count": len(req.sessions), "total_dreams": len(all_dreams), "lang": lang}
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@router.post("/journal")
async def journal(req: JournalRequest):
    try:
        lang = req.lang if req.lang != "auto" else detect_lang(req.dream)
        moon = lunar_phase()

        answer = llm([{"role": "system", "content":
            f"""Write a personal dream journal entry. First person, literary but clear, no analysis headers.

Date: {datetime.now().strftime('%B %d, %Y')}  Moon: {moon['phase']} {moon['emoji']}

Open with the date and moon naturally woven in. Describe the dream with sensory detail. Weave the interpretation in as felt realisation, not clinical explanation. 180-240 words. Close with one sentence worth returning to.

Reply in language: {lang}.

Dream: {req.dream}
Interpretation: {req.interpretation}"""}],
        temperature=0.72, max_tokens=400)

        return {"entry": answer, "date": datetime.now().strftime('%B %d, %Y'), "moon": moon, "lang": lang}
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@router.post("/report")
async def generate_report(req: ReportRequest):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import mm
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                         HRFlowable, Table, TableStyle, PageBreak, KeepTogether)
        from reportlab.lib.enums import TA_CENTER
        from reportlab.pdfgen import canvas as pdf_canvas

        W, H   = A4
        INK    = colors.HexColor('#0e0b1e')
        ACC    = colors.HexColor('#8b6fd4')
        MID    = colors.HexColor('#6a6082')
        DIVIDER= colors.HexColor('#d4cce8')
        CODE_BG= colors.HexColor('#f0ecf8')
        WHITE  = colors.white

        with sqlite3.connect(DB_PATH) as c:
            if req.session_ids:
                ph = ",".join("?" * len(req.session_ids))
                metas  = c.execute(f"SELECT dream_type,emotion,core_symbol,intensity,moon_phase,ts FROM dream_meta WHERE session IN ({ph}) ORDER BY ts", req.session_ids).fetchall()
                dreams = c.execute(f"SELECT content,ts FROM sessions WHERE session IN ({ph}) AND role='user' ORDER BY ts", req.session_ids).fetchall()
            else:
                metas  = c.execute("SELECT dream_type,emotion,core_symbol,intensity,moon_phase,ts FROM dream_meta ORDER BY ts").fetchall()
                dreams = c.execute("SELECT content,ts FROM sessions WHERE role='user' ORDER BY ts").fetchall()

        if not dreams:
            raise HTTPException(400, "No dream data found. Record some dreams first.")

        dream_texts   = [d[0] for d in dreams]
        lang          = req.lang if req.lang != "auto" else "en"
        type_counts   = Counter(m[0] for m in metas if m[0])
        emot_counts   = Counter(m[1] for m in metas if m[1])
        sym_counts    = Counter(m[2] for m in metas if m[2])
        avg_intensity = round(sum(m[3] for m in metas if m[3]) / max(len(metas), 1), 1)

        block = "\n\n".join(f"Dream {i+1}: {t}" for i, t in enumerate(dream_texts[:15]))
        narrative = llm([{"role": "system", "content":
            f"""Write a personalised unconscious report (350-450 words) based on these recorded dreams.

{block}

Structure: opening observation about this person's inner life → 2-3 dominant symbolic themes (named specifically) → primary archetype (named precisely) → shadow content → emotional trajectory → closing message from the unconscious to this person.

Specific, direct, expert. No headers. Flowing prose. Reply in language: {lang}."""}],
        temperature=0.68, max_tokens=700)

        buf = io.BytesIO()

        def S(n, **k): return ParagraphStyle(n, **k)
        sTitle  = S('T',  fontName='Helvetica-Bold',    fontSize=24, leading=30, textColor=INK, alignment=TA_CENTER, spaceAfter=4)
        sSub    = S('Su', fontName='Helvetica',          fontSize=10, leading=14, textColor=MID, alignment=TA_CENTER, spaceAfter=16)
        sH1     = S('H1', fontName='Helvetica-Bold',    fontSize=14, leading=18, textColor=ACC, spaceBefore=14, spaceAfter=5)
        sH2     = S('H2', fontName='Helvetica-Bold',    fontSize=11, leading=14, textColor=INK, spaceBefore=9,  spaceAfter=3)
        sBody   = S('B',  fontName='Helvetica',          fontSize=10, leading=15, textColor=INK, spaceAfter=5)
        sItalic = S('I',  fontName='Helvetica-Oblique', fontSize=10, leading=15, textColor=INK, spaceAfter=5)

        class RC(pdf_canvas.Canvas):
            def __init__(self, *a, **k): super().__init__(*a, **k); self._s = []
            def showPage(self): self._s.append(dict(self.__dict__)); self._startPage()
            def save(self):
                n = len(self._s)
                for st in self._s:
                    self.__dict__.update(st); pg = self._pageNumber
                    if pg > 1:
                        self.setStrokeColor(DIVIDER); self.setLineWidth(.4)
                        self.line(18*mm, H-14*mm, W-18*mm, H-14*mm)
                        self.setFont('Helvetica', 7.5); self.setFillColor(MID)
                        self.drawString(18*mm, H-11*mm, 'Oniromancer — Unconscious Intelligence Report')
                        self.drawRightString(W-18*mm, H-11*mm, f'Page {pg} of {n}')
                        self.line(18*mm, 13*mm, W-18*mm, 13*mm)
                        self.drawCentredString(W/2, 9*mm, '☽  Personal & Confidential')
                    super().showPage()
                super().save()

        doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=22*mm, rightMargin=22*mm, topMargin=20*mm, bottomMargin=20*mm)
        story = []
        hr  = lambda c=DIVIDER, t=.5: HRFlowable(width='100%', thickness=t, color=c, spaceAfter=5, spaceBefore=5)
        sp  = lambda n=6: Spacer(1, n)
        def h1(t): return Paragraph(t, sH1)
        def h2(t): return Paragraph(t, sH2)
        def body(t): return Paragraph(t, sBody)
        def italic(t): return Paragraph(f'<i>{t}</i>', sItalic)

        moon_now = lunar_phase()
        story += [sp(32),
            Paragraph('☽', S('G', fontName='Helvetica', fontSize=52, textColor=ACC, alignment=TA_CENTER, spaceAfter=8)),
            Paragraph('Unconscious Intelligence Report', sTitle),
            Paragraph('Generated by Oniromancer', sSub),
            hr(ACC, 1), sp(10)]

        cover_data = [
            ['Report Date',     datetime.now().strftime('%B %d, %Y')],
            ['Moon Phase',      f"{moon_now['emoji']} {moon_now['phase']}"],
            ['Dreams Analysed', str(len(dream_texts))],
            ['Avg Intensity',   f'{avg_intensity} / 5'],
            ['Primary Emotion', emot_counts.most_common(1)[0][0].title() if emot_counts else 'N/A'],
            ['Dominant Type',   type_counts.most_common(1)[0][0].title() if type_counts else 'N/A'],
        ]
        ct = Table(cover_data, colWidths=[48*mm, 108*mm])
        ct.setStyle(TableStyle([
            ('FONTNAME',(0,0),(0,-1),'Helvetica-Bold'), ('FONTNAME',(1,0),(1,-1),'Helvetica'),
            ('FONTSIZE',(0,0),(-1,-1),9.5),
            ('TEXTCOLOR',(0,0),(0,-1),MID), ('TEXTCOLOR',(1,0),(1,-1),INK),
            ('ROWBACKGROUNDS',(0,0),(-1,-1),[WHITE,CODE_BG]),
            ('TOPPADDING',(0,0),(-1,-1),6), ('BOTTOMPADDING',(0,0),(-1,-1),6), ('LEFTPADDING',(0,0),(-1,-1),10),
            ('BOX',(0,0),(-1,-1),.4,DIVIDER), ('INNERGRID',(0,0),(-1,-1),.3,DIVIDER),
        ]))
        story += [ct, PageBreak()]

        story += [h1('I.  Your Unconscious Landscape'), hr(), sp(4)]
        for para in narrative.split('\n\n'):
            p = para.strip()
            if p: story.append(italic(p))
        story.append(sp(6))

        story += [PageBreak(), h1('II.  Dream Analytics'), hr(), sp(4)]

        if type_counts:
            story.append(h2('Dream Type Distribution'))
            td = [['Type','Count','%']] + [[k.title(), str(v), f'{round(v/sum(type_counts.values())*100)}%'] for k,v in type_counts.most_common()]
            tt = Table(td, colWidths=[70*mm,30*mm,30*mm])
            tt.setStyle(TableStyle([
                ('BACKGROUND',(0,0),(-1,0),ACC),('TEXTCOLOR',(0,0),(-1,0),WHITE),
                ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTNAME',(0,1),(-1,-1),'Helvetica'),
                ('FONTSIZE',(0,0),(-1,-1),9),('ROWBACKGROUNDS',(0,1),(-1,-1),[WHITE,CODE_BG]),
                ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('LEFTPADDING',(0,0),(-1,-1),8),
                ('BOX',(0,0),(-1,-1),.4,DIVIDER),('INNERGRID',(0,0),(-1,-1),.3,DIVIDER),
            ]))
            story += [tt, sp(10)]

        if emot_counts:
            story.append(h2('Emotional Signature'))
            ed = [['Emotion','Count','Bar']] + [[k.title(), str(v), '▪'*min(v,10)] for k,v in emot_counts.most_common()]
            et = Table(ed, colWidths=[60*mm,30*mm,70*mm])
            et.setStyle(TableStyle([
                ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#4a3870')),('TEXTCOLOR',(0,0),(-1,0),WHITE),
                ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTNAME',(0,1),(-1,-1),'Helvetica'),
                ('FONTSIZE',(0,0),(-1,-1),9),('ROWBACKGROUNDS',(0,1),(-1,-1),[WHITE,CODE_BG]),
                ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('LEFTPADDING',(0,0),(-1,-1),8),
                ('BOX',(0,0),(-1,-1),.4,DIVIDER),('INNERGRID',(0,0),(-1,-1),.3,DIVIDER),
            ]))
            story += [et, sp(10)]

        if sym_counts:
            story.append(h2('Recurring Core Symbols'))
            sd = [['Symbol','Appearances']] + [[k,str(v)] for k,v in sym_counts.most_common(10)]
            st2 = Table(sd, colWidths=[100*mm,60*mm])
            st2.setStyle(TableStyle([
                ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#5a4020')),('TEXTCOLOR',(0,0),(-1,0),WHITE),
                ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTNAME',(0,1),(-1,-1),'Helvetica'),
                ('FONTSIZE',(0,0),(-1,-1),9),('ROWBACKGROUNDS',(0,1),(-1,-1),[WHITE,CODE_BG]),
                ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('LEFTPADDING',(0,0),(-1,-1),8),
                ('BOX',(0,0),(-1,-1),.4,DIVIDER),('INNERGRID',(0,0),(-1,-1),.3,DIVIDER),
            ]))
            story.append(st2)

        story += [PageBreak(), h1('III.  Dream Record'), hr(), sp(4)]
        for i, (text, ts) in enumerate(dreams[:20]):
            date_str = ts[:10] if ts else ''
            story.append(KeepTogether([
                h2(f'Dream {i+1}   ·   {date_str}'),
                italic(text[:500] + ('…' if len(text) > 500 else '')),
                sp(4),
            ]))

        story += [hr(ACC,1), sp(8),
            Paragraph('☽  Oniromancer', S('F', fontName='Helvetica-BoldOblique', fontSize=12, textColor=ACC, alignment=TA_CENTER, spaceAfter=3)),
            Paragraph('This report is a personal reflection tool. For clinical concerns, consult a qualified professional.',
                S('Fn', fontName='Helvetica-Oblique', fontSize=8, textColor=MID, alignment=TA_CENTER))]

        doc.build(story, canvasmaker=RC)
        buf.seek(0)
        filename = f"oniromancer_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        return StreamingResponse(buf, media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'})

    except ImportError:
        raise HTTPException(500, "reportlab not installed: pip install reportlab")
    except Exception as e:
        logger.error(f"Report error: {e}", exc_info=True)
        raise HTTPException(500, detail=str(e))


@router.get("/dashboard")
async def dashboard():
    try:
        with sqlite3.connect(DB_PATH) as c:
            metas         = c.execute("SELECT dream_type,emotion,core_symbol,intensity,moon_phase,ts FROM dream_meta ORDER BY ts").fetchall()
            dreams        = c.execute("SELECT content,ts,session FROM sessions WHERE role='user' ORDER BY ts").fetchall()
            session_count = c.execute("SELECT COUNT(DISTINCT session) FROM sessions").fetchone()[0]

        if not metas:
            return {"empty": True, "session_count": 0, "dream_count": 0}

        type_counts   = Counter(m[0] for m in metas if m[0])
        emot_counts   = Counter(m[1] for m in metas if m[1])
        sym_counts    = Counter(m[2] for m in metas if m[2])
        moon_counts   = Counter(m[4] for m in metas if m[4])
        avg_intensity = round(sum(m[3] for m in metas if m[3]) / max(len(metas), 1), 1)

        timeline: dict = {}
        for _, ts, _ in dreams:
            day = ts[:10] if ts else None
            if day: timeline[day] = timeline.get(day, 0) + 1

        emotion_timeline = [{"ts": m[5][:10], "emotion": m[1]} for m in metas[-20:] if m[1]]

        return {
            "empty":            False,
            "dream_count":      len(dreams),
            "session_count":    session_count,
            "avg_intensity":    avg_intensity,
            "dream_types":      dict(type_counts.most_common()),
            "emotions":         dict(emot_counts.most_common()),
            "top_symbols":      dict(sym_counts.most_common(10)),
            "moon_phases":      dict(moon_counts.most_common()),
            "timeline":         timeline,
            "recent_symbols":   [m[2] for m in metas[-10:] if m[2]],
            "emotion_timeline": emotion_timeline,
            "dominant_emotion": emot_counts.most_common(1)[0][0] if emot_counts else "",
            "dominant_type":    type_counts.most_common(1)[0][0] if type_counts else "",
            "dominant_symbol":  sym_counts.most_common(1)[0][0] if sym_counts else "",
        }
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@router.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        audio = await file.read()
        ext   = os.path.splitext(file.filename or "rec.webm")[1] or ".webm"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(audio); tmp_path = tmp.name
        with open(tmp_path, "rb") as af:
            result = groq.audio.transcriptions.create(
                model="whisper-large-v3",
                file=(file.filename or f"rec{ext}", af),
                response_format="verbose_json")
        os.unlink(tmp_path)
        text     = result.text or ""
        detected = getattr(result, "language", None) or detect_lang(text)
        return {"transcript": text, "detected_lang": detected}
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@router.get("/lunar")
async def get_lunar(): return lunar_phase()


@router.get("/sessions")
async def list_sessions():
    with sqlite3.connect(DB_PATH) as c:
        rows = c.execute("""
            SELECT session, MIN(ts), COUNT(*)/2,
                   (SELECT content FROM sessions s2 WHERE s2.session=s1.session AND role='user' ORDER BY id LIMIT 1)
            FROM sessions s1 WHERE role='user'
            GROUP BY session ORDER BY MIN(ts) DESC LIMIT 50""").fetchall()
    return {"sessions": [{"id":r[0],"started":r[1],"turns":r[2],"preview":(r[3] or "")[:65]} for r in rows]}


@router.get("/history/{session_id}")
async def get_history(session_id: str):
    with sqlite3.connect(DB_PATH) as c:
        rows = c.execute("SELECT role,content,ts FROM sessions WHERE session=? ORDER BY id", (session_id,)).fetchall()
    return {"history": [{"role":r,"content":ct,"ts":t} for r,ct,t in rows]}


@router.get("/health")
async def health():
    moon = lunar_phase()
    return {"status": "ok", "moon": moon["phase"], "moon_emoji": moon["emoji"]}