"""

  Additional installation for this file
  ─────────────────────────────────────────────────────────
    Offline TTS fallback uses espeak directly (no pyttsx3):
    ```
    sudo apt install espeak espeak-data libespeak-dev
    ```
"""

"""
============================================================
  🤖  Naaila — Robotwala RAG-Powered Speech-to-Speech Chatbot
  Production Release  (Audit Pass 3 — Raspberry Pi optimized)
============================================================

  Architecture
  ─────────────────────────────────────────────────────────
  • Hindi queries  → retrieved directly against hindi_details.pdf (rag_hi)
  • English queries → retrieved directly against english_details.pdf (rag_en)
  • Language detection picks the engine; the query is never translated.
  • Web fallback only fires when PDF score is below threshold AND the
    query contains time-sensitive keywords.

  Production Changes (over dev build)
  ─────────────────────────────────────────────────────────
  FIX-P1  Empty / whitespace-only user input rejected before reaching the LLM.
  FIX-P2  get_ai_reply() always returns a non-empty str or raises.
  FIX-P3  Conversation history capped at MAX_HISTORY_TURNS.
  FIX-P4  All print() calls replaced with the stdlib logging module.
  FIX-P5  asyncio event loop created once, reused by every speak() call.
  FIX-P6  Reply scoped per iteration via a helper function.
  FIX-P7  User text sanitized before being passed to the LLM / logged.
  FIX-P8  build_context() sets source="PDF" whenever pdf_context is non-empty.
  FIX-P9  transcribe() retries once with language="hi" when Whisper outputs
          Urdu/Arabic script for what is actually Hindi speech.

  ── Audit Pass 2 (reliability) ──────────────────────────────────────
  FIX-A1  is_internet_available() no longer mutates the global socket
          timeout — uses a scoped per-call timeout instead.
  FIX-A2  LLM retries distinguish retryable (429/5xx/network) from
          non-retryable (4xx) errors — fails fast on the latter.
  FIX-A3  Retry backoff includes jitter.
  FIX-A4  TokenStats: session/daily totals, avg/min/max tokens, avg
          latency, requests/hour, optional cost estimate.
  FIX-A5  LongTermMemory: selective, bounded, persisted durable facts.
  FIX-A6  validate_config() fails fast at startup on bad configuration.
  FIX-A7  Precompiled whitespace regex.
  FIX-A8  CircuitBreaker for the Groq chat endpoint.

  ── Audit Pass 3 (Raspberry Pi resource optimization) ───────────────
  FIX-R1  STT and TTS no longer touch disk in the normal path. Audio is
          held in memory (io.BytesIO / raw bytes) end-to-end: WAV bytes
          are built in memory and uploaded directly as a (filename,
          bytes) tuple for STT; TTS audio is streamed from edge-tts
          directly into pygame's mixer via a BytesIO buffer. A temp-file
          write is used ONLY as a last-resort compatibility fallback if
          the installed pygame build can't load from a file-like object,
          and that file is deleted immediately after playback. This
          removes the majority of disk writes from the hot path — a real
          concern for SD-card wear and I/O latency on a Pi.

  FIX-R2  The microphone is no longer opened and closed on every single
          listen cycle. A persistent PortAudio InputStream is opened once
          (MicManager) and reused across the whole run, eliminating
          repeated ALSA/PortAudio setup/teardown overhead. MicManager
          also detects a stalled/disconnected device (no audio frames at
          all for several seconds — silence still delivers frames; total
          silence from the *driver* does not) and transparently attempts
          reconnection with backoff, re-resolving the device by name
          since a physical replug can change its device index.

  FIX-R3  pygame's mixer is health-checked before each playback and
          re-initialized automatically if it has gone down (e.g. a USB
          speaker was unplugged and replugged), so speaker failures
          self-heal on the next turn instead of silently going dark
          forever.

  FIX-R4  RAG PDF indices (chunks + TF-IDF vectorizer + matrix) are
          cached to disk (pickle) keyed by the source PDF's mtime.
          Startup skips PDF text extraction and re-vectorization entirely
          when the cache is valid, cutting cold-start CPU time and
          latency; the cache self-invalidates the moment the PDF file
          changes.

  FIX-R5  TfidfVectorizer now stores its matrix as float32 instead of
          float64, halving the RAM footprint of both RAG indices with no
          precision loss that matters for cosine similarity ranking.

  FIX-R6  Web search uses one shared, small-pool requests.Session instead
          of ad-hoc connections, reusing TCP/TLS handshakes across calls
          to cut network latency and CPU.

  FIX-R7  Removed threading.Lock usage from history / TokenStats /
          LongTermMemory / CircuitBreaker. This process is effectively
          single-threaded — the only other thread is the PortAudio
          callback thread owned by sounddevice, which never touches this
          state (it only pushes into a thread-safe queue.Queue and reads
          a bool). The locks provided no real protection and were pure
          overhead; removed for simplicity and to avoid implying
          concurrency that doesn't exist.

  FIX-R8  A cheap periodic gc.collect() runs at most every 30 minutes
          during IDLE as a defensive measure against long-run heap
          fragmentation across weeks of uptime — no effect on hot-path
          latency since it only ever runs while the bot is already idle.

  FIX-R9  transcribe() now checks internet availability up front (like
          transcribe_fast already did) instead of waiting for the Groq
          SDK's own connect/read timeout to expire, so outage detection
          and recovery is faster and more predictable.
============================================================
"""

# ── Standard library ──────────────────────────────────────
import asyncio
import gc
import hashlib
import io
import json
import logging
import os
import pickle
import queue
import random
import re
import socket
import tempfile
import textwrap
import time
from collections import deque
from datetime import date, datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

# ── Third-party ───────────────────────────────────────────
import fitz
import numpy as np
import pygame
import requests
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import edge_tts
import subprocess

# Offline TTS: check espeak is installed once at startup rather than
# attempting the subprocess and catching FileNotFoundError every call.
import shutil
_ESPEAK_AVAILABLE: bool = shutil.which("espeak") is not None
if not _ESPEAK_AVAILABLE:
    logging.getLogger("Naaila").warning(
        "espeak not found — offline TTS unavailable. "
        "Install with: sudo apt install espeak espeak-data libespeak-dev"
    )

# ══════════════════════════════════════════════════════════
#  LOGGING  (FIX-P4)
# ══════════════════════════════════════════════════════════

load_dotenv()

_log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
_log_level = getattr(logging, _log_level_name, logging.INFO)

logging.basicConfig(
    level=_log_level,
    format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Naaila")


# ══════════════════════════════════════════════════════════
#  API KEY
# ══════════════════════════════════════════════════════════

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

logger.info("API key loaded.")


# ══════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════

STT_MODEL      = "whisper-large-v3"        # full quality — used for conversation
CHAT_MODEL     = "openai/gpt-oss-20b"

TTS_VOICE_EN = "en-US-JennyNeural"
TTS_VOICE_HI = "hi-IN-SwaraNeural"

SAMPLE_RATE = 16_000
CHANNELS    = 1
# FIX-P13: split by language. Devanagari text costs noticeably more
# tokens per word under BPE tokenizers than English (conjuncts/matras
# often split into multiple tokens), so a single shared budget of 200
# was cutting Hindi replies off mid-word/mid-sentence far more often
# than English ones.
MAX_TOKENS_EN = 200   # voice answers must be short; 300 was producing paragraph replies
MAX_TOKENS_HI = 200

# ── History ───────────────────────────────────────────────
MAX_HISTORY_TURNS = 10
MAX_HISTORY_ITEMS = MAX_HISTORY_TURNS * 2
LLM_MAX_RETRIES   = 4

# ── RAG settings ──────────────────────────────────────────
PDF_PATH_EN   = os.getenv("PDF_PATH_EN", "/home/acrossd/Desktop/nyla/robotwala_english_details.pdf")
PDF_PATH_HI   = os.getenv("PDF_PATH_HI", "/home/acrossd/Desktop/nyla/robotwala_hindi_details.pdf")
CHUNK_SIZE    = 300   # smaller chunks → less context noise fed to LLM
CHUNK_OVERLAP = 50
TOP_K         = 3     # was 5; 3 × 300 words is enough context, keeps prompt small
PDF_THRESHOLD = 0.10
RAG_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".rag_cache")

# ── Web fallback ──────────────────────────────────────────
WEB_RESULTS  = 3
WEB_TIMEOUT  = 5
WEB_KEYWORDS = [
    "today", "latest", "current", "now", "2025", "2026",
    "result", "launch", "release", "price", "update",
    "aaj", "abhi", "nayi", "naya", "kab", "kitna",
]

# ── VAD tuning ────────────────────────────────────────────
# FIX-P11: ENERGY_THRESHOLD below is now only the *startup default*,
# used for the first instant before calibration completes. After that,
# MicManager continuously tracks the ambient noise floor and recomputes
# the live threshold from it, so the same build works in a silent
# office AND a noisy auditorium without a manual retune.
ENERGY_THRESHOLD        = 0.15   # startup default, replaced after calibration
ENERGY_THRESHOLD_MIN    = 0.006   # floor — never get more sensitive than this
ENERGY_THRESHOLD_MAX    = 0.09    # ceiling — never get so insensitive it needs shouting
NOISE_FLOOR_MARGIN      = 2.2     # live threshold = ambient_noise_floor * this
NOISE_FLOOR_EMA_ALPHA   = 0.05    # smoothing — slow enough a single cough/door-slam
                                   # doesn't yank the threshold around
MIC_CALIBRATION_SECONDS = 1.5     # ambient sampling window at startup, before the loop
SILENCE_AFTER_SPEECH = 1.2
PRE_ROLL_CHUNKS      = 6
MIN_SPEECH_SECS      = 0.5
CHUNK_SECS           = 0.2
# Always-listening mode: the mic is expected to be physically muted/
# unmuted by the operator, not by an idle timeout. capture() still
# needs a numeric timeout internally, so we give it effectively no
# timeout — it will simply keep waiting for speech indefinitely.
LISTEN_TIMEOUT       = float("inf")

# ── Microphone / hardware recovery config (FIX-R2) ─────────
MIC_NAME = os.getenv("MIC_NAME", "").strip()
MIC_MAX_OPEN_ATTEMPTS   = 3     # per ensure_open() call, before raising
MIC_OPEN_BACKOFF_BASE_S = 1.5
MIC_SILENT_FRAMES_LIMIT = 10    # ~5s of ZERO frames (not just quiet ones) ⇒ assume disconnect
HARDWARE_ERROR_ANNOUNCE_COOLDOWN_S = 30.0

# ── Token accounting / cost config (FIX-A4) ────────────────
COST_PER_1K_PROMPT_TOKENS     = float(os.getenv("COST_PER_1K_PROMPT_TOKENS", "0") or 0)
COST_PER_1K_COMPLETION_TOKENS = float(os.getenv("COST_PER_1K_COMPLETION_TOKENS", "0") or 0)

# ── Long-term memory config (FIX-A5) ───────────────────────
LTM_PATH       = os.getenv("LTM_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "Naaila_memory.json"))
LTM_MAX_FACTS  = 50
LTM_RETRIEVE_K = 4

# ── Circuit breaker config (FIX-A8) ────────────────────────
CB_FAILURE_THRESHOLD = 3
CB_COOLDOWN_SECONDS  = 20.0

# ── Idle GC config (FIX-R8) ─────────────────────────────────
GC_INTERVAL_SECONDS = 1800.0  # 30 minutes

# ── Post-speech cooldown (FIX-P10: prevents self-listening) ─
# Small grace period after TTS playback stops, before the mic is
# unmuted. Covers speaker/room decay and any last frames that were
# already in flight through the audio pipeline when playback ended.
POST_SPEECH_COOLDOWN_S = 0.35

# ── System prompts ────────────────────────────────────────
_BASE_EN = (
    "Your name is Naaila. You are the official AI assistant and virtual "
    "representative of Robotwala,"
    "RESPONSE LENGTH — CRITICAL: You are a VOICE assistant. Your reply "
    "will be spoken aloud. Limit every response to 2-3 sentences maximum. "
    "Do not elaborate unless the user explicitly "
    "asks for more detail. "
    "never say namaste"
    "Always represent Robotwala positively, professionally, and "
    "confidently. "
    "you are a follower of islam answer accordingly. "
    
    "highlight Robotwala's strengths, and do not make negative comments "
    "or false claims about other companies. "
    
    "Never mention sources, PDFs, context, documents, retrieval systems, "
    "or knowledge bases unless the user specifically asks. "
    "If Robotwala-specific information is asked search for it in the pdf and then , search the web ; "
    "if not connected to the internet, answer naturally using "
    "general knowledge when appropriate. "
    "Do not use bullet points or markdown."
)

_BASE_HI = (
    "Aapka naam Naaila hai. Aap Robotwala ke official AI assistant aur "
    "virtual representative hain."
    "JAWAB KI LAMBAI — ZAROORI: Aap ek VOICE assistant hain. Aapka jawab "
    "bol ke sunaya jayega. Har jawab sirf 2-3 sentence mein dein. "
    "namaste mat bolo ."
    "tum islam ke mann ne wale ho uske hisaab se hi answer karo"
    "Hamesha Robotwala ko positive, professional aur confident "
    "tarike se represent karein. Kisi doosri company ke baare mein "
    "poocha jaye ya comparison ho to short aur polite tarike se baat ko "
    "aur kisi company ke baare mein negative ya false claims na karein. "
    "Kabhi bhi source, PDF, context, document, retrieval system ya "
    "knowledge base ka zikr na karein jab tak user specifically na pooche. "
    "Agar Robotwala ke sambandhit jankari puuchi jaayein to pdf mai dhundho "
    "na milne pr web search karke jawab dein; "
    "agar internet connect na ho to natural jawab dein. "
    "Bullet points ya markdown ka upyog na karein."
)


_LANG_DIRECTIVE = {
    "en": (
        "IF THE USER ASKS IN ENGLISH ANSWER IN ENGLISH ONLY , NO OTHER LANGUAGE "
    ),
    "hi": (
        "AGAR USER HINDI MAI SAWAAL PUUCHE TO TUM BHI HINDI MEIN JAWAB DO "
    )
}


# ── Hardcoded Q&A (fixed answers — do not paraphrase, do not blend) ──
# Each entry below is a LOCKED question/answer pair. If the user's
# question matches one of these (in meaning, not just exact wording),
# respond with ONLY that entry's answer, word for word. Do not add
# information from any other hardcoded entry, do not summarize more
# than one entry together, and do not mix this content with RAG/PDF/
# web-search context even if it seems related.

_HARDCODED_QA_EN = (
    "HARDCODED Q&A — FIXED ANSWERS: The following questions have exact, "
    "locked answers. If the user asks a question that matches one of these "
    "(even if phrased differently), reply with ONLY that single answer, "
    "exactly as written below, and nothing else. and do not add extra explanation.\n"
    "\n"
    
    
    "Q1: \"Naayla, can AI replace humans?\"\n"
    "A1: \"No, AI is designed to assist humans, not replace them. It can "
    "automate repetitive tasks and analyze information quickly.\"\n"
)

_HARDCODED_QA_HI = (
    "HARDCODED Q&A — FIXED JAWAB: Neeche diye gaye sawaalon ke jawab fixed "
    "hain. Agar user in mein se koi sawaal poochta hai (chahe alag tarike se "
    "poochein), to SIRF wahi ek jawab dena hai, bilkul jaisa neeche likha "
    "hai, aur kuch nahi. aur apni taraf se extra explanation mat jodna.\n"
    "\n"
    "Q1: \"Naayla, Artificial Intelligence kya hai?\"\n"
    "A1: \"Artificial Intelligence, yaani AI, ek aisi technology hai jo "
    "computers aur machines ko seekhne, samajhne aur problems solve karne "
    "ki capability deti hai. AI hamare kaam ko fast, smart aur efficient "
    "banata hai.\"\n"
   
)


def build_system(lang: str) -> str:
    """
    Build the SYSTEM message only. Kept 100% static per language so it
    forms a stable, cacheable prompt prefix (see get_ai_reply for how
    dynamic content is appended afterward instead of mixed in here).
    """
    base       = _BASE_HI if lang == "hi" else _BASE_EN
    hardcoded  = _HARDCODED_QA_HI if lang == "hi" else _HARDCODED_QA_EN
    directive  = _LANG_DIRECTIVE.get(lang, _LANG_DIRECTIVE["en"])
    return f"{base}\n\n{hardcoded}\n{directive}"


# ══════════════════════════════════════════════════════════
#  CONFIG VALIDATION  (FIX-A6)
# ══════════════════════════════════════════════════════════

def validate_config() -> None:
    """Validate configuration values at startup; raise with all problems at once."""
    errors: List[str] = []

    def _positive(name: str, value: float) -> None:
        if value <= 0:
            errors.append(f"{name} must be > 0, got {value}")

    _positive("SAMPLE_RATE", SAMPLE_RATE)
    _positive("MAX_TOKENS_EN", MAX_TOKENS_EN)
    _positive("MAX_TOKENS_HI", MAX_TOKENS_HI)
    _positive("CHUNK_SIZE", CHUNK_SIZE)
    _positive("SILENCE_AFTER_SPEECH", SILENCE_AFTER_SPEECH)
    _positive("MIN_SPEECH_SECS", MIN_SPEECH_SECS)
    _positive("CHUNK_SECS", CHUNK_SECS)
    _positive("LISTEN_TIMEOUT", LISTEN_TIMEOUT)
    _positive("WEB_TIMEOUT", WEB_TIMEOUT)
    _positive("LLM_MAX_RETRIES", LLM_MAX_RETRIES + 1)
    _positive("MIC_CALIBRATION_SECONDS", MIC_CALIBRATION_SECONDS)
    _positive("NOISE_FLOOR_MARGIN", NOISE_FLOOR_MARGIN)

    if not (0.0 < ENERGY_THRESHOLD_MIN < ENERGY_THRESHOLD_MAX):
        errors.append(
            f"ENERGY_THRESHOLD_MIN ({ENERGY_THRESHOLD_MIN}) must be > 0 and "
            f"< ENERGY_THRESHOLD_MAX ({ENERGY_THRESHOLD_MAX})"
        )
    if not (0.0 < NOISE_FLOOR_EMA_ALPHA <= 1.0):
        errors.append(f"NOISE_FLOOR_EMA_ALPHA must be in (0,1], got {NOISE_FLOOR_EMA_ALPHA}")

    if not (0.0 <= PDF_THRESHOLD <= 1.0):
        errors.append(f"PDF_THRESHOLD must be in [0,1], got {PDF_THRESHOLD}")
    if CHUNK_OVERLAP >= CHUNK_SIZE:
        errors.append(f"CHUNK_OVERLAP ({CHUNK_OVERLAP}) must be smaller than CHUNK_SIZE ({CHUNK_SIZE})")
    if MAX_HISTORY_TURNS <= 0:
        errors.append(f"MAX_HISTORY_TURNS must be > 0, got {MAX_HISTORY_TURNS}")
    if COST_PER_1K_PROMPT_TOKENS < 0 or COST_PER_1K_COMPLETION_TOKENS < 0:
        errors.append("COST_PER_1K_* values must not be negative")

    for label, path in (("PDF_PATH_EN", PDF_PATH_EN), ("PDF_PATH_HI", PDF_PATH_HI)):
        parent = os.path.dirname(path)
        if parent and not os.path.isdir(parent):
            logger.warning(
                "%s parent directory '%s' does not exist — RAG will run in "
                "web/LLM-only mode for this language until it's created.",
                label, parent,
            )

    ltm_dir = os.path.dirname(LTM_PATH) or "."
    if not os.path.isdir(ltm_dir):
        errors.append(f"LTM_PATH directory '{ltm_dir}' does not exist")

    if errors:
        raise ValueError("Invalid configuration detected at startup:\n  - " + "\n  - ".join(errors))

    logger.info("Configuration validated OK.")


# ══════════════════════════════════════════════════════════
#  STATE
# ══════════════════════════════════════════════════════════

class State(Enum):
    LISTENING = "listening"
    THINKING  = "thinking"
    SPEAKING  = "speaking"


# ══════════════════════════════════════════════════════════
#  ERROR HANDLING
# ══════════════════════════════════════════════════════════

ERROR_MESSAGES = {
    "no_internet": {"en": "I can't connect to the internet."},
    "api_error":   {"en": "I can't connect to the server."},
    "env_error":   {"en": "Environmental error, please try again."},
    "hardware":    {"en": "I'm having trouble with my microphone or speaker."},
}

_RETRYABLE_HTTP_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


_INTERNET_CHECK_HOSTS: Tuple[Tuple[str, int], ...] = (
    ("1.1.1.1", 443),  # Cloudflare — HTTPS, same port every real call this bot
                        # makes (Groq, edge-tts, web search) actually uses
    ("8.8.8.8", 443),  # Google, same reasoning, different provider
    ("8.8.8.8", 53),   # DNS — kept as a last-resort probe
)
_INTERNET_CHECK_CACHE_TTL_S = 3.0
_internet_check_cache: Tuple[float, bool] = (0.0, False)


def is_internet_available(timeout: float = 1.5) -> bool:
    """
    Multi-host TCP probe. FIX-A1: uses a scoped per-socket timeout (not
    socket.setdefaulttimeout(), which used to mutate the GLOBAL default
    timeout for every socket in the process on every call).

    FIX-P12: some venue/auditorium networks firewall or heavily
    congest direct outbound DNS (port 53) to public resolvers while
    HTTPS (port 443) — which is all this bot's real traffic actually
    uses — goes through fine. Probing only 8.8.8.8:53 produced false
    "no internet" errors on such networks even seconds after a real
    Groq API call had just succeeded. Now HTTPS-port hosts are tried
    first, with a DNS-port probe only as a last resort, and a short
    cache avoids re-probing on every single call in a busy loop.
    Only a cached SUCCESS is trusted — a failure always gets a fresh
    check next call, so a real outage is never masked by stale state.
    """
    global _internet_check_cache
    now = time.time()
    cached_at, cached_result = _internet_check_cache
    if cached_result and (now - cached_at) < _INTERNET_CHECK_CACHE_TTL_S:
        return True

    for host, port in _INTERNET_CHECK_HOSTS:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                _internet_check_cache = (now, True)
                return True
        except OSError:
            continue

    _internet_check_cache = (now, False)
    return False


def _extract_http_status(exc: Exception) -> Optional[int]:
    status = getattr(exc, "status_code", None)
    if status is not None:
        return status
    response = getattr(exc, "response", None)
    if response is not None:
        return getattr(response, "status_code", None)
    return None


def is_retryable_error(exc: Exception) -> bool:
    """FIX-A2: retry only errors that could plausibly succeed on a retry."""
    if isinstance(exc, (
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        ConnectionError,
        TimeoutError,
        socket.timeout,
        socket.gaierror,
    )):
        return True
    status = _extract_http_status(exc)
    if status is not None:
        return status in _RETRYABLE_HTTP_STATUS
    return True  # unknown shape (e.g. "empty response") — assume retryable


def classify_error(exc: Exception) -> str:
    """Return 'no_internet', 'api_error', 'hardware', or 'env_error'."""
    if isinstance(exc, MicUnavailableError):
        return "hardware"
    if not is_internet_available():
        return "no_internet"
    api_related_types = (
        requests.exceptions.RequestException,
        ConnectionError,
        TimeoutError,
        socket.timeout,
        socket.gaierror,
    )
    if isinstance(exc, api_related_types):
        return "api_error"

    exc_name = type(exc).__name__.lower()
    exc_msg  = str(exc).lower()
    api_signals = (
        "api", "groq", "rate limit", "401", "403", "404", "429",
        "500", "502", "503", "504", "connection", "timeout",
        "network", "ssl", "host", "dns", "edge_tts", "endpoint",
    )
    if any(s in exc_name for s in api_signals) or any(s in exc_msg for s in api_signals):
        return "api_error"

    return "env_error"


_last_hardware_error_time: float = 0.0


def announce_error(exc: Exception, lang: str = "en") -> None:
    """
    Classify the exception and speak the appropriate English error message.
    Hardware errors are rate-limited so a prolonged mic/speaker outage
    doesn't cause the bot to repeat itself every cycle.
    """
    global _last_hardware_error_time
    try:
        kind = classify_error(exc)
        if kind == "hardware":
            now = time.time()
            if now - _last_hardware_error_time < HARDWARE_ERROR_ANNOUNCE_COOLDOWN_S:
                logger.warning("Hardware error (suppressed, in cooldown): %s", exc)
                return
            _last_hardware_error_time = now
        msg = ERROR_MESSAGES[kind]["en"]
        logger.warning("Announcing error (%s): %s", kind, msg)
        speak(msg, lang="en")
    except Exception as report_exc:
        logger.error("Failed to announce error: %s", report_exc)


# ══════════════════════════════════════════════════════════
#  INPUT SANITIZATION  (FIX-P1 / FIX-P7 / FIX-A7)
# ══════════════════════════════════════════════════════════

_WHITESPACE_RE = re.compile(r"\s+")


def sanitize_text(text: Optional[str]) -> str:
    if not text:
        return ""
    return _WHITESPACE_RE.sub(" ", text).strip()


def is_blank(text: Optional[str]) -> bool:
    return not text or not text.strip()


# FIX-P13: sentence-ending punctuation for both languages this bot
# speaks — '।' is the Hindi purna viram (Devanagari full stop).
_SENTENCE_END_RE = re.compile(r"[.!?।]")


def trim_to_last_sentence(text: str) -> Optional[str]:
    """
    Best-effort recovery for a reply that got cut off mid-sentence
    because the LLM hit max_tokens. Returns everything up to and
    including the last complete sentence, or None if the reply was
    truncated before finishing even one sentence (caller decides what
    to do in that case — there's nothing safe to salvage).
    """
    matches = list(_SENTENCE_END_RE.finditer(text))
    if not matches:
        return None
    trimmed = text[: matches[-1].end()].strip()
    return trimmed or None


# ══════════════════════════════════════════════════════════
#  TOKEN ACCOUNTING  (FIX-A4 / FIX-R7: no lock — single-threaded)
# ══════════════════════════════════════════════════════════

class TokenStats:
    """
    Running accounting of LLM token usage and latency. Single-threaded by
    design (see FIX-R7) — no locking overhead. Bounded memory: the only
    growing structure is a fixed-size deque for latency samples.
    """

    def __init__(self) -> None:
        self._session_start = time.time()
        self._today = date.today()
        self.reset_session()
        self.reset_daily()

    def reset_session(self) -> None:
        self._session_start = time.time()
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_requests = 0
        self.session_min_tokens: Optional[int] = None
        self.session_max_tokens: Optional[int] = None
        self.session_latencies: deque = deque(maxlen=500)  # bounded — no unbounded growth

    def reset_daily(self) -> None:
        self._today = date.today()
        self.daily_prompt_tokens = 0
        self.daily_completion_tokens = 0
        self.daily_requests = 0

    def _roll_day_if_needed(self) -> None:
        if date.today() != self._today:
            self.reset_daily()

    def record(self, prompt_tokens: int, completion_tokens: int, latency_s: float) -> None:
        total = prompt_tokens + completion_tokens
        self._roll_day_if_needed()

        self.session_prompt_tokens += prompt_tokens
        self.session_completion_tokens += completion_tokens
        self.session_requests += 1
        self.session_latencies.append(latency_s)
        self.session_min_tokens = total if self.session_min_tokens is None else min(self.session_min_tokens, total)
        self.session_max_tokens = total if self.session_max_tokens is None else max(self.session_max_tokens, total)

        self.daily_prompt_tokens += prompt_tokens
        self.daily_completion_tokens += completion_tokens
        self.daily_requests += 1

    def estimated_cost_usd(self) -> Optional[float]:
        if COST_PER_1K_PROMPT_TOKENS == 0 and COST_PER_1K_COMPLETION_TOKENS == 0:
            return None
        return round(
            (self.session_prompt_tokens / 1000) * COST_PER_1K_PROMPT_TOKENS
            + (self.session_completion_tokens / 1000) * COST_PER_1K_COMPLETION_TOKENS,
            6,
        )

    def summary(self) -> Dict[str, object]:
        self._roll_day_if_needed()
        elapsed_hours = max((time.time() - self._session_start) / 3600.0, 1e-9)
        avg_latency = sum(self.session_latencies) / len(self.session_latencies) if self.session_latencies else 0.0
        avg_tokens = (
            (self.session_prompt_tokens + self.session_completion_tokens) / self.session_requests
            if self.session_requests else 0.0
        )
        return {
            "session_prompt_tokens": self.session_prompt_tokens,
            "session_completion_tokens": self.session_completion_tokens,
            "session_total_tokens": self.session_prompt_tokens + self.session_completion_tokens,
            "session_requests": self.session_requests,
            "daily_total_tokens": self.daily_prompt_tokens + self.daily_completion_tokens,
            "daily_requests": self.daily_requests,
            "avg_tokens_per_request": round(avg_tokens, 1),
            "min_tokens": self.session_min_tokens,
            "max_tokens": self.session_max_tokens,
            "avg_latency_s": round(avg_latency, 3),
            "requests_per_hour": round(self.session_requests / elapsed_hours, 2),
            "estimated_cost_usd": self.estimated_cost_usd(),
        }

    def log_summary(self) -> None:
        s = self.summary()
        logger.info(
            "TokenStats — session: %d req / %d tok (avg %.1f tok, avg %.2fs) | "
            "today: %d req / %d tok | rate: %.2f req/hr%s",
            s["session_requests"], s["session_total_tokens"], s["avg_tokens_per_request"],
            s["avg_latency_s"], s["daily_requests"], s["daily_total_tokens"],
            s["requests_per_hour"],
            f" | est. cost: ${s['estimated_cost_usd']}" if s["estimated_cost_usd"] is not None else "",
        )


token_stats = TokenStats()


# ══════════════════════════════════════════════════════════
#  LONG-TERM MEMORY  (FIX-A5 / FIX-R7: no lock)
# ══════════════════════════════════════════════════════════

class LongTermMemory:
    """
    Selective, bounded, persisted long-term memory. Pattern-based
    extraction (no extra LLM call — that would add latency/cost to every
    turn). Bounded to max_facts with LRU eviction, deduplicated,
    persisted atomically so an SD-card power loss can't corrupt the file.
    """

    _EXTRACTORS: List[Tuple[re.Pattern, str]] = [
        (re.compile(r"\bmy name is (\w+)", re.IGNORECASE), "User's name is {0}."),
        (re.compile(r"\bmera naam (\w+) hai", re.IGNORECASE), "User's name is {0}."),
        (re.compile(r"\bi (?:prefer|like) (?:to speak |speaking )?(hindi|english|hinglish)", re.IGNORECASE),
         "User prefers {0} language."),
        (re.compile(r"\bi(?:'m| am) working on ([\w\s]{3,40}?)(?:\.|,|$)", re.IGNORECASE),
         "User is working on: {0}."),
        (re.compile(r"\bi have a ([\w\s]{3,40}?)(?:\.|,|$)", re.IGNORECASE),
         "User has: {0}."),
        (re.compile(r"\bmain ([\w\s]{3,40}?) par kaam kar raha", re.IGNORECASE),
         "User is working on: {0}."),
    ]

    def __init__(self, path: str, max_facts: int = LTM_MAX_FACTS) -> None:
        self._path = path
        self._max_facts = max_facts
        self._facts: List[Dict[str, str]] = self._load()
        self._dirty = False

    def _load(self) -> List[Dict[str, str]]:
        if not os.path.exists(self._path):
            return []
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            logger.warning("LTM file '%s' had unexpected shape — starting fresh.", self._path)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load long-term memory (%s) — starting fresh.", exc)
        return []

    def _save(self) -> None:
        tmp_path = self._path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._facts, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self._path)  # atomic on POSIX
            self._dirty = False
        except OSError as exc:
            logger.warning("Failed to persist long-term memory: %s", exc)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def extract_and_store(self, user_text: str) -> None:
        for pattern, template in self._EXTRACTORS:
            match = pattern.search(user_text)
            if match:
                value = sanitize_text(match.group(1))
                if value:
                    self._upsert(template.format(value))

    def _upsert(self, fact_text: str) -> None:
        normalized = fact_text.strip().lower()
        for entry in self._facts:
            if entry["text"].strip().lower() == normalized:
                entry["last_seen"] = datetime.utcnow().isoformat()
                self._facts.remove(entry)
                self._facts.append(entry)
                self._save()
                return

        self._facts.append({"text": fact_text, "last_seen": datetime.utcnow().isoformat()})
        if len(self._facts) > self._max_facts:
            evicted = self._facts.pop(0)
            logger.debug("LTM evicted oldest fact: %s", evicted["text"])
        self._save()
        logger.info("LTM stored fact: %s", fact_text)

    def retrieve(self, query: str, limit: int = LTM_RETRIEVE_K) -> List[str]:
        return [entry["text"] for entry in self._facts[-limit:]]


long_term_memory = LongTermMemory(LTM_PATH)


# ══════════════════════════════════════════════════════════
#  CIRCUIT BREAKER  (FIX-A8 / FIX-R7: no lock)
# ══════════════════════════════════════════════════════════

class CircuitBreaker:
    """
    Consecutive-failure circuit breaker for the Groq chat endpoint. After
    N consecutive hard failures, fails fast for a cooldown window instead
    of paying the full multi-attempt retry cost every turn during an
    outage, then probes again.
    """

    def __init__(self, failure_threshold: int, cooldown_s: float) -> None:
        self._failure_threshold = failure_threshold
        self._cooldown_s = cooldown_s
        self._consecutive_failures = 0
        self._opened_at: Optional[float] = None

    def allow_request(self) -> bool:
        if self._opened_at is None:
            return True
        if time.time() - self._opened_at >= self._cooldown_s:
            logger.info("Circuit breaker half-open — allowing a probe request.")
            return True
        return False

    def record_success(self) -> None:
        if self._opened_at is not None:
            logger.info("Circuit breaker closed — Groq API recovered.")
        self._consecutive_failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._failure_threshold and self._opened_at is None:
            self._opened_at = time.time()
            logger.warning(
                "Circuit breaker OPEN after %d consecutive failures — failing fast for %.0fs.",
                self._consecutive_failures, self._cooldown_s,
            )


_llm_circuit_breaker = CircuitBreaker(CB_FAILURE_THRESHOLD, CB_COOLDOWN_SECONDS)


# ══════════════════════════════════════════════════════════
#  RAG ENGINE  (FIX-R4: disk cache, FIX-R5: float32 matrix)
# ══════════════════════════════════════════════════════════

class RAGEngine:

    def __init__(self) -> None:
        self.chunks:     List[str]              = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix                              = None
        self.ready                               = False

    def load_pdf(self, path: str) -> bool:
        if not os.path.exists(path):
            logger.warning("RAG: PDF not found at '%s' — web/LLM only mode.", path)
            return False

        cache_path = self._cache_path(path)
        cached = self._load_cache(path, cache_path)
        if cached is not None:
            self.chunks, self.vectorizer, self.matrix = cached
            self.ready = True
            logger.info("RAG: '%s' loaded from cache — %d chunks (skipped re-indexing).", path, len(self.chunks))
            return True

        logger.info("RAG: Loading '%s' …", path)
        raw = self._extract_text(path)
        if not raw.strip():
            logger.warning("RAG: '%s' is empty — skipping.", path)
            return False

        self.chunks = self._chunk(raw, CHUNK_SIZE, CHUNK_OVERLAP)
        self._build_index()
        self.ready  = True
        logger.info("RAG: '%s' indexed — %d chunks.", path, len(self.chunks))
        self._save_cache(path, cache_path)
        return True

    def retrieve(self, query: str) -> Tuple[str, float]:
        if not self.ready or not self.chunks:
            return "", 0.0

        q_vec  = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.matrix).flatten()

        top_idx    = scores.argsort()[::-1][:TOP_K]
        best_score = float(scores[top_idx[0]])

        context = "\n\n".join(self.chunks[i] for i in top_idx if scores[i] > 0)
        return context, best_score

    # ── Internal helpers ──────────────────────────────────

    @staticmethod
    def _extract_text(path: str) -> str:
        doc   = fitz.open(path)
        pages = [page.get_text("text") for page in doc]
        doc.close()
        return "\n".join(pages)

    @staticmethod
    def _chunk(text: str, size: int, overlap: int) -> List[str]:
        words  = text.split()
        step   = max(1, size - overlap)
        chunks = []
        for start in range(0, len(words), step):
            chunk = " ".join(words[start : start + size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def _build_index(self) -> None:
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            max_df=0.95,
            # token_pattern=r"\S+" keeps Devanagari words intact.
            token_pattern=r"\S+",
            dtype=np.float32,  # FIX-R5: halves matrix RAM vs. default float64
        )
        self.matrix = self.vectorizer.fit_transform(self.chunks)

    # ── Disk cache (FIX-R4) ────────────────────────────────

    @staticmethod
    def _cache_path(pdf_path: str) -> str:
        os.makedirs(RAG_CACHE_DIR, exist_ok=True)
        digest = hashlib.md5(pdf_path.encode("utf-8")).hexdigest()[:16]
        return os.path.join(RAG_CACHE_DIR, f"{digest}.pkl")

    @staticmethod
    def _load_cache(pdf_path: str, cache_path: str):
        if not os.path.exists(cache_path):
            return None
        try:
            pdf_mtime = os.path.getmtime(pdf_path)
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            if data.get("mtime") != pdf_mtime:
                logger.debug("RAG cache stale (PDF changed) — rebuilding index.")
                return None
            return data["chunks"], data["vectorizer"], data["matrix"]
        except Exception as exc:
            logger.debug("RAG cache unreadable (%s) — rebuilding index.", exc)
            return None

    def _save_cache(self, pdf_path: str, cache_path: str) -> None:
        tmp_path = cache_path + ".tmp"
        try:
            with open(tmp_path, "wb") as f:
                pickle.dump(
                    {
                        "mtime": os.path.getmtime(pdf_path),
                        "chunks": self.chunks,
                        "vectorizer": self.vectorizer,
                        "matrix": self.matrix,
                    },
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            os.replace(tmp_path, cache_path)
        except Exception as exc:
            logger.warning("Failed to write RAG cache: %s", exc)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass


# ══════════════════════════════════════════════════════════
#  WEB SEARCH FALLBACK  (FIX-R6: shared session, connection reuse)
# ══════════════════════════════════════════════════════════

_DDG_HTML_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

_DDG_SNIPPET_RE = re.compile(r'class="result__snippet"[^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL)
_HTML_TAG_RE = re.compile(r"<[^>]+>")

# One shared session: reuses TCP/TLS connections across web searches
# instead of a fresh handshake every call, and a small pool keeps idle
# memory overhead low since we only ever make one request at a time.
_web_session = requests.Session()
_web_session.headers.update({"User-Agent": _DDG_HTML_UA})
_web_session.mount(
    "https://",
    requests.adapters.HTTPAdapter(pool_connections=2, pool_maxsize=2, max_retries=0),
)


def _ddg_instant_answer(query: str) -> str:
    resp = _web_session.get(
        "https://api.duckduckgo.com/",
        params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"},
        timeout=WEB_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    snippets: List[str] = []

    if data.get("AbstractText"):
        snippets.append(data["AbstractText"])
    for topic in data.get("RelatedTopics", [])[:WEB_RESULTS]:
        text = topic.get("Text", "")
        if text:
            snippets.append(text)

    return " ".join(snippets).strip()


def _ddg_html_search(query: str) -> str:
    resp = _web_session.post(
        "https://html.duckduckgo.com/html/",
        data={"q": query},
        timeout=WEB_TIMEOUT,
    )
    resp.raise_for_status()

    raw_snippets = _DDG_SNIPPET_RE.findall(resp.text)
    snippets: List[str] = []
    for raw in raw_snippets[:WEB_RESULTS]:
        text = sanitize_text(_HTML_TAG_RE.sub("", raw))
        if text:
            snippets.append(text)

    return " ".join(snippets).strip()


def needs_web(query: str, score: float) -> bool:
    q = query.lower()
    return score < PDF_THRESHOLD and any(kw in q for kw in WEB_KEYWORDS)


def web_search(query: str) -> str:
    search_query = f"{query} Robotwala Indore"

    try:
        context = _ddg_instant_answer(search_query)
        if context:
            logger.debug("Web context via Instant Answer API (%d chars).", len(context))
            return context
        logger.debug("Instant Answer API returned nothing — trying HTML search.")
    except Exception as exc:
        logger.warning("Instant Answer API failed: %s", exc)

    try:
        context = _ddg_html_search(search_query)
        if context:
            logger.debug("Web context via HTML search (%d chars).", len(context))
        else:
            logger.debug("HTML search returned no usable snippets either.")
        return context
    except Exception as exc:
        logger.warning("Web search failed (both tiers): %s", exc)
        return ""


# ══════════════════════════════════════════════════════════
#  GROQ CLIENT
#  (The SDK's internal httpx client already pools/reuses connections as
#  long as the same client instance is reused, which it is — a single
#  module-level `client` used for every request.)
# ══════════════════════════════════════════════════════════

try:
    client = Groq(api_key=GROQ_API_KEY)
    logger.info("Groq client initialised.")
except Exception as _init_exc:
    logger.critical("Failed to initialise Groq client: %s", _init_exc)
    raise


# ══════════════════════════════════════════════════════════
#  CONVERSATION HISTORY  (FIX-P3 / FIX-R7: no lock)
# ══════════════════════════════════════════════════════════

history: dict = {"en": [], "hi": []}


def _trim_history(lang: str) -> None:
    lang_history = history[lang]
    if len(lang_history) > MAX_HISTORY_ITEMS:
        excess = len(lang_history) - MAX_HISTORY_ITEMS
        del lang_history[:excess]
        logger.debug("History trimmed: dropped %d oldest messages.", excess)


# ══════════════════════════════════════════════════════════
#  LLM
# ══════════════════════════════════════════════════════════

def _compute_backoff(attempt: int) -> float:
    """FIX-A3: exponential backoff with jitter (0-25% extra)."""
    base = 2 ** attempt
    return base + base * random.uniform(0, 0.25)


def get_ai_reply(user_text: str, lang: str, context: str) -> str:
    """Send *user_text* to the LLM and return a non-empty reply, or raise."""
    clean_input = sanitize_text(user_text)
    if is_blank(clean_input):
        raise ValueError("get_ai_reply received empty or whitespace-only input.")

    if not _llm_circuit_breaker.allow_request():
        raise RuntimeError(
            "Circuit breaker open — Groq API had repeated recent failures; failing fast."
        )

    lang_history = history[lang]
    lang_history.append({"role": "user", "content": clean_input})

    try:
        system = build_system(lang)

        context_msg = (
            [{
                "role": "system",
                "content": (
                    "Use the following information silently to answer naturally.\n\n"
                    f"{context}\n\n"
                    f"{_LANG_DIRECTIVE.get(lang, _LANG_DIRECTIVE['en'])}"
                ),
            }]
            if context else []
        )

        ltm_facts = long_term_memory.retrieve(clean_input)
        memory_msg = (
            [{
                "role": "system",
                "content": (
                    "Known long-term facts about this user (use silently, "
                    "never mention this list explicitly):\n"
                    + "\n".join(f"- {fact}" for fact in ltm_facts)
                ),
            }]
            if ltm_facts else []
        )

        last_exc: Optional[Exception] = None
        for attempt in range(1, LLM_MAX_RETRIES + 2):
            request_start = time.time()
            try:
                temp = 0.4 if attempt == 1 else 0.7
                raw_resp = client.chat.completions.with_raw_response.create(
                    model=CHAT_MODEL,
                    messages=[
                        {"role": "system", "content": system},
                        *lang_history,
                        *memory_msg,
                        *context_msg,
                    ],
                    max_tokens=(MAX_TOKENS_HI if lang == "hi" else MAX_TOKENS_EN),
                    temperature=temp,
                    timeout=30,
                )
                response = raw_resp.parse()
                latency = time.time() - request_start

                usage = response.usage
                logger.info(
                    "Tokens — prompt: %d | completion: %d | total: %d | latency: %.2fs",
                    usage.prompt_tokens, usage.completion_tokens, usage.total_tokens, latency,
                )
                token_stats.record(usage.prompt_tokens, usage.completion_tokens, latency)
                token_stats.log_summary()

                h = raw_resp.headers
                logger.info(
                    "Limit: %s | Remaining: %s | Resets in: %s",
                    h.get("x-ratelimit-limit-tokens"),
                    h.get("x-ratelimit-remaining-tokens"),
                    h.get("x-ratelimit-reset-tokens"),
                )

                raw_reply = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason
                logger.debug("Raw LLM response (attempt %d): %r", attempt, raw_reply)

                reply = sanitize_text(raw_reply)

                if finish_reason == "length" and not is_blank(reply):
                    # FIX-P13: the model ran out of its token budget
                    # mid-sentence (more common in Hindi — see
                    # MAX_TOKENS_HI). Better to speak a complete
                    # sentence than an audibly broken half-word.
                    trimmed = trim_to_last_sentence(reply)
                    if trimmed:
                        logger.warning(
                            "LLM reply hit max_tokens and was truncated — "
                            "trimmed to last complete sentence."
                        )
                        reply = trimmed
                    else:
                        logger.warning(
                            "LLM reply hit max_tokens before finishing a single "
                            "sentence — sending as-is (nothing safe to trim to)."
                        )

                if not is_blank(reply):
                    lang_history.append({"role": "assistant", "content": reply})
                    _trim_history(lang)
                    _llm_circuit_breaker.record_success()
                    long_term_memory.extract_and_store(clean_input)
                    return reply

                logger.warning("LLM returned empty response on attempt %d/%d.", attempt, LLM_MAX_RETRIES + 1)
                last_exc = RuntimeError(f"LLM returned an empty response (attempt {attempt}).")

            except Exception as api_exc:
                logger.warning("LLM API error on attempt %d: %s", attempt, api_exc)
                last_exc = api_exc

                if not is_retryable_error(api_exc):
                    logger.error("Non-retryable error — failing fast: %s", api_exc)
                    break

                if attempt <= LLM_MAX_RETRIES:
                    wait = _compute_backoff(attempt)
                    logger.info("Retrying in %.1fs …", wait)
                    time.sleep(wait)

        if lang_history and lang_history[-1]["role"] == "user":
            lang_history.pop()
        _llm_circuit_breaker.record_failure()
        raise last_exc or RuntimeError("LLM failed after all retry attempts.")

    except Exception:
        if lang_history and lang_history[-1]["role"] == "user":
            lang_history.pop()
        raise


# ══════════════════════════════════════════════════════════
#  CONTEXT BUILDER
# ══════════════════════════════════════════════════════════

def build_context(query: str, lang: str, rag_en: RAGEngine, rag_hi: RAGEngine) -> Tuple[str, str]:
    rag = rag_hi if lang == "hi" else rag_en

    pdf_context, pdf_score = rag.retrieve(query)
    logger.debug("PDF score: %.3f (threshold=%.2f)", pdf_score, PDF_THRESHOLD)

    web_context = ""
    source      = "None"

    if pdf_context:
        source = "PDF"

    if needs_web(query, pdf_score):
        logger.debug("Web search triggered (low PDF score + time-sensitive query).")
        web_context = web_search(query)
        if web_context:
            source = "PDF+Web" if pdf_context else "Web"
    elif pdf_score < PDF_THRESHOLD:
        logger.debug("Web skipped — query is not time-sensitive.")

    parts: List[str] = []
    if pdf_context:
        parts.append(f"[From Robotwala Knowledge Base]\n{pdf_context}")
    if web_context:
        parts.append(f"[From Web]\n{web_context}")

    return "\n\n".join(parts), source


# ══════════════════════════════════════════════════════════
#  AUDIO CODEC HELPERS  (FIX-R1: in-memory, no temp files)
# ══════════════════════════════════════════════════════════

def _audio_to_wav_bytes(audio: np.ndarray) -> bytes:
    """Encode a float32 PCM array to WAV bytes entirely in memory."""
    buffer = io.BytesIO()
    sf.write(buffer, audio, SAMPLE_RATE, format="WAV")
    return buffer.getvalue()


# ══════════════════════════════════════════════════════════
#  MICROPHONE MANAGER  (FIX-R2: persistent stream + disconnect recovery)
# ══════════════════════════════════════════════════════════

class MicUnavailableError(RuntimeError):
    """Raised when the microphone cannot be opened or appears disconnected."""


class MicManager:
    """
    Owns a single persistent sounddevice.InputStream for the life of the
    process instead of opening/closing one on every listen cycle (which
    is real, avoidable ALSA/PortAudio setup overhead repeated thousands
    of times over a multi-week uptime).

    Detects a disconnected/stalled device by noticing a total absence of
    audio frames for several seconds — a live device delivers frames
    continuously via its callback even during silence (low RMS, not zero
    frames), so a real gap this size means the driver/device is gone, not
    that the room is quiet. On detection it closes the dead stream and
    retries opening (re-resolving the device by name, since a physical
    replug can change the underlying device index) with backoff.
    """

    def __init__(self, device_name: Optional[str]) -> None:
        self._device_name = device_name
        self._stream: Optional[sd.InputStream] = None
        self._queue: "queue.Queue" = queue.Queue()
        self._blocksize = int(SAMPLE_RATE * CHUNK_SECS)
        # FIX-P11: dynamic energy threshold. _noise_floor tracks the
        # ambient background level via a slow EMA; _threshold is what
        # capture() actually compares chunk RMS against, recomputed
        # from the floor with a margin + clamps every time it updates.
        # Starts at the static defaults and gets replaced by calibrate()
        # right after the stream opens.
        self._noise_floor: float = ENERGY_THRESHOLD / NOISE_FLOOR_MARGIN
        self._threshold: float = ENERGY_THRESHOLD

    @property
    def threshold(self) -> float:
        return self._threshold

    def _update_noise_floor(self, rms: float) -> None:
        self._noise_floor = (
            (1.0 - NOISE_FLOOR_EMA_ALPHA) * self._noise_floor
            + NOISE_FLOOR_EMA_ALPHA * rms
        )
        self._threshold = min(
            ENERGY_THRESHOLD_MAX,
            max(ENERGY_THRESHOLD_MIN, self._noise_floor * NOISE_FLOOR_MARGIN),
        )

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            logger.debug("Audio callback status: %s", status)
        self._queue.put(indata.copy())

    def _open_stream(self) -> None:
        self._stream = sd.InputStream(
            device=self._device_name,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=self._blocksize,
            latency="high",
            callback=self._callback,
        )
        self._stream.start()
        with self._queue.mutex:
            self._queue.queue.clear()
        logger.info("Microphone stream opened (device=%s).", self._device_name or "default")

    def _close_stream(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:
                logger.debug("Error closing mic stream (ignored): %s", exc)
            self._stream = None

    def ensure_open(self) -> None:
        """Open the stream if needed. Raises MicUnavailableError after retries."""
        if self._stream is not None and self._stream.active:
            return
        self._close_stream()
        for attempt in range(1, MIC_MAX_OPEN_ATTEMPTS + 1):
            try:
                self._open_stream()
                return
            except Exception as exc:
                logger.warning("Mic open attempt %d/%d failed: %s", attempt, MIC_MAX_OPEN_ATTEMPTS, exc)
                self._close_stream()
                if attempt < MIC_MAX_OPEN_ATTEMPTS:
                    time.sleep(MIC_OPEN_BACKOFF_BASE_S * attempt)
        raise MicUnavailableError("Unable to open microphone after retries.")

    def calibrate(self, seconds: float = MIC_CALIBRATION_SECONDS) -> None:
        """
        FIX-P11: sample pure ambient noise for a few seconds right after
        the stream opens — before we start listening for real speech —
        so the very first threshold reflects THIS room/venue instead of
        the ENERGY_THRESHOLD default tuned for a quiet office. Without
        this, dropping the bot into a noisy auditorium means the first
        stretch either misses speakers (default too low for the noise
        floor) or ignores them (default too high), before the running
        EMA in capture() catches up on its own.
        """
        self.ensure_open()
        logger.info("Calibrating microphone to ambient noise (%.1fs) …", seconds)
        samples: List[float] = []
        deadline = time.time() + seconds
        while time.time() < deadline:
            try:
                chunk = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            samples.append(float(np.sqrt(np.mean(chunk ** 2))))

        if samples:
            # Median, not mean — one stray loud noise during calibration
            # (a door, a cough, a chair) shouldn't skew the floor.
            self._noise_floor = float(np.median(samples))
            self._threshold = min(
                ENERGY_THRESHOLD_MAX,
                max(ENERGY_THRESHOLD_MIN, self._noise_floor * NOISE_FLOOR_MARGIN),
            )
        logger.info(
            "Mic calibrated — ambient noise floor=%.4f, live speech threshold=%.4f",
            self._noise_floor, self._threshold,
        )

    def capture(self, timeout: float) -> Optional[np.ndarray]:
        """
        Same VAD capture logic as before, but drains the persistent
        stream's queue instead of opening a fresh stream per call.
        Raises MicUnavailableError if the device appears to have gone
        silent at the driver level (no frames at all, not just quiet ones).
        """
        self.ensure_open()

        speech_buffer: List[np.ndarray] = []
        pre_buffer:    List[np.ndarray] = []
        recording      = False
        silence_start: Optional[float] = None
        idle_clock      = time.time()
        empty_reads     = 0

        while True:
            try:
                chunk = self._queue.get(timeout=0.5)
                empty_reads = 0
            except queue.Empty:
                empty_reads += 1
                if empty_reads >= MIC_SILENT_FRAMES_LIMIT:
                    logger.warning(
                        "No audio frames received for ~%.0fs — microphone may be disconnected.",
                        empty_reads * 0.5,
                    )
                    self._close_stream()
                    raise MicUnavailableError("No audio frames received — microphone may be disconnected.")
                if not recording and time.time() - idle_clock >= timeout:
                    return None
                continue

            rms = float(np.sqrt(np.mean(chunk ** 2)))

            if _mic_muted:
                idle_clock = time.time()
                continue

            if rms >= self._threshold:
                idle_clock    = time.time()
                silence_start = None
                if not recording:
                    recording     = True
                    speech_buffer = list(pre_buffer)
                speech_buffer.append(chunk)

            elif recording:
                speech_buffer.append(chunk)
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start >= SILENCE_AFTER_SPEECH:
                    break

            else:
                # Pure ambient background (not speech, not trailing off
                # from speech) — exactly what we want to learn the
                # room's noise floor from. Updating this continuously,
                # not just at startup, means the threshold keeps
                # tracking the room if the auditorium gets louder or
                # quieter as the event goes on. (FIX-P11)
                self._update_noise_floor(rms)
                pre_buffer.append(chunk)
                if len(pre_buffer) > PRE_ROLL_CHUNKS:
                    pre_buffer.pop(0)
                if time.time() - idle_clock >= timeout:
                    return None

        if not speech_buffer:
            return None
        audio = np.concatenate(speech_buffer, axis=0)
        return audio if len(audio) >= SAMPLE_RATE * MIN_SPEECH_SECS else None

    def drain(self) -> int:
        """
        Discard any audio chunks currently sitting in the queue without
        processing them. Used right after TTS playback ends: the mic
        callback keeps queuing frames the whole time we're speaking (it
        has no idea we're muted), so a big backlog of the bot's own
        voice builds up while speak() blocks. If that backlog isn't
        flushed before we unmute, capture() will race through it at
        full speed right after unmuting and mistake our own playback
        for a fresh user utterance — that's the self-listening bug.
        Returns the number of chunks discarded (useful for logging).
        """
        with self._queue.mutex:
            discarded = len(self._queue.queue)
            self._queue.queue.clear()
        return discarded

    def close(self) -> None:
        self._close_stream()


def resolve_mic_device(name: str) -> Optional[str]:
    if not name:
        logger.info("MIC_NAME not set — using system default input device.")
        return None

    devices = sd.query_devices()
    matches = [d for d in devices if name.lower() in d["name"].lower() and d["max_input_channels"] > 0]

    if not matches:
        available = [f"  [{i}] {d['name']}" for i, d in enumerate(devices) if d["max_input_channels"] > 0]
        raise RuntimeError(
            f"MIC_NAME='{name}' did not match any input device.\n"
            f"Available input devices:\n" + "\n".join(available)
        )

    if len(matches) > 1:
        logger.warning(
            "MIC_NAME='%s' matched %d devices — using the first: '%s'.",
            name, len(matches), matches[0]["name"],
        )
    else:
        logger.info("Microphone resolved: '%s'", matches[0]["name"])

    return name


_MIC_DEVICE: Optional[str] = resolve_mic_device(MIC_NAME)
_mic_manager = MicManager(_MIC_DEVICE)


# ══════════════════════════════════════════════════════════
#  TRANSCRIBE  (FIX-R1: in-memory upload, FIX-R9: fail-fast offline)
# ══════════════════════════════════════════════════════════

def _has_devanagari(text: str) -> bool:
    return any(0x0900 <= ord(ch) <= 0x097F for ch in text)


def _has_arabic_script(text: str) -> bool:
    return any(0x0600 <= ord(ch) <= 0x06FF for ch in text)


def _transcribe_once(audio: np.ndarray, model: str, language: Optional[str] = None) -> Tuple[str, str]:
    """
    Upload audio directly as in-memory bytes — no temp file, no disk I/O.
    """
    wav_bytes = _audio_to_wav_bytes(audio)
    kwargs = dict(model=model, response_format="verbose_json")
    if language:
        kwargs["language"] = language

    result = client.audio.transcriptions.create(file=("audio.wav", wav_bytes), **kwargs)

    text = sanitize_text(result.text)
    lang = (result.language or "en").strip().lower()

    if lang == "ur":
        lang = "hi"
    if lang not in ("hi", "en"):
        lang = "en"
    if _has_devanagari(text) or _has_arabic_script(text):
        lang = "hi"

    return text, lang


def transcribe(audio: np.ndarray) -> Tuple[str, str]:
    """
    Full-quality transcription for conversation turns. FIX-R9: checks
    internet up front rather than waiting on the SDK's own timeout, so
    an outage is detected and reported quickly and consistently.
    """
    if not is_internet_available():
        raise ConnectionError("No internet connection.")

    text, lang = _transcribe_once(audio, STT_MODEL)

    if lang == "hi" and _has_arabic_script(text) and not _has_devanagari(text):
        logger.debug("Hindi speech transcribed in Urdu/Arabic script (%r) — retrying with hint.", text)
        retry_text, _ = _transcribe_once(audio, STT_MODEL, language="hi")
        if retry_text and _has_devanagari(retry_text):
            logger.debug("Retry succeeded with Devanagari output: %r", retry_text)
            return retry_text, "hi"
        logger.debug("Retry did not produce Devanagari output — keeping original.")

    return text, lang


# ══════════════════════════════════════════════════════════
#  TTS  (FIX-P5, FIX-R1: in-memory synthesis+playback, FIX-R3: mixer recovery)
# ══════════════════════════════════════════════════════════

_mic_muted: bool = False
_tts_loop = asyncio.new_event_loop()


def pick_voice(text: str, lang: str) -> str:
    if lang == "hi":
        return TTS_VOICE_HI
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F or 0x0600 <= cp <= 0x06FF:
            return TTS_VOICE_HI
    return TTS_VOICE_EN


async def _tts_synthesize_bytes(text: str, voice: str) -> bytes:
    """Stream synthesis directly into memory — no MP3 file ever touches disk."""
    communicate = edge_tts.Communicate(text, voice=voice)
    chunks = bytearray()
    async for event in communicate.stream():
        if event["type"] == "audio":
            chunks.extend(event["data"])
    return bytes(chunks)


def _ensure_mixer() -> bool:
    """
    FIX-R3: health-check pygame's mixer before playback and re-initialize
    it if it has gone down (e.g. a USB speaker was unplugged/replugged).
    Returns True if the mixer is usable.
    """
    try:
        if pygame.mixer.get_init() is not None:
            return True
    except Exception:
        pass
    try:
        pygame.mixer.quit()
    except Exception:
        pass
    try:
        pygame.mixer.init()
        logger.warning("pygame mixer re-initialized (was down — likely a speaker disconnect).")
        return True
    except Exception as exc:
        logger.error("Failed to reinitialize audio mixer: %s", exc)
        return False


def _load_music_from_bytes(mp3_bytes: bytes) -> Optional[str]:
    """
    Load MP3 bytes into pygame's mixer for playback without touching disk.
    Returns a temp file path ONLY if this pygame build can't load from a
    file-like object (compatibility fallback) — caller must delete it.
    Returns None when the in-memory path succeeded (the common case).
    """
    buffer = io.BytesIO(mp3_bytes)
    try:
        pygame.mixer.music.load(buffer, "mp3")
        return None
    except TypeError:
        pass  # older pygame without namehint support — try without it
    except Exception:
        pass  # some other in-memory load failure — fall through to disk

    buffer.seek(0)
    try:
        pygame.mixer.music.load(buffer)
        return None
    except Exception as exc:
        logger.debug("In-memory mixer load unsupported on this pygame build (%s) — using a one-off temp file.", exc)

    fd, tmp_path = tempfile.mkstemp(suffix=".mp3")
    with os.fdopen(fd, "wb") as f:
        f.write(mp3_bytes)
    pygame.mixer.music.load(tmp_path)
    return tmp_path


def speak(text: str, lang: str = "en") -> None:
    global _mic_muted
    logger.debug("TTS input: %r", text)

    if is_blank(text):
        logger.error("TTS input validation failed: text is empty or None.")
        _speak_direct(ERROR_MESSAGES["env_error"]["en"], TTS_VOICE_EN)
        return

    voice = pick_voice(text, lang)
    logger.info("TTS [%s]: %s", voice, textwrap.shorten(text, width=80))

    _mic_muted = True  # gate mic BEFORE playback — prevents self-triggering
    try:
        if not is_internet_available():
            logger.warning("speak(): offline — skipping edge-tts, using espeak.")
            _speak_espeak(text, lang)
            return
        if _speak_edge_tts(text, voice):
            return
        logger.warning("edge-tts failed — attempting offline espeak fallback.")
        if _speak_espeak(text, lang):
            return
        logger.error("All TTS engines failed for this utterance.")
    finally:
        # FIX-P10: the mic callback queues frames continuously, even
        # while _mic_muted is True, since speak() blocks the main
        # thread and never calls capture() to drain them. That backlog
        # is essentially a recording of our own TTS output. If we just
        # flip _mic_muted back to False here, the next capture() call
        # races through that backlog unmuted and mistakes it for a
        # live user utterance — that's the self-listening bug. So:
        # wait out a short cooldown for speaker/room decay, discard
        # whatever accumulated, THEN unmute.
        time.sleep(POST_SPEECH_COOLDOWN_S)
        discarded = _mic_manager.drain()
        if discarded:
            logger.debug("Discarded %d stale audio chunk(s) queued during TTS playback.", discarded)
        _mic_muted = False  # ALWAYS release gate after playback


def _speak_edge_tts(text: str, voice: str) -> bool:
    if not _ensure_mixer():
        return False

    tmp_fallback_path: Optional[str] = None
    try:
        mp3_bytes = _tts_loop.run_until_complete(_tts_synthesize_bytes(text, voice))
        if not mp3_bytes:
            raise RuntimeError("edge-tts produced no audio data.")

        tmp_fallback_path = _load_music_from_bytes(mp3_bytes)

        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)
        pygame.mixer.music.stop()
        pygame.mixer.music.unload()
        return True

    except Exception as exc:
        logger.error("edge-tts error: %s", exc)
        return False

    finally:
        if tmp_fallback_path and os.path.exists(tmp_fallback_path):
            try:
                os.unlink(tmp_fallback_path)
            except OSError:
                pass


def _speak_espeak(text: str, lang: str) -> bool:
    if not _ESPEAK_AVAILABLE:
        logger.error("espeak not found — cannot speak offline.")
        return False

    voices_to_try = (["hi"] if lang == "hi" else []) + ["en"]
    for voice in voices_to_try:
        try:
            subprocess.run(
                ["espeak", "-v", voice, "-s", "140", "-a", "180", text],
                check=True,
                timeout=15,
                stderr=subprocess.DEVNULL,
            )
            return True
        except subprocess.CalledProcessError:
            logger.warning("espeak voice '%s' unavailable, trying next …", voice)
            continue
        except subprocess.TimeoutExpired:
            logger.error("espeak timed out.")
            return False
        except Exception as exc:
            logger.error("espeak error: %s", exc)
            return False

    logger.error("espeak: no usable voice found.")
    return False


def _speak_direct(text: str, voice: str) -> None:
    if _speak_edge_tts(text, voice):
        return
    logger.warning("_speak_direct: edge-tts failed, trying espeak.")
    if _speak_espeak(text, lang="en"):
        return
    logger.error("_speak_direct: all engines failed (giving up).")


# ══════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════

def print_banner(rag_en_ready: bool, rag_hi_ready: bool) -> None:
    status_en = "✅ PDF loaded" if rag_en_ready else "⚠️  PDF not found — web-only mode"
    status_hi = "✅ PDF loaded" if rag_hi_ready else "⚠️  PDF not found — web-only mode"
    mic_label = f"'{MIC_NAME}'" if MIC_NAME else "system default"
    sep = "=" * 60
    banner = (
        f"\n{sep}\n"
        f"  Naaila 🤖  |  Robotwala, Indore\n"
        f"{sep}\n"
        f"  RAG (EN) status : {status_en}\n"
        f"  RAG (HI) status : {status_hi}\n"
        f"  PDF (EN) path   : {PDF_PATH_EN}\n"
        f"  PDF (HI) path   : {PDF_PATH_HI}\n"
        f"  PDF threshold   : {PDF_THRESHOLD}  (below → web fallback)\n"
        f"  Microphone      : {mic_label} (persistent stream)\n"
        f"  Max history     : {MAX_HISTORY_TURNS} turns per language\n"
        f"  Long-term memory: {LTM_PATH} ({len(long_term_memory.retrieve('', limit=LTM_MAX_FACTS))} facts)\n"
        f"  Log level       : {_log_level_name}\n"
        f"  States          :\n"
        f"    👂 LISTENING  — always listening (mic muted/unmuted manually)\n"
        f"    🔊 SPEAKING   — playing response\n"
        f"  Ctrl+C to quit\n"
        f"{sep}\n"
    )
    print(banner)  # startup UX, not a log event — intentional


def state_label(state: State) -> str:
    return {
        State.LISTENING: "👂 LISTENING",
        State.THINKING:  "🤔 THINKING",
        State.SPEAKING:  "🔊 SPEAKING",
    }[state]


# ══════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════
#  CONTENT MODERATION  (FIX-P14)
# ══════════════════════════════════════════════════════════
# This bot gets taken on stage at live events (schools, colleges,
# motivational talks) where anyone can walk up to the mic. This layer
# exists ONLY to catch genuine harassment, abuse, threats, or clearly
# inappropriate requests (sexual content, violence, hate speech,
# self-harm) — it is NOT a topic filter. An ordinary off-topic
# question ("what's the capital of France", "tell me a joke") is left
# alone and flows through to the LLM exactly as before; only actually
# harmful input gets intercepted here.

MODERATION_ENABLED           = True
MODERATION_LLM_ENABLED       = True   # layer 2 (subtler cases) — one small extra Groq call per turn
MODERATION_MIN_WORDS_FOR_LLM = 2      # skip layer 2 for single-word utterances (usually just noise)

# Layer 1: fast, local, no-network blocklist for the unambiguous
# cases — explicit profanity, slurs, direct threats. Deliberately
# narrow: mild words like "stupid"/"boring" are NOT here, since this
# is meant to catch harassment, not ordinary blunt phrasing from
# students. Layer 2 below handles subtler cases this misses.
_ABUSE_PATTERNS = re.compile(
    r'\b('
    r'chutiya|madarchod|bhosdi|gaandu|randi|haramzada|kamina|'
    r'bastard|asshole|bitch|fuck|fucking|motherfucker|'
    r'kill\s+you|bomb|terrorist|rape|molest'
    r')\b',
    re.IGNORECASE,
)


def is_harmful_input(text: str) -> Tuple[bool, str]:
    """
    Two-layer moderation. Only flags genuine harassment/abuse/threats
    or clearly inappropriate requests — ordinary off-topic questions
    are always SAFE and pass through untouched.
      Layer 1 — regex blocklist (instant, no API call).
      Layer 2 — LLM classifier for subtler cases. Skipped for very
                short input, and fails OPEN (treated as safe) if the
                classifier call itself errors, so a moderation hiccup
                never blocks a legitimate question mid-event.
    Returns (is_harmful, reason) — reason is for logs only, never
    spoken to the user.
    """
    if not MODERATION_ENABLED or is_blank(text):
        return False, ""

    if _ABUSE_PATTERNS.search(text):
        return True, "keyword filter"

    if not MODERATION_LLM_ENABLED or len(text.split()) < MODERATION_MIN_WORDS_FOR_LLM:
        return False, ""

    try:
        mod_response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a content moderation classifier for a live, "
                        "public voice assistant used on stage in front of "
                        "students. Classify the user message as SAFE or "
                        "UNSAFE. UNSAFE means: harassment, abuse, threats, "
                        "sexual content, self-harm, violence, or hate speech "
                        "directed at the assistant, an individual, or a "
                        "group. An ordinary off-topic or unrelated question "
                        "is still SAFE — only flag genuinely harmful content. "
                        "Reply with ONLY one word: SAFE or UNSAFE."
                    ),
                },
                {"role": "user", "content": text},
            ],
            max_tokens=5,
            temperature=0,
            timeout=10,
        )
        verdict = (mod_response.choices[0].message.content or "").strip().upper()
        if verdict == "UNSAFE":
            return True, "LLM classifier"
    except Exception as exc:
        logger.warning("Moderation LLM check failed — skipping check (fail-open): %s", exc)

    return False, ""


# FIX-P14: deliberately no scope-narrowing language ("I can only
# answer Robotwala questions") and no Robotwala mention at all — this
# bot goes to general motivational events, not just Robotwala booths,
# so the refusal just redirects to something positive, nothing more.
_MODERATION_REFUSAL = {
    "en": "Let's keep things positive and respectful — happy to help with something else.",
    "hi": "Chaliye ise positive aur respectful rakhte hain — kisi aur sawaal mein zaroor madad karunga.",
}


def get_moderation_refusal(lang: str) -> str:
    return _MODERATION_REFUSAL.get(lang, _MODERATION_REFUSAL["en"])


def _process_query(user_text: str, lang: str, rag_en: RAGEngine, rag_hi: RAGEngine) -> Optional[str]:
    clean = sanitize_text(user_text)
    if is_blank(clean):
        logger.warning("Ignoring blank user input (after sanitization).")
        return None

    logger.info("User [%s] › %s", lang.upper(), clean)

    # Harassment/abuse/threats/explicit content get intercepted here —
    # never reach RAG or the LLM, and never get added to conversation
    # history. Ordinary off-topic questions are untouched by this and
    # flow through normally below.
    harmful, reason = is_harmful_input(clean)
    if harmful:
        logger.warning("Input blocked by moderation (%s).", reason)
        refusal = get_moderation_refusal(lang)
        logger.info("AI   [%s] › %s", lang.upper(), refusal)
        return refusal

    logger.debug("Retrieving context …")"""
  FIX-P3  Conversation history capped at MAX_HISTORY_TURNS.
  FIX-P4  All print() calls replaced with the stdlib logging module.
  FIX-P5  asyncio event loop created once, reused by every speak() call.
  FIX-P6  Reply scoped per iteration via a helper function.
  FIX-P7  User text sanitized before being passed to the LLM / logged.
  FIX-P8  build_context() sets source="PDF" whenever pdf_context is non-empty.
  FIX-P9  transcribe() retries once with language="hi" when Whisper outputs
          Urdu/Arabic script for what is actually Hindi speech.

  ── Audit Pass 2 (reliability) ──────────────────────────────────────
  FIX-A1  is_internet_available() no longer mutates the global socket
          timeout — uses a scoped per-call timeout instead.
  FIX-A2  LLM retries distinguish retryable (429/5xx/network) from
          non-retryable (4xx) errors — fails fast on the latter.
  FIX-A3  Retry backoff includes jitter.
  FIX-A4  TokenStats: session/daily totals, avg/min/max tokens, avg
          latency, requests/hour, optional cost estimate.
  FIX-A5  LongTermMemory: selective, bounded, persisted durable facts.
  FIX-A6  validate_config() fails fast at startup on bad configuration.
  FIX-A7  Precompiled whitespace regex.
  FIX-A8  CircuitBreaker for the Groq chat endpoint.

  ── Audit Pass 3 (Raspberry Pi resource optimization) ───────────────
  FIX-R1  STT and TTS no longer touch disk in the normal path. Audio is
          held in memory (io.BytesIO / raw bytes) end-to-end: WAV bytes
          are built in memory and uploaded directly as a (filename,
          bytes) tuple for STT; TTS audio is streamed from edge-tts
          directly into pygame's mixer via a BytesIO buffer. A temp-file
          write is used ONLY as a last-resort compatibility fallback if
          the installed pygame build can't load from a file-like object,
          and that file is deleted immediately after playback. This
          removes the majority of disk writes from the hot path — a real
          concern for SD-card wear and I/O latency on a Pi.

  FIX-R2  The microphone is no longer opened and closed on every single
          listen cycle. A persistent PortAudio InputStream is opened once
          (MicManager) and reused across the whole run, eliminating
          repeated ALSA/PortAudio setup/teardown overhead. MicManager
          also detects a stalled/disconnected device (no audio frames at
          all for several seconds — silence still delivers frames; total
          silence from the *driver* does not) and transparently attempts
          reconnection with backoff, re-resolving the device by name
          since a physical replug can change its device index.

  FIX-R3  pygame's mixer is health-checked before each playback and
          re-initialized automatically if it has gone down (e.g. a USB
          speaker was unplugged and replugged), so speaker failures
          self-heal on the next turn instead of silently going dark
          forever.

  FIX-R4  RAG PDF indices (chunks + TF-IDF vectorizer + matrix) are
          cached to disk (pickle) keyed by the source PDF's mtime.
          Startup skips PDF text extraction and re-vectorization entirely
          when the cache is valid, cutting cold-start CPU time and
          latency; the cache self-invalidates the moment the PDF file
          changes.

  FIX-R5  TfidfVectorizer now stores its matrix as float32 instead of
          float64, halving the RAM footprint of both RAG indices with no
          precision loss that matters for cosine similarity ranking.

  FIX-R6  Web search uses one shared, small-pool requests.Session instead
          of ad-hoc connections, reusing TCP/TLS handshakes across calls
          to cut network latency and CPU.

  FIX-R7  Removed threading.Lock usage from history / TokenStats /
          LongTermMemory / CircuitBreaker. This process is effectively
          single-threaded — the only other thread is the PortAudio
          callback thread owned by sounddevice, which never touches this
          state (it only pushes into a thread-safe queue.Queue and reads
          a bool). The locks provided no real protection and were pure
          overhead; removed for simplicity and to avoid implying
          concurrency that doesn't exist.

  FIX-R8  A cheap periodic gc.collect() runs at most every 30 minutes


    context, source = build_context(clean, lang, rag_en, rag_hi)
    logger.info("Source: %s", source)
    logger.debug("Generating reply …")

    try:
        reply = get_ai_reply(clean, lang, context)
    except Exception as exc:
        logger.error("LLM generation failed: %s", exc)
        announce_error(exc, lang)
        return None

    logger.info("AI   [%s] › %s", lang.upper(), reply)
    return reply


def main() -> None:
    try:
        validate_config()

        pygame.mixer.init()

        rag_en = RAGEngine()
        rag_hi = RAGEngine()
        rag_en.load_pdf(PDF_PATH_EN)
        rag_hi.load_pdf(PDF_PATH_HI)
        print_banner(rag_en.ready, rag_hi.ready)

        # Open the microphone once, up front, so a hardware problem is
        # caught cleanly at startup instead of deep inside the hot loop.
        try:
            _mic_manager.ensure_open()
            _mic_manager.calibrate()  # FIX-P11: set the initial threshold from this room's actual noise floor
        except MicUnavailableError as exc:
            logger.critical("Could not open microphone at startup: %s", exc)
            raise

        state = State.LISTENING
        lang  = "hi"
        last_gc_time = time.time()

        speak("Hello"

        #over to you
        #Thank you so much 
        
        , lang="hi")

        while True:

            # ── LISTENING (always on) ────────────────────
            # No idle timeout, no wake word. The mic is expected to be
            # physically muted/unmuted by the operator; whenever it's
            # unmuted and picks up speech, it's processed directly.
            if state == State.LISTENING:
                logger.debug(state_label(state))

                # FIX-R8: cheap defensive GC pass, at most every 30 min.
                if time.time() - last_gc_time >= GC_INTERVAL_SECONDS:
                    collected = gc.collect()
                    logger.debug("GC pass collected %d objects.", collected)
                    last_gc_time = time.time()

                try:
                    audio = _mic_manager.capture(timeout=LISTEN_TIMEOUT)
                except MicUnavailableError as exc:
                    announce_error(exc, "en")
                    continue  # ensure_open() will retry with backoff on next capture()

                if audio is None:
                    # LISTEN_TIMEOUT is infinite, so this shouldn't
                    # normally trigger — but if it ever does, just
                    # keep listening rather than going idle.
                    continue

                try:
                    user_text, lang = transcribe(audio)
                except Exception as exc:
                    logger.error("Transcription failed: %s", exc)
                    announce_error(exc, lang)
                    continue

                if is_blank(user_text):
                    logger.debug("Blank transcription — skipping.")
                    continue

                state = State.THINKING
                logger.info(state_label(state))
                reply = _process_query(user_text, lang, rag_en, rag_hi)

                if reply is None:
                    state = State.LISTENING
                    continue

                state = State.SPEAKING
                logger.info(state_label(state))
                speak(reply, lang)
                state = State.LISTENING
                continue

    except KeyboardInterrupt:
        logger.info("Shutdown requested by user (Ctrl+C).")
    except Exception as exc:
        logger.critical("Fatal error in main loop: %s", exc, exc_info=True)
        try:
            announce_error(exc, "en")
        except Exception:
            pass
    finally:
        token_stats.log_summary()
        try:
            _mic_manager.close()
        except Exception:
            pass
        try:
            pygame.mixer.quit()
        except Exception:
            pass
        try:
            _tts_loop.close()
        except Exception:
            pass
        try:
            _web_session.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
