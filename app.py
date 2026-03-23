
## Buildin a basic Speech-to-Speech streamlit app
## 1) Record audio in the browser using st.audio_input(WAV)
## 2) Transcribe speech -> text ussing fast whisper (cpu, int8)
# 3) Generate a reply
# 4) Convert reply text -> speech using:
# - gtts (light, needs internet)
# - piper (offline, needs .onnx + .onnx.json voice files)


# Asssignment

# Auto-run: button ke bghair record hote he pipelie run ho.
# Chat Bubble : st.chat_messsage use karein. Make Sure your UI should look professional
# clear chat : session_state reset button
# (optional) latency time : ASR/LLM/TTS time measure aur display
# (optional) Better Validatio: empty audio, noise handling
# Voice Switch: Piper Voices dropdown (agr modl available ho)
# (optional) : try using it with one new language


from __future__ import annotations
import os
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


# config
@dataclass(frozen=True)
class AppConfig:
    # ASR
    whisper_model: str
    whisper_language: str
    whisper_device: str
    whisper_compute_type: str
    whisper_cpu_threads: int

    # LLM
    groq_api_key: str
    groq_model_id: str
    system_prompt: str

    # TTS
    tts_engine: str  # "gtts" or "piper"
    gtts_lang: str
    piper_model_path: str
    piper_config_path: str


def load_config() -> AppConfig:
    def _get_int(name: str, default: int) -> int:
        raw = os.getenv(name, "").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            return default

    return AppConfig(
        whisper_model=os.getenv("WHISPER_MODEL", "base.en").strip(),
        whisper_language=os.getenv("WHISPER_LANGUAGE", "en").strip(),
        whisper_device=os.getenv("WHISPER_DEVICE", "cpu").strip(),
        whisper_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8").strip(),
        whisper_cpu_threads=_get_int("WHISPER_CPU_THREADS", 4),

        groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
        groq_model_id=os.getenv("GROQ_MODEL_ID", "llama-3.1-8b-instant").strip(),
        system_prompt=os.getenv(
            "SYSTEM_PROMPT",
            "You are a helpful voice assistant for students. Keep replies short and clear."
        ).strip(),

        tts_engine=os.getenv("TTS_ENGINE", "gtts").strip().lower(),
        gtts_lang=os.getenv("GTTS_LANG", "en").strip(),
        piper_model_path=os.getenv("PIPER_MODEL_PATH", "").strip(),
        piper_config_path=os.getenv("PIPER_CONFIG_PATH", "").strip(),
    )

CFG = load_config()

# ASR : faster - whispr
@st.cache_resource(show_spinner=False)
def get_whisper_model(model_size: str, device: str, compute_type: str, cpu_threads: int):
    """
    Loads Whisper model once and caches it across Streamlit reruns.
    For low RAM laptops:
      - prefer model_size = tiny.en or base.en
      - compute_type = int8 on CPU
      - cpu_threads around 4 can reduce memory spikes on some machines
    """
    from faster_whisper import WhisperModel
    return WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
    )


def transcribe_wav_bytes(wav_bytes: bytes) -> str:
    """
    Streamlit st.audio_input returns WAV bytes, so we can save and transcribe directly.
    """
    model = get_whisper_model(
        CFG.whisper_model,
        CFG.whisper_device,
        CFG.whisper_compute_type,
        CFG.whisper_cpu_threads,
    )

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            tmp_path = f.name

        segments, _info = model.transcribe(
            tmp_path,
            language=CFG.whisper_language,
            beam_size=1,        # keep small for speed
            vad_filter=True,    # reduce silence / noise
        )

        text = "".join(seg.text for seg in segments).strip()
        return text

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

# LLM Groq
def offline_demo_reply(user_text: str) -> str:
    if not user_text:
        return "I did not catch that. Please try again."
    return (
        "Offline demo mode: I can hear you, but I am not connected to an AI model yet.\n\n"
        f"You said: {user_text}\n\n"
        "To enable real AI replies, add GROQ_API_KEY in your .env."
    )


def groq_chat_completion(messages: List[Dict[str, str]]) -> str:
    """
    Calls Groq OpenAI-compatible chat completions endpoint.
    """
    import requests

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {CFG.groq_api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": CFG.groq_model_id,
        "messages": messages,
        "temperature": 0.4,
        "max_tokens": 250,
    }

    r = requests.post(url, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()


def generate_reply(user_text: str, history: List[Dict[str, str]]) -> str:
    if not CFG.groq_api_key:
        return offline_demo_reply(user_text)


    trimmed = history[-6:] if len(history) > 6 else history

    messages = [{"role": "system", "content": CFG.system_prompt}]
    messages.extend(trimmed)
    messages.append({"role": "user", "content": user_text})

    try:
        return groq_chat_completion(messages)
    except Exception as e:
        
        return f"I could not reach Groq right now. Error: {e}"

# TTS: gTTS (online) or Piper(offline)

def tts_to_audio_file(text: str) -> Tuple[bytes, str, str]:
    """
    Returns: (audio_bytes, mime_type, file_name)
    """
    text = (text or "").strip()
    if not text:
        text = "I do not have a response to speak."

    if CFG.tts_engine == "piper":
        return piper_tts(text)

    # default: gtts
    return gtts_tts(text)


def gtts_tts(text: str) -> Tuple[bytes, str, str]:
    """
    Very lightweight TTS, but requires internet.
    Output: MP3 bytes.
    """
    from gtts import gTTS

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        out_path = f.name

    try:
        tts = gTTS(text=text, lang=CFG.gtts_lang)
        tts.save(out_path)

        with open(out_path, "rb") as rf:
            audio_bytes = rf.read()

        return audio_bytes, "audio/mpeg", "reply.mp3"
    finally:
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass


def piper_tts(text: str) -> Tuple[bytes, str, str]:
    """
    Offline neural TTS using Piper.
    Requires:
      - PIPER_MODEL_PATH (onnx)
      - PIPER_CONFIG_PATH (onnx.json)
    Output: WAV bytes.
    """
    import wave
    from piper import PiperVoice

    if not CFG.piper_model_path or not CFG.piper_config_path:
        
        return gtts_tts(
            "Piper is not configured. Please set PIPER_MODEL_PATH and PIPER_CONFIG_PATH in .env."
        )

    voice = PiperVoice.load(CFG.piper_model_path, CFG.piper_config_path)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        out_path = f.name

    try:
        with wave.open(out_path, "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)

        with open(out_path, "rb") as rf:
            audio_bytes = rf.read()

        return audio_bytes, "audio/wav", "reply.wav"
    finally:
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass

# Streamlit UI
st.set_page_config(page_title="Speech-to-Speech", layout="centered")
st.title("VoiceBridge Speech to Speech AI")


with st.sidebar:
    st.subheader("Settings")
    st.write("These come from your .env")
    st.code(
        "\n".join(
            [

                # ASR (SPEECH TO TEXT)
                f"WHISPER_MODEL={CFG.whisper_model}",

                f"WHISPER_LANGUAGE={CFG.whisper_language}",
                f"WHISPER_DEVICE={CFG.whisper_device}",
                f"WHISPER_COMPUTE_TYPE={CFG.whisper_compute_type}",
                f"WHISPER_CPU_THREADS={CFG.whisper_cpu_threads}",


                f"GROQ_API_KEY= {'SET' if bool(CFG.groq_api_key) else 'NOT SET (offline demo)'}",
                f"GROQ_MODEL_ID ={CFG.groq_model_id}",
               
                ## TTS 
                f"TTS_ENGINE={CFG.tts_engine}",
                f"GTTS_LANG={CFG.gtts_lang}",

                f"PIPER_MODEL_PATH={CFG.piper_model_path}",
                f"PIPER_CONFIG_PATH={CFG.piper_config_path}" 

            ]
        )
    )
    st.caption("Tip: For low RAM, try whisper tiny.en")

if "chat_history" not in st.session_state:
    st.session_state.chat_history =[]

st.write("Record your voice, then click the button to hear the AI Reply")

audio_value = st.audio_input("Record your Voice")

if audio_value:
    st.audio(audio_value)

    if st.button("Transcribe -> Reply -> Speak", type="primary"):
        wav_bytes = audio_value.getvalue()

        with st.spinner("1) Transcribing speech to text..."):
            transcript = transcribe_wav_bytes(wav_bytes)

        if not transcript:
            st.error("No Speech detected. Try again in a quieter environment.")
        else:
            st.success("Transcript Ready")
            st.write('### You said')
            st.write(transcript)
        
        with st.spinner("2) Generate reply... "):
            reply_text = generate_reply(transcript, st.session_state.chat_history)
        
        st.write("## AI Reply")
        st.write(reply_text)

        # updted history
        st.session_state.chat_history.append({"role":"user","content":transcript})
        st.session_state.chat_history.append({"role": "assistant", "content":reply_text})

        with st.spinner("3) Converting reply to speech... "):
            audio_bytes, mime, fname= tts_to_audio_file(reply_text)
        
        st.write("### AI Voice response")
        st.audio(audio_bytes, format=mime)

        st.download_button(
            "Download reply audio",
            data = audio_bytes,
            file_name=fname,
            mime=mime
        )
st.divider()
st.write("## Debug / Helper")

if st.checkbox("Show chat history"):
    st.json(st.session_state.chat_history)



        
