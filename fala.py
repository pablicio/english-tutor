# app.py — English Tutor Bot v3 (Conversation/Grammar/Writing/Vocabulary + Speaking)
# -----------------------------------------------------------------------------------
# Novidades:
# - Modo "Speaking" com microfone via streamlit-webrtc
# - STT local com faster-whisper (fallback elegante se não estiver instalado)
# - Score simples de pronúncia (0–100) com base em confianças do ASR
# - Sugestões de frases para repetir + TTS opcional do alvo
# Mantido:
# - Prompt pedagógico, pós-processamento, modelo automático (Phi-3 -> TinyLlama)
# -----------------------------------------------------------------------------------

import os
import re
import json
import tempfile
from typing import List, Dict

import streamlit as st

st.set_page_config(page_title="English Tutor Bot", page_icon="🧑‍🏫", layout="centered")

# -----------------------------------------------------------------------------------
# Cabeçalho
# -----------------------------------------------------------------------------------
st.title("📚 English Tutor Bot")
st.caption("*Learn, practice, and get real-time feedback — now with Speaking mode.*")

# -----------------------------------------------------------------------------------
# Sidebar Settings
# -----------------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    if "level" not in st.session_state:
        st.session_state.level = "Intermediate"
    if "mode" not in st.session_state:
        st.session_state.mode = "Conversation"
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.7
    if "max_new_tokens" not in st.session_state:
        st.session_state.max_new_tokens = 180
    if "use_tts" not in st.session_state:
        st.session_state.use_tts = False
    if "target_sentence" not in st.session_state:
        st.session_state.target_sentence = "I go to school every day."

    level = st.selectbox("English Level", ["Beginner", "Intermediate", "Advanced"], key="level")
    mode = st.selectbox("Practice Mode", ["Conversation", "Grammar", "Writing", "Vocabulary", "Speaking"], key="mode")
    temperature = st.slider("Temperature", 0.0, 1.2, st.session_state.temperature, 0.1, key="temperature")
    max_new_tokens = st.slider("Max new tokens", 32, 512, st.session_state.max_new_tokens, 8, key="max_new_tokens")
    st.session_state.use_tts = st.toggle("🔊 Read answers (TTS)", value=st.session_state.use_tts)

    colA, colB = st.columns(2)
    with colA:
        if st.button("✨ Welcome Message"):
            st.session_state.messages = [{"role": "assistant", "content": "👋 Hi! I'm your English Tutor. Let's start practicing!"}]
            st.rerun()
    with colB:
        if st.button("🧹 Clear Chat"):
            st.session_state.messages = []
            st.rerun()

# -----------------------------------------------------------------------------------
# Estado inicial
# -----------------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "stats" not in st.session_state:
    st.session_state.stats = {"turns": 0}

# -----------------------------------------------------------------------------------
# Prompts pedagógicos
# -----------------------------------------------------------------------------------
def get_system_prompt(level: str, mode: str) -> str:
    base = (
        f"You are a professional English tutor specialized in helping {level.lower()} students improve grammar, vocabulary, and fluency. "
        "Always reply clearly, kindly, and educationally. "
        "Use Markdown formatting for clarity. "
        "Never include the words USER or ASSISTANT in your answers."
    )

    if mode == "Conversation":
        return base + (
            " If the student says something incorrect, correct it gently. "
            "Your answer must follow this structure:\n"
            "1️⃣ **Correction:** show the correct form.\n"
            "2️⃣ **Explanation:** explain why.\n"
            "3️⃣ **Example:** one short example sentence.\n"
            "End with a friendly question related to the topic."
        )
    elif mode == "Grammar":
        return base + (
            " Focus on grammar accuracy and clarity. "
            "Always use this structure:\n"
            "1️⃣ **Detected Error**\n2️⃣ **Correct Form**\n3️⃣ **Explanation**\n4️⃣ **Example**"
        )
    elif mode == "Writing":
        return base + (
            " When the student submits a text, reply in this structure:\n"
            "1️⃣ **What's Good:** highlight positives.\n"
            "2️⃣ **Corrections:** fix grammar and phrasing.\n"
            "3️⃣ **Explanation:** explain key mistakes.\n"
            "4️⃣ **Improved Version:** rewrite the text naturally."
        )
    elif mode == "Vocabulary":
        return base + (
            " When teaching vocabulary, answer in this structure:\n"
            "1️⃣ **Word Meaning**\n2️⃣ **Synonyms**\n3️⃣ **Example Sentence**\n"
            "Then ask: 'Can you use this word in your own sentence?'"
        )
    else:  # Speaking
        return base + (
            " You are a pronunciation coach. The student will speak or read a short sentence. "
            "Return feedback with this structure:\n"
            "1️⃣ **Overall Score (0–100)**\n"
            "2️⃣ **Mispronounced Words** (list)\n"
            "3️⃣ **Tips** (2–3 short tips to improve)\n"
            "4️⃣ **Model Repetition** (repeat the correct sentence for the student to shadow)\n"
            "Keep it concise and motivating."
        )

# -----------------------------------------------------------------------------------
# Construir prompt concatenado (para modelos text-generation)
# -----------------------------------------------------------------------------------
def build_prompt(system_prompt, history):
    prompt = f"SYSTEM: {system_prompt}\n"
    for msg in history:
        role = "ASSISTANT" if msg["role"] == "assistant" else "USER"
        content = msg["content"].strip()
        prompt += f"{role}: {content}\n"
    prompt += "ASSISTANT:"
    return prompt

# -----------------------------------------------------------------------------------
# Carregar modelo automaticamente (chat LLM)
# -----------------------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_pipeline():
    from transformers import pipeline
    st.info("⏳ Loading model...")
    try:
        model_id = "microsoft/Phi-3-mini-4k-instruct"
        pipe = pipeline("text-generation", model=model_id)
        st.success("✅ Using Phi-3-mini-4k-instruct")
        return pipe
    except Exception:
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        pipe = pipeline("text-generation", model=model_id)
        st.warning("⚙️ Using TinyLlama (smaller fallback model)")
        return pipe

# -----------------------------------------------------------------------------------
# Geração + limpeza
# -----------------------------------------------------------------------------------
def generate(pipe, prompt, max_new_tokens, temperature):
    out = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=pipe.tokenizer.eos_token_id if hasattr(pipe, "tokenizer") else None,
    )
    text = out[0]["generated_text"]
    if "ASSISTANT:" in text:
        text = text.split("ASSISTANT:", 1)[-1]
    text = re.sub(r"(USER|ASSISTANT):", "", text)
    return text.strip()

def postprocess_response(resp):
    resp = re.sub(r"(USER|ASSISTANT):", "", resp)
    resp = re.sub(r"\[.*?\]", "", resp)
    resp = re.sub(r"\s{3,}", "\n\n", resp)
    return resp.strip()

# -----------------------------------------------------------------------------------
# TTS (opcional)
# -----------------------------------------------------------------------------------
def try_tts(text: str):
    if not st.session_state.use_tts:
        return
    try:
        from gtts import gTTS
        tts = gTTS(text)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmp.name)
        st.audio(tmp.name)
    except Exception as e:
        st.info(f"🔇 TTS unavailable ({e}). You can disable it in Settings.")

# -----------------------------------------------------------------------------------
# Render histórico (não mostrar no modo Speaking para focar)
# -----------------------------------------------------------------------------------
if st.session_state.mode != "Speaking":
    st.markdown(f"**🎯 Level:** `{st.session_state.level}` | **Mode:** `{st.session_state.mode}`")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑‍🏫" if msg["role"] == "assistant" else "🧑‍💻"):
            st.markdown(msg["content"])

# -----------------------------------------------------------------------------------
# Entrada de texto (modos não-Speaking)
# -----------------------------------------------------------------------------------
def run_text_modes():
    if user_text := st.chat_input("Type here..."):
        # Reformula perguntas muito curtas para dar contexto ao modelo
        if len(user_text.split()) < 5:
            user_text = f"Please check if this sentence is correct: {user_text}"

        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_text)

        with st.chat_message("assistant", avatar="🧑‍🏫"):
            with st.spinner("Thinking..."):
                pipe = load_pipeline()
                system_prompt = get_system_prompt(st.session_state.level, st.session_state.mode)
                history_slice = st.session_state.messages[-6:]
                prompt = build_prompt(system_prompt, history_slice)

                raw = generate(pipe, prompt, st.session_state.max_new_tokens, st.session_state.temperature)
                bot_response = postprocess_response(raw)

                st.markdown(bot_response)
                try_tts(bot_response)

        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        st.session_state.stats["turns"] += 1

# -----------------------------------------------------------------------------------
# Speaking Mode (microfone + STT + feedback)
# -----------------------------------------------------------------------------------
def speaking_mode_ui():
    st.markdown("### 🗣️ Speaking Practice")
    st.markdown("Read the target sentence or speak freely. Then click **Transcribe & Evaluate**.")

    # Frase-alvo (pode ser alterada pelo aluno)
    st.session_state.target_sentence = st.text_input(
        "🎯 Target sentence (optional):",
        value=st.session_state.target_sentence,
        help="Say this sentence for targeted pronunciation practice, or leave as-is and speak freely."
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔈 Play target (TTS)"):
            try_tts(st.session_state.target_sentence)
    with col2:
        st.caption("Tip: Speak close to the mic and avoid background noise.")

    # Tentativa de importar libs do webrtc/ASR
    try:
        from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
        import av
    except Exception as e:
        st.error(f"🎤 Speaking requires `streamlit-webrtc`. Install and restart. Error: {e}")
        st.code("pip install streamlit-webrtc av")
        return

    # Buffer de áudio em sessão
    if "audio_frames" not in st.session_state:
        st.session_state.audio_frames = []

    # Processor que acumula os frames de áudio
    class AudioProcessor:
        def __init__(self) -> None:
            self._frames = []

        def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
            # guarda uma cópia mono (canal 0)
            pcm = frame.to_ndarray()
            if pcm.ndim == 2:
                # [channels, samples] -> pega canal 0
                pcm = pcm[0]
            st.session_state.audio_frames.append(pcm.copy())
            return frame

    rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    webrtc_ctx = webrtc_streamer(
        key="speaking",
        mode=WebRtcMode.SENDRECV,
        audio_receiver_size=256,  # menor latência
        video_receiver_size=0,
        rtc_configuration=rtc_config,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
        audio_processor_factory=AudioProcessor,
    )

    st.markdown("When you're done speaking, click the button below:")
    if st.button("🧠 Transcribe & Evaluate"):
        if not st.session_state.audio_frames:
            st.warning("No audio captured yet. Please press 'Start' and speak a sentence.")
            return

        # Salva o áudio bruto como WAV temporário (16kHz mono)
        import numpy as np
        import soundfile as sf

        audio = np.concatenate(st.session_state.audio_frames).astype("float32")
        st.session_state.audio_frames = []  # limpa buffer p/ próxima tentativa

        # Normaliza e salva
        if audio.size < 16000:
            st.warning("Very short audio. Try speaking for 2–3 seconds.")
        tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        sf.write(tmp_wav, audio, 48000)  # taxa típica do WebRTC

        # Transcreve com faster-whisper (fallback elegante se indisponível)
        try:
            from faster_whisper import WhisperModel
            # modelo pequeno e rápido
            asr_model_name = os.environ.get("ASR_MODEL", "small")
            model = WhisperModel(asr_model_name, device="cpu", compute_type="int8")
            segments, info = model.transcribe(tmp_wav, beam_size=1, vad_filter=True, word_timestamps=True)

            words = []
            word_probs = []
            transcript_text = ""
            for seg in segments:
                transcript_text += seg.text
                if seg.words:
                    for w in seg.words:
                        words.append(w.word)
                        if hasattr(w, "probability") and w.probability is not None:
                            word_probs.append(float(w.probability))

            transcript_text = transcript_text.strip()
            st.success("✅ Transcription complete")
            st.markdown(f"**📝 You said:** {transcript_text or '*<empty>*'}")

            # Scoring simples (0–100) com base nas probabilidades médias do ASR
            score = 0
            if word_probs:
                avg_p = float(np.clip(np.mean(word_probs), 0.0, 1.0))
                score = int(round(avg_p * 100))
            else:
                # Fallback: usa tamanho do áudio e presença de texto
                score = 50 if transcript_text else 0

            # Dicas baseadas em divergência do alvo
            tips = []
            target = st.session_state.target_sentence.strip()
            if target:
                # distância simples por palavras (mismatch rate)
                t_words = re.findall(r"\w+", target.lower())
                s_words = re.findall(r"\w+", transcript_text.lower())
                if t_words:
                    mismatches = sum(1 for i, w in enumerate(t_words) if i >= len(s_words) or s_words[i] != w)
                    mismatch_rate = mismatches / max(1, len(t_words))
                    if mismatch_rate > 0.5:
                        tips.append("Slow down and enunciate each word clearly.")
                    if len(s_words) < len(t_words) * 0.7:
                        tips.append("Don't drop endings (e.g., final consonants).")
                    if len(s_words) > len(t_words) * 1.3:
                        tips.append("Avoid adding extra words; keep it concise.")
                if not tips:
                    tips.append("Great! Keep a steady rhythm and natural intonation.")
            else:
                tips.append("Try short sentences first, then gradually increase length.")

            # Mostra relatório
            st.markdown(f"### 🧾 Pronunciation feedback")
            st.markdown(f"**Overall Score:** `{score}/100`")
            if words:
                st.markdown("**Detected words:** " + ", ".join(words[:12]) + ("…" if len(words) > 12 else ""))
            if tips:
                st.markdown("**Tips:**")
                for t in tips[:3]:
                    st.markdown(f"- {t}")

            # Gera resposta do tutor (resumo pedagógico) usando LLM
            pipe = load_pipeline()
            system_prompt = get_system_prompt(st.session_state.level, "Speaking")
            # Monta histórico mínimo para o LLM (apenas uma rodada com o texto falado)
            history = [{"role": "user", "content": f"My spoken sentence was: '{transcript_text}'. Target: '{target}'. Estimated score: {score}."}]
            prompt = build_prompt(system_prompt, history)
            raw = generate(pipe, prompt, st.session_state.max_new_tokens, st.session_state.temperature)
            coach = postprocess_response(raw)

            with st.chat_message("assistant", avatar="🧑‍🏫"):
                st.markdown(coach)

            # Repetição guiada (shadowing) com TTS da frase alvo
            st.markdown("#### 🔁 Shadowing (repeat after me)")
            st.markdown(f"**Model repetition:** {target or transcript_text or 'Let’s try a short sentence like: I go to school every day.'}")
            if target:
                try_tts(target)

        except Exception as e:
            st.error(f"ASR unavailable or failed: {e}")
            st.info(
                "Install local speech recognition to enable transcription:\n"
                "```bash\n"
                "pip install faster-whisper soundfile\n"
                "```\n"
                "If you can't install now, you can still use the other modes normally."
            )

# -----------------------------------------------------------------------------------
# Roteamento por modo
# -----------------------------------------------------------------------------------
if st.session_state.mode == "Speaking":
    speaking_mode_ui()
else:
    st.markdown(f"**🎯 Level:** `{st.session_state.level}` | **Mode:** `{st.session_state.mode}`")
    run_text_modes()

