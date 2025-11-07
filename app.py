# app.py — English Tutor Bot v2
# Thiago Pablicio Edition ✨
# ----------------------------------------------
# Recursos:
# ✅ Modelo automático (usa Phi-3 se disponível, senão TinyLlama)
# ✅ Prompts pedagógicos (Grammar, Conversation, Writing, Vocabulary)
# ✅ Correção automática de frases curtas
# ✅ Pós-processamento inteligente (remove USER/ASSISTANT)
# ✅ TTS opcional (gTTS)
# ✅ UX com avatares e cabeçalhos
# ----------------------------------------------

import os
import re
import json
import tempfile
import streamlit as st

st.set_page_config(page_title="English Tutor Bot", page_icon="🧑‍🏫", layout="centered")

# ----------------------------------------------
# Cabeçalho
# ----------------------------------------------
st.title("📚 English Tutor Bot")
st.caption("*Learn, practice, and get real-time feedback from your AI English tutor.*")

# ----------------------------------------------
# Sidebar Settings
# ----------------------------------------------
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

    level = st.selectbox("English Level", ["Beginner", "Intermediate", "Advanced"], key="level")
    mode = st.selectbox("Practice Mode", ["Conversation", "Grammar", "Writing", "Vocabulary"], key="mode")
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

# ----------------------------------------------
# Estado inicial
# ----------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "stats" not in st.session_state:
    st.session_state.stats = {"turns": 0}

# ----------------------------------------------
# Prompts pedagógicos
# ----------------------------------------------
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
    else:
        return base + (
            " When teaching vocabulary, answer in this structure:\n"
            "1️⃣ **Word Meaning**\n2️⃣ **Synonyms**\n3️⃣ **Example Sentence**\n"
            "Then ask: 'Can you use this word in your own sentence?'"
        )

# ----------------------------------------------
# Construir prompt concatenado
# ----------------------------------------------
def build_prompt(system_prompt, history):
    prompt = f"SYSTEM: {system_prompt}\n"
    for msg in history:
        role = "ASSISTANT" if msg["role"] == "assistant" else "USER"
        content = msg["content"].strip()
        prompt += f"{role}: {content}\n"
    prompt += "ASSISTANT:"
    return prompt

# ----------------------------------------------
# Carregar modelo automaticamente
# ----------------------------------------------
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

# ----------------------------------------------
# Geração + limpeza
# ----------------------------------------------
def generate(pipe, prompt, max_new_tokens, temperature):
    out = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=pipe.tokenizer.eos_token_id if hasattr(pipe, "tokenizer") else None,
    )
    text = out[0]["generated_text"]
    # pega só a parte útil
    if "ASSISTANT:" in text:
        text = text.split("ASSISTANT:", 1)[-1]
    # limpa rótulos
    text = re.sub(r"(USER|ASSISTANT):", "", text)
    return text.strip()

def postprocess_response(resp):
    resp = re.sub(r"(USER|ASSISTANT):", "", resp)
    resp = re.sub(r"\[.*?\]", "", resp)
    resp = re.sub(r"\s{3,}", "\n\n", resp)
    resp = resp.strip()
    return resp

# ----------------------------------------------
# TTS (opcional)
# ----------------------------------------------
def try_tts(text):
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

# ----------------------------------------------
# Render histórico
# ----------------------------------------------
st.markdown(f"**🎯 Level:** `{st.session_state.level}` | **Mode:** `{st.session_state.mode}`")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍🏫" if msg["role"] == "assistant" else "🧑‍💻"):
        st.markdown(msg["content"])

# ----------------------------------------------
# Entrada do usuário
# ----------------------------------------------
if user_text := st.chat_input("Type here..."):

    # Autoajuste para mensagens curtas
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

