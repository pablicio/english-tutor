import streamlit as st

st.title("📚 English Tutor Bot")
st.markdown("*Practice English with AI-powered feedback*")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    level = st.selectbox(
        "English Level",
        ["Beginner", "Intermediate", "Advanced"]
    )
    
    mode = st.selectbox(
        "Practice Mode",
        ["Conversation", "Grammar", "Writing", "Vocabulary"]
    )
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Inicializar histórico
if "messages" not in st.session_state:
    st.session_state.messages = []

# System prompt FORTE
def get_system_prompt(level, mode):
    base = f"You are an expert English tutor for {level} students. "
    
    if mode == "Conversation":
        return base + "Have natural conversation. When student makes mistakes, gently show the correct form. Ask follow-up questions. Be encouraging!"
    elif mode == "Grammar":
        return base + "Focus on grammar. Explain rules with examples. Help student understand why the correction is needed."
    elif mode == "Writing":
        return base + "Review writing. Point out what's good first, then suggest improvements. Explain each correction."
    else:
        return base + "Build vocabulary. Explain word meanings, give examples, teach synonyms. Make learning fun!"

# Carregar pipeline (mais compatível)
@st.cache_resource
def load_pipeline():
    st.info("⏳ Loading model...")
    from transformers import pipeline
    return pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device_map=None,  # Sem device_map
    )

try:
    pipe = load_pipeline()
    
    # Exibir histórico
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # Input
    if user_text := st.chat_input("Type here..."):
        st.session_state.messages.append({"role": "user", "content": user_text})
        
        with st.chat_message("user"):
            st.write(user_text)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Preparar mensagens para o modelo
                messages = [
                    {"role": "system", "content": get_system_prompt(level, mode)},
                ]
                
                # Adicionar histórico (últimas 4)
                for msg in st.session_state.messages[-4:]:
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
                
                # Gerar resposta
                response = pipe(
                    messages,
                    max_new_tokens=150,
                    temperature=0.7,
                )
                
                # Extrair texto - funciona com várias versões
                try:
                    # Tentar formato novo (lista de dicts)
                    if isinstance(response[0]["generated_text"], list):
                        bot_response = response[0]["generated_text"][-1]["content"]
                    else:
                        # Formato string
                        bot_response = response[0]["generated_text"]
                        # Remove duplicação
                        if user_text in bot_response:
                            bot_response = bot_response.split(user_text)[-1].strip()
                except:
                    bot_response = str(response[0]["generated_text"])
                
                st.write(bot_response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": bot_response
                })
    
except Exception as e:
    st.error(f"Error: {e}")
    st.info("""
    **Troubleshooting:**
    
    1. Update transformers:
    ```
    pip install --upgrade transformers torch
    ```
    
    2. Or reinstall everything:
    ```
    pip uninstall transformers torch streamlit -y
    pip install transformers torch streamlit
    ```
    """)
