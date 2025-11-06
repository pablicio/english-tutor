import streamlit as st
from transformers import pipeline
import re


st.set_page_config(
    page_title="English Tutor 🎓 (Teacher Shara)",
    page_icon="🌍",
    layout="wide"
)


st.title("🌍 English Tutor AI (Teacher Shara)   ")
st.markdown("Practice English with AI-powered corrections and explanations (100% Free - Hugging Face)")


# Carregar modelo conversacional
@st.cache_resource
def load_model():
    conversation = pipeline(
        "conversational",
        model="facebook/blenderbot-400M-distill",
        device=-1
    )
    return conversation


try:
    chat_model = load_model()
except Exception as e:
    st.error(f"Erro ao carregar modelo: {e}")
    st.stop()


# Inicializar histórico
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []


# Exibir histórico
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div style='background: #e3f2fd; padding: 12px; border-radius: 8px; margin: 10px 0; text-align: right;'>
            <strong style='color: #1565c0;'>👤 You:</strong><br>
            {msg['content']}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background: #f3e5f5; padding: 12px; border-radius: 8px; margin: 10px 0;'>
            <strong style='color: #6a1b9a;'>🎓 Teacher Shara:</strong><br>
            {msg['content']}
        </div>
        """, unsafe_allow_html=True)


# Input do usuário
col1, col2 = st.columns([5, 1])

with col1:
    user_input = st.text_input(
        "Type something in English:",
        placeholder="e.g., I going to the store yesterday...",
        label_visibility="collapsed"
    )

with col2:
    send_button = st.button("📤 Send", use_container_width=True)


# Banco de regras de correção (CORRIGIDO)
CORRECTIONS_DB = {
    r'\bi\s': {
        'correction': 'I',
        'rule': 'Always capitalize "I" in English'
    },
    r'\b(he|she|it)\s+(go|like|want|need|have|think|see|know|get|make|come|take)\s': {
        'correction': 'add "s"',
        'rule': 'Third person singular needs "s": he goes, she likes'
    },
    r'\b(they|we)\s+is\b': {
        'correction': 'are',
        'rule': 'Plural subjects use "are", not "is"'
    },
    r'\b(dont|cant|isnt|wasnt|wont)\b': {
        'correction': "use apostrophe",
        'rule': "Use contractions: don't, can't, isn't, wasn't, won't"
    },
    r'\byesterday.{0,20}going\b': {
        'correction': 'past simple',
        'rule': 'With "yesterday" use past simple (went), not present continuous'
    },
}


def detect_errors(text):
    """Detecta erros e retorna dicionário com sugestões"""
    text_lower = text.lower()
    
    for pattern, correction_data in CORRECTIONS_DB.items():
        if re.search(pattern, text_lower):
            return correction_data  # Retorna DICIONÁRIO, não lista
    
    return None  # Sem erro encontrado


def generate_tutor_response(user_text):
    """Gera resposta pedagógica"""
    
    error_data = detect_errors(user_text)
    
    # Se encontrou erros
    if error_data is not None:
        response = f"""**Student wrote:** "{user_text}"\n\n"""
        response += f"""**Issue detected:** {error_data['rule']}\n\n"""
        response += f"""**Correction:** Use "{error_data['correction']}"\n\n"""
        
        # Adicionar exemplos práticos
        if 'capitalize' in error_data['rule'].lower():
            response += """**Examples:** "I like pizza" ✓ | "I am going to school" ✓\n\n"""
        elif 'third person' in error_data['rule'].lower():
            response += """**Examples:** "She likes cats" ✓ | "He wants coffee" ✓ | "It goes away" ✓\n\n"""
        elif 'plural' in error_data['rule'].lower():
            response += """**Examples:** "They are happy" ✓ | "We are students" ✓\n\n"""
        elif 'apostrophe' in error_data['rule'].lower():
            response += """**Examples:** "I don't know" ✓ | "She can't swim" ✓ | "He isn't here" ✓\n\n"""
        elif 'past simple' in error_data['rule'].lower():
            response += """**Examples:** "I went yesterday" ✓ | "She studied last week" ✓\n\n"""
        
        response += """**Keep practicing!** 💪"""
        return response
    
    else:
        # Frase correta - retornar feedback positivo
        return f"""✅ **Great job!** Your sentence is correct!\n\n**Sentence:** "{user_text}"\n\n**Tip:** Try making it longer or using more complex structures next time! 🚀"""


# Processar mensagem
if send_button and user_input.strip():
    # Adicionar mensagem do usuário
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    # Gerar resposta
    with st.spinner("🤔 Teacher Shara is thinking..."):
        tutor_response = generate_tutor_response(user_input)
    
    # Adicionar resposta do tutor
    st.session_state.messages.append({
        "role": "tutor",
        "content": tutor_response
    })
    
    st.rerun()


# Seção de dicas
st.divider()

with st.expander("💡 Common Mistakes & Rules"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("❌ Common Errors")
        st.write("""
        1. **i going** → **I am going**
           Capitalize I, use am for present continuous
        
        2. **she go** → **she goes**
           Third person singular needs 's'
        
        3. **they is** → **they are**
           Plural uses 'are', not 'is'
        
        4. **yesterday going** → **yesterday went**
           Past time = past tense
        
        5. **dont** → **don't**
           Contractions need apostrophe
        """)
    
    with col2:
        st.subheader("✅ Quick Reference")
        st.write("""
        **Present Simple:**
        - I/you/we/they + verb
        - he/she/it + verb + s
        
        **Past Simple:**
        - Regular: -ed (walked, played)
        - Irregular: went, saw, did, was
        
        **Present Continuous:**
        - be (am/is/are) + verb + -ing
        
        **Key Contractions:**
        - don't, can't, isn't, aren't
        - I'm, he's, they're, we've
        """)


# Botão para limpar
if st.button("🔄 Start New Conversation"):
    st.session_state.messages = []
    st.session_state.conversation_history = []
    st.rerun()


st.caption("Made with ❤️ using Streamlit + BlenderBot (Free Hugging Face) | No API keys needed! 🚀")
