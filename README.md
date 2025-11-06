
# 💬 Chatbot de Idiomas com IA

Um mini-projeto educacional que utiliza **Streamlit** e **Hugging Face Transformers** para criar um tutor de idiomas interativo.

## 🚀 Como executar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deploy no Hugging Face Spaces

1. Crie um novo Space em [https://huggingface.co/spaces](https://huggingface.co/spaces)
   - SDK: **Streamlit**
   - Nome: por exemplo `language-tutor-bot`
2. Faça upload dos arquivos (`app.py`, `requirements.txt`, `README.md`)
3. Aguarde a build automática do Hugging Face

## 🧠 O que o app faz
- O aluno escreve algo no idioma alvo (ex: inglês, espanhol etc.)
- O chatbot (modelo `facebook/blenderbot-400M-distill`) responde com:
  - Correções de gramática e vocabulário
  - Explicações breves
  - Sugestões de frases alternativas

## 🔮 Próximos passos
- Adicionar reconhecimento de fala (`streamlit-webrtc`)
- Avaliação automática de nível do aluno (A1–C1)
- Suporte multilíngue com tradução (`Helsinki-NLP`)

---
Desenvolvido com ❤️ por Shara
