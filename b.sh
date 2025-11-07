# 1. Limpar cache do pip
pip cache purge

# 2. Desinstalar tudo
pip uninstall transformers torch streamlit -y

# 3. Instalar tudo novamente (versões compatíveis)
pip install torch transformers==4.45.0 streamlit

# 4. Verificar
python -c "from transformers import pipeline; print('OK!')"

# 5. Executar
streamlit run app.py
