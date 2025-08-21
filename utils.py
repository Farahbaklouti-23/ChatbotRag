# Importation des bibliothèques nécessaires
import json
import os
import streamlit as st

# === Fonction pour charger l'historique des conversations ===
def load_history(history_file):
    if os.path.exists(history_file):
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []

# === Fonction pour sauvegarder l'historique des conversations ===
def save_history(history, history_file):
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

# === Fonction pour obtenir du texte depuis un input Streamlit ===
def get_text_input():
    user_input = st.text_input("📝 Entrez votre texte :")
    if user_input.strip():
        return user_input.strip()
    else:
        st.toast("❌ Aucun texte saisi")
        return ""
