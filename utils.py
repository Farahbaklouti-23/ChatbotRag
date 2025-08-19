# Importation des bibliothèques nécessaires
import json                 # Pour lire et écrire des fichiers JSON (historique des conversations)
import os                   # Pour vérifier l'existence des fichiers
import speech_recognition as sr  # Pour la reconnaissance vocale (transcription audio)
import streamlit as st      # Framework pour créer des interfaces web interactives

# === Fonction pour charger l'historique des conversations ===
def load_history(history_file):
    # Vérifie si le fichier existe
    if os.path.exists(history_file):
        try:
            # Ouvre le fichier et charge le JSON
            with open(history_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            # Si erreur lors de la lecture, retourne une liste vide
            return []
    # Si le fichier n'existe pas, retourne une liste vide
    return []

# === Fonction pour sauvegarder l'historique des conversations ===
def save_history(history, history_file):
    # Écrit l'historique dans un fichier JSON
    with open(history_file, "w", encoding="utf-8") as f:
        # Indentation pour lisibilité et ensure_ascii=False pour conserver les accents/français
        json.dump(history, f, indent=2, ensure_ascii=False)

# === Fonction pour transcrire de l'audio en texte ===
def speech_to_text():
    r = sr.Recognizer()  # Crée un objet Recognizer pour la reconnaissance vocale
    try:
        # Utilise le microphone comme source audio
        with sr.Microphone() as source:
            st.session_state.recording = True  # Indique que l'enregistrement est en cours
            st.toast("Écoute en cours... Parlez maintenant")  # Message informatif à l'utilisateur
            # Écoute le son avec un timeout et une limite de durée de phrase
            audio = r.listen(source, timeout=5, phrase_time_limit=30)
        
        try:
            # Tente de reconnaître le texte en français via l'API Google
            text = r.recognize_google(audio, language='fr-FR')
            return text  # Retourne le texte reconnu
        except sr.UnknownValueError:
            # Cas où l'audio n'est pas compréhensible
            st.toast("Impossible de comprendre l'audio", icon="❌")
        except sr.RequestError as e:
            # Cas où l'API Google rencontre une erreur
            st.toast(f"Erreur de service : {e}", icon="❌")
    except Exception as e:
        # Cas général d'erreur lors de l'accès au microphone
        st.toast(f"Erreur d'accès au microphone: {str(e)}", icon="❌")
    finally:
        # Assure que l'état d'enregistrement est réinitialisé
        st.session_state.recording = False
    
    # Retourne une chaîne vide si la transcription échoue
    return ""
