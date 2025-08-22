# ===============================
# main.py - Application Streamlit avec RAG
# ===============================

# --- Importation des bibliothèques nécessaires ---
import streamlit as st          # Framework pour créer des applications web interactives
import datetime                 # Pour gérer les dates et heures
import os                       # Pour interagir avec le système de fichiers
from rag import initialize_rag_system, generate_rag_response  # Fonctions personnalisées pour le RAG
from utils import load_history, save_history                  # Fonctions utilitaires pour l'historique
from langchain.schema import HumanMessage, AIMessage          # Classes pour représenter les messages


# --- Configuration des chemins des fichiers ---
CHROMA_DIR = "chroma_index"
JSON_FILE = "fichierfinal_nettoye.json"
HISTORY_FILE = "history.json"
CSS_FILE = "style.css"


# --- Chargement du CSS ---
def load_css():
    if os.path.exists(CSS_FILE):
        with open(CSS_FILE, encoding="utf-8") as f:
            return f.read()
    return ""


st.set_page_config(page_title="Agent RAG LangChain - PEP", layout="wide")
st.markdown(f"<style>{load_css()}</style>", unsafe_allow_html=True)


# --- Affichage du titre principal avec icône SVG ---
st.markdown("""
<div class="main-title">
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"></path>
    </svg>
    Agent RAG sur les Personnalités Politiques
</div>
""", unsafe_allow_html=True)


# --- Initialisation de l'état de session ---
def init_session_state():
    if "history" not in st.session_state:
        st.session_state.history = load_history(HISTORY_FILE)
    if "current_conversation" not in st.session_state:
        st.session_state.current_conversation = []
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "input_reset_counter" not in st.session_state:
        st.session_state.input_reset_counter = 0
    if "last_retrieved" not in st.session_state:
        st.session_state.last_retrieved = []


# --- Barre latérale (historique + reset) ---
def render_sidebar():
    with st.sidebar:
        try:
            st.image("excellianouv.png", width=250)
        except:
            st.info("Logo non trouvé")

        st.markdown("## Historique récent")
        if st.session_state.history:
            for i, (q, a) in enumerate(reversed(st.session_state.history[-10:])):
                st.markdown(
                    f'<div class="history-item" '
                    f'onclick=\'document.getElementById("input_prompt_{st.session_state.input_reset_counter}").value = "{q[:50]}";\'>'
                    f'<strong>{q[:45]}{"..." if len(q) > 45 else ""}</strong>'
                    f'<div style="font-size:0.8em;color:#6B7280;margin-top:4px;">{a[:60]}{"..." if len(a) > 60 else ""}</div>'
                    f'</div>', unsafe_allow_html=True
                )
        else:
            st.info("Aucune question posée.")

        st.markdown("---")
        if st.button("🗑️ Réinitialiser l'historique", use_container_width=True):
            st.session_state.history = []
            st.session_state.current_conversation = []
            st.session_state.processing = False
            st.session_state.input_reset_counter += 1
            st.session_state.last_retrieved = []
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
            st.rerun()


# --- Interface principale du chat ---
def render_chat_interface():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    # En-tête de chat
    st.markdown("""
    <div class="chat-header">
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
        </svg>
        <h2>Conversation en cours</h2>
    </div>
    """, unsafe_allow_html=True)

    # Messages affichés
    st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
    for role, message, timestamp in st.session_state.current_conversation:
        if role == "user":
            st.markdown(f'<div class="message user-message">{message}<div class="timestamp">{timestamp}</div></div>', unsafe_allow_html=True)
        elif role == "ai":
            st.markdown(f'<div class="message ai-message">{message.replace(chr(10), "<br>")}<div class="timestamp">{timestamp}</div></div>', unsafe_allow_html=True)

    # Indicateur de saisie
    if st.session_state.processing:
        st.markdown("""
        <div class="typing-indicator">
            <div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>
            <span>Le chatbot rédige sa réponse...</span>
        </div>
        """, unsafe_allow_html=True)

    # Sources récupérées
    if st.session_state.last_retrieved:
        with st.expander("📂 Documents source récupérés (Débogage)"):
            st.info(f"{len(st.session_state.last_retrieved)} documents pertinents trouvés")
            for i, doc in enumerate(st.session_state.last_retrieved):
                st.markdown(f"**Document {i+1}**")
                st.code(doc.page_content)
                st.divider()

    st.markdown('</div>', unsafe_allow_html=True)

    # Zone d'entrée + bouton
    with st.form(key=f'message_form_{st.session_state.input_reset_counter}'):
        cols = st.columns([8, 2])
        with cols[0]:
            current_key = f"input_prompt_{st.session_state.input_reset_counter}"
            st.text_input("Message", key=current_key, placeholder="Posez votre question...", label_visibility="collapsed")
        with cols[1]:
            submit_button = st.form_submit_button(" Rechercher", use_container_width=True, type="primary")

    st.markdown('</div>', unsafe_allow_html=True)
    return submit_button


# --- Logique : traitement de la question ---
def process_question():
    prompt = st.session_state.get(f"input_prompt_{st.session_state.input_reset_counter}", "")
    if not prompt.strip():
        st.warning("Veuillez saisir une question")
        return False

    user_timestamp = datetime.datetime.now().strftime("%H:%M")
    st.session_state.current_conversation.append(("user", prompt.strip(), user_timestamp))
    st.session_state.processing = True
    st.session_state.input_reset_counter += 1
    st.session_state.last_retrieved = []
    return True


# --- Logique : génération de la réponse ---
def handle_response(rag_chain):
    prompt = st.session_state.current_conversation[-1][1]
    chat_history = []
    for q, a in st.session_state.history:
        chat_history.append(HumanMessage(content=q))
        chat_history.append(AIMessage(content=a))

    try:
        response, source_docs = generate_rag_response(rag_chain, prompt, chat_history)
    except Exception as e:
        response, source_docs = f"⚠️ Erreur lors de la génération de la réponse : {e}", []

    ai_timestamp = datetime.datetime.now().strftime("%H:%M")
    st.session_state.current_conversation.append(("ai", response, ai_timestamp))
    st.session_state.history.append((prompt, response))
    save_history(st.session_state.history, HISTORY_FILE)
    st.session_state.last_retrieved = source_docs
    st.session_state.processing = False


# --- Point d'entrée principal ---
def main():
    init_session_state()
    render_sidebar()

    # Initialisation du RAG
    rag_chain = initialize_rag_system()
    if not rag_chain:
        st.error("❌ Erreur d'initialisation du système RAG (vérifiez Chroma et le JSON)")
        return

    # Interface + interaction
    submit_button = render_chat_interface()
    if submit_button:
        if process_question():
            handle_response(rag_chain)
            st.rerun()


# --- Exécution du programme ---
if __name__ == "__main__":
    main()
