# Importation des bibliothèques nécessaires
import streamlit as st          # Framework pour créer des applications web interactives
import datetime                 # Pour gérer les dates et heures
import os                       # Pour interagir avec le système de fichiers
from rag import initialize_rag_system, generate_rag_response  # Fonctions personnalisées pour le RAG (Retrieval-Augmented Generation)
from utils import load_history, save_history, speech_to_text       # Fonctions utilitaires pour l'historique et la transcription audio
from langchain.schema import HumanMessage, AIMessage              # Classes pour représenter les messages dans la conversation

# === Configuration des chemins des fichiers ===
CHROMA_DIR = "chroma_index"        # Répertoire où l'index Chroma est stocké
JSON_FILE = "fichierfinal_nettoye.json"  # Fichier JSON contenant les données sources
HISTORY_FILE = "history.json"      # Fichier pour sauvegarder l'historique des conversations
CSS_FILE = "style.css"             # Fichier CSS pour styliser l'interface

# === Fonction pour charger le CSS ===
def load_css():
    # Ouvre le fichier CSS et retourne son contenu
    with open(CSS_FILE) as f:
        return f.read()

# Injection du CSS dans Streamlit pour styliser l'application
st.markdown(f"<style>{load_css()}</style>", unsafe_allow_html=True)

# === Configuration générale de la page Streamlit ===
st.set_page_config(
    page_title="Agent RAG LangChain - PEP",  # Titre de la page
    layout="wide"                            # Largeur de la page étendue
)

# === Affichage du titre principal avec icône SVG ===
st.markdown("""
<div class="main-title">
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"></path>
    </svg>
    Agent RAG sur les Personnalités Politiques
</div>
""", unsafe_allow_html=True)

# === Initialisation des variables d'état de session Streamlit ===
def init_session_state():
    # Historique des conversations précédentes
    if "history" not in st.session_state:
        st.session_state.history = load_history(HISTORY_FILE)
        
    # Conversation en cours (messages utilisateurs et AI)
    if "current_conversation" not in st.session_state:
        st.session_state.current_conversation = []

    # Indicateur de traitement en cours
    if "processing" not in st.session_state:
        st.session_state.processing = False

    # Texte transcrit depuis l'audio
    if "transcribed_text" not in st.session_state:
        st.session_state.transcribed_text = ""

    # Compteur pour réinitialiser les inputs
    if "input_reset_counter" not in st.session_state:
        st.session_state.input_reset_counter = 0

    # Indicateur si l'utilisateur enregistre de l'audio
    if "recording" not in st.session_state:
        st.session_state.recording = False

    # Derniers documents récupérés par le RAG
    if "last_retrieved" not in st.session_state:
        st.session_state.last_retrieved = []

# === Barre latérale avec l'historique des conversations ===
def render_sidebar():
    with st.sidebar:
        # Affichage du logo
        try:
            st.image("excellianouv.png", width=250)
        except:
            st.info("Logo non trouvé")
        
        st.markdown("## Historique récent")
        
        # Affichage des 10 dernières questions/réponses
        if st.session_state.history:
            for i, (q, a) in enumerate(reversed(st.session_state.history[-10:])):
                with st.container():
                    st.markdown(f'<div class="history-item" onclick=\'document.getElementById("input_prompt_{st.session_state.input_reset_counter}").value = "{q[:50]}";\'>' 
                                f'<strong>{q[:45]}{"..." if len(q) > 45 else ""}</strong>'
                                f'<div style="font-size:0.8em;color:#6B7280;margin-top:4px;">{a[:60]}{"..." if len(a) > 60 else ""}</div>'
                                f'</div>', unsafe_allow_html=True)
        else:
            st.info("Aucune question posée.")

        st.markdown("---")
        # Bouton pour réinitialiser l'historique
        if st.button("🗑️ Réinitialiser l'historique", use_container_width=True):
            st.session_state.history = []
            st.session_state.current_conversation = []
            st.session_state.processing = False
            st.session_state.transcribed_text = ""
            st.session_state.input_reset_counter += 1
            st.session_state.last_retrieved = []
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
            st.rerun()

# === Interface principale du chat ===
def render_chat_interface():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # En-tête de la conversation
    st.markdown("""
    <div class="chat-header">
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
        </svg>
        <h2>Conversation en cours</h2>
    </div>
    """, unsafe_allow_html=True)

    # Affichage des messages de la conversation
    st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
    for role, message, timestamp in st.session_state.current_conversation:
        if role == "user":
            st.markdown(f"""
            <div class="message user-message">
                {message}
                <div class="timestamp">{timestamp}</div>
            </div>
            """, unsafe_allow_html=True)
        elif role == "ai":
            st.markdown(f"""
            <div class="message ai-message">
                {message.replace(chr(10), '<br>')}
                <div class="timestamp">{timestamp}</div>
            </div>
            """, unsafe_allow_html=True)

    # Indicateur "en cours de rédaction" si AI répond
    if st.session_state.processing:
        st.markdown("""
        <div class="typing-indicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <span>Le chatbot rédige sa réponse...</span>
        </div>
        """, unsafe_allow_html=True)

    # Section de débogage pour les documents source récupérés
    if st.session_state.last_retrieved:
        with st.expander(" Documents source récupérés (Débogage)"):
            st.info(f"{len(st.session_state.last_retrieved)} documents pertinents trouvés")
            for i, doc in enumerate(st.session_state.last_retrieved):
                st.markdown(f"**Document {i+1}**")
                st.code(doc.page_content)
                st.divider()

    st.markdown('<div style="flex-grow: 1;"></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Zone d'entrée de texte et boutons
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    with st.form(key=f'message_form_{st.session_state.input_reset_counter}'):
        cols = st.columns([7, 2.5, 2.5])
        with cols[0]:
            current_key = f"input_prompt_{st.session_state.input_reset_counter}"
            default_value = st.session_state.transcribed_text if st.session_state.transcribed_text else ""
            user_input = st.text_input(
                "Message",
                key=current_key,
                value=default_value,
                placeholder="Posez votre question...",
                label_visibility="collapsed"
            )
        with cols[1]:
            submit_button = st.form_submit_button(
                " Rechercher", 
                use_container_width=True,
                type="primary"
            )
        with cols[2]:
            voice_button = st.form_submit_button(
                " Audio", 
                use_container_width=True,
                type="primary"
            )
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    return submit_button, voice_button

# === Logique pour traiter la question utilisateur ===
def process_question():
    prompt = st.session_state.get(f"input_prompt_{st.session_state.input_reset_counter}", "")
    if not prompt:
        st.warning("Veuillez saisir une question")
        return False

    # Ajouter le message utilisateur à la conversation
    user_timestamp = datetime.datetime.now().strftime("%H:%M")
    st.session_state.current_conversation.append(("user", prompt, user_timestamp))
    st.session_state.processing = True
    st.session_state.input_reset_counter += 1
    st.session_state.transcribed_text = ""
    st.session_state.last_retrieved = []
    return True

# === Logique pour générer la réponse via le RAG ===
def handle_response(rag_chain):
    prompt = st.session_state.current_conversation[-1][1]

    # Conversion de l'historique en format HumanMessage / AIMessage
    chat_history = []
    for q, a in st.session_state.history:
        chat_history.append(HumanMessage(content=q))
        chat_history.append(AIMessage(content=a))
    
    # Génération de la réponse et récupération des documents
    response, source_docs = generate_rag_response(rag_chain, prompt, chat_history)
    
    # Ajouter la réponse AI à la conversation
    ai_timestamp = datetime.datetime.now().strftime("%H:%M")
    st.session_state.current_conversation.append(("ai", response, ai_timestamp))
    st.session_state.history.append((prompt, response))
    save_history(st.session_state.history, HISTORY_FILE)
    st.session_state.last_retrieved = source_docs
    st.session_state.processing = False

# === Point d'entrée principal de l'application ===
def main():
    init_session_state()        # Initialiser l'état de session
    render_sidebar()            # Afficher la barre latérale avec historique
    
    # Initialiser le système RAG
    rag_chain = initialize_rag_system(CHROMA_DIR, JSON_FILE)
    if not rag_chain:
        st.error("Erreur d'initialisation du système RAG")
        return
    
    # Afficher l'interface du chat
    submit_button, voice_button = render_chat_interface()
    
    # Gestion des boutons
    if voice_button:
        st.session_state.transcribed_text = speech_to_text()
        st.rerun()
    
    if submit_button:
        if process_question():
            handle_response(rag_chain)
            st.rerun()

# === Exécution du programme ===
if __name__ == "__main__":
    main()
