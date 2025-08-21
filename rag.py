# ===============================
# rag.py - Système RAG pour personnalités politiques
# ===============================

# --- Imports nécessaires ---
import json  # Pour lire les fichiers JSON contenant les données
import os    # Pour vérifier l'existence de fichiers et dossiers

# --- Imports mis à jour pour LangChain sans dépréciation ---
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate


# ===============================
# Configuration des modèles et paramètres
# ===============================
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Modèle utilisé pour les embeddings
OLLAMA_MODEL = "mistral"              # Modèle LLM Ollama
TEMPERATURE = 0.1                      # Température pour contrôler la créativité du LLM
TOP_K = 10                             # Nombre de documents récupérés pour chaque question

# ===============================
# Prompt personnalisé pour le LLM
# ===============================
CUSTOM_PROMPT_TEMPLATE = """
Tu es un expert spécialisé dans les personnalités politiques mondiales.  

**Extrais les informations suivantes si disponibles dans le contexte:**
- Nom complet
- Date de naissance
- Genre
- Pays
- Région/Subdivision
- Postes politiques occupés

**Règles:**
1. Utilise UNIQUEMENT les informations du contexte
2. Si la question NE CONCERNE PAS une personnalité politique, réponds UNIQUEMENT:
   "Désolé, je ne peux répondre qu'aux questions concernant des personnalités politiques."
3. Si la personnalité demandée n'est pas présente dans la base, réponds:
   "Le nom demandé n'est pas disponible dans la base."
4. Pour les champs manquants, indique "Non spécifié"
5. Présente les postes politiques sous forme de liste à puces
6. Structure la réponse clairement avec les sections suivantes:

Structure de réponse OBLIGATOIRE:
Nom: [nom complet]
Naissance: [date de naissance]
Genre: [genre]
Pays: [pays]
Région: [région/subdivision]
Postes:
• [poste 1]
• [poste 2]
• ...

Contexte:
{context}

Question: {question}

Réponse:
"""

# ===============================
# Fonction principale pour initialiser le système RAG
# ===============================
def initialize_rag_system(chroma_dir, json_file):
    """
    Initialise et retourne la chaîne RAG complète.
    1. Charge le modèle d'embeddings
    2. Charge et formate les textes depuis le fichier JSON
    3. Crée ou charge l'index Chroma
    4. Charge le LLM Ollama
    5. Retourne la chaîne RAG prête à être utilisée
    """
    embeddings = load_embeddings()  # Chargement du modèle d'embeddings
    texts = load_texts(json_file)  # Chargement des textes
    chroma_index = build_or_load_chroma(texts, embeddings, chroma_dir)  # Création ou chargement de l'index
    llm = load_llm()  # Chargement du LLM

    if not chroma_index or not llm:
        return None  # Retourne None si un composant échoue

    return create_rag_chain(llm, chroma_index)  # Création de la chaîne RAG

# ===============================
# Fonction pour charger les embeddings
# ===============================
def load_embeddings():
    """
    Crée et retourne un objet HuggingFaceEmbeddings
    qui va transformer le texte en vecteurs pour l'indexation.
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# ===============================
# Fonction pour charger et formater les textes depuis le JSON
# ===============================
def load_texts(json_file):
    """
    Lit le fichier JSON et retourne une liste de textes formatés.
    Chaque ligne du fichier JSON correspond à une personnalité.
    """
    texts = []
    if not os.path.exists(json_file):
        return texts  # Retourne une liste vide si le fichier n'existe pas

    with open(json_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                props = entry.get("properties", {})

                # Extraction des informations principales
                name = entry.get("caption_latin", "Inconnu")
                birth = props.get("birthDate", ["Non spécifié"])[0] if props.get("birthDate") else "Non spécifié"
                positions = "; ".join(props.get("position", []))
                country = " ".join(props.get("country", []))
                sub_area = " ".join(props.get("subnationalArea", []))
                gender = " ".join(props.get("gender", []))

                # Construction d'une chaîne texte formatée
                text_content = f"""
                Nom: {name}
                Naissance: {birth}
                Genre: {gender}
                Pays: {country}
                Région: {sub_area}
                Postes: {positions}
                """
                texts.append(text_content)
            except:
                continue  # Ignore les lignes mal formées
    return texts

# ===============================
# Fonction pour créer ou charger l'index Chroma
# ===============================
def build_or_load_chroma(texts, embeddings, chroma_dir):
    """
    Crée un nouvel index Chroma si aucun n'existe.
    Sinon, charge l'index existant depuis le dossier.
    """
    if os.path.exists(chroma_dir) and os.listdir(chroma_dir):
        return Chroma(persist_directory=chroma_dir, embedding_function=embeddings)
    elif texts:
        return Chroma.from_texts(texts, embeddings, persist_directory=chroma_dir)
    return None

# ===============================
# Fonction pour charger le LLM Ollama
# ===============================
def load_llm():
    """
    Initialise le modèle Ollama Mistral avec la température spécifiée.
    """
    return ChatOllama(model=OLLAMA_MODEL, temperature=TEMPERATURE)

# ===============================
# Fonction pour créer la chaîne RAG conversationnelle
# ===============================
def create_rag_chain(llm, chroma_index):
    """
    Crée une chaîne RAG:
    1. Récupère les documents pertinents via Chroma
    2. Interroge le LLM avec un PromptTemplate personnalisé
    """
    QA_PROMPT = PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=chroma_index.as_retriever(search_kwargs={"k": TOP_K}),
        return_source_documents=False,  # Pas besoin des documents sources pour l'affichage
        combine_docs_chain_kwargs={"prompt": QA_PROMPT}
    )

# ===============================
# Fonction pour générer une réponse RAG
# ===============================
def generate_rag_response(rag_chain, prompt, chat_history):
    """
    Génère une réponse RAG pour une question donnée.
    - prompt: question de l'utilisateur
    - chat_history: historique de la conversation
    Retourne: (réponse texte, documents sources)
    """
    try:
        result = rag_chain.invoke({
            "question": prompt,
            "chat_history": chat_history
        })
        return result["answer"], result.get("source_documents", [])
    except Exception as e:
        return f"Erreur lors du traitement: {str(e)}", []
