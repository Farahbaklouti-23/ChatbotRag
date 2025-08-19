# Importation des bibliothèques nécessaires
import json                       # Pour lire les fichiers JSON contenant les données des personnalités
import os                         # Pour vérifier l'existence des fichiers et dossiers
from langchain_huggingface import HuggingFaceEmbeddings  # Pour générer les embeddings à partir du texte
from langchain_chroma import Chroma                       # Pour créer un index vectoriel Chroma
from langchain.chains import ConversationalRetrievalChain  # Pour créer une chaîne RAG conversationnelle
from langchain_ollama import ChatOllama                   # Pour utiliser le LLM Ollama (Mistral)
from langchain.prompts import PromptTemplate              # Pour définir un template de prompt personnalisé

# === Configuration des modèles et paramètres ===
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # Modèle d'embeddings pour vectorisation
OLLAMA_MODEL = "mistral"               # Modèle LLM utilisé
TEMPERATURE = 0.1                       # Température du LLM pour le contrôle de créativité
TOP_K = 10                              # Nombre de documents récupérés par la recherche

# === Prompt personnalisé pour le LLM ===
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

# === Fonction principale pour initialiser le système RAG ===
def initialize_rag_system(chroma_dir, json_file):
    """
    Initialise et retourne la chaîne RAG complète.
    - Crée les embeddings
    - Charge les textes depuis le JSON
    - Crée ou charge l'index Chroma
    - Initialise le LLM
    - Retourne la chaîne RAG
    """
    embeddings = load_embeddings()              # Chargement du modèle d'embeddings
    texts = load_texts(json_file)              # Chargement des textes à indexer
    chroma_index = build_or_load_chroma(texts, embeddings, chroma_dir)  # Création ou chargement de l'index
    llm = load_llm()                           # Chargement du LLM
    
    if not chroma_index or not llm:
        return None  # Si l'un des composants échoue, retourne None
        
    return create_rag_chain(llm, chroma_index)  # Crée et retourne la chaîne RAG

# === Fonction pour charger les embeddings ===
def load_embeddings():
    # Crée un objet HuggingFaceEmbeddings avec le modèle choisi
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# === Fonction pour charger et formater les textes depuis le fichier JSON ===
def load_texts(json_file):
    texts = []
    if not os.path.exists(json_file):
        return texts  # Retourne liste vide si fichier inexistant
        
    # Lecture du fichier ligne par ligne (chaque ligne = JSON d'une personnalité)
    with open(json_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                props = entry.get("properties", {})
                
                # Récupération des champs principaux
                name = entry.get("caption_latin", "Inconnu")
                birth = "Non spécifié"
                if "birthDate" in props and props["birthDate"]:
                    birth = props["birthDate"][0]
                
                positions = "; ".join(props.get("position", []))
                country = " ".join(props.get("country", []))
                sub_area = " ".join(props.get("subnationalArea", []))
                gender = " ".join(props.get("gender", []))
                
                # Construction d'une chaîne de texte formatée
                text_content = f"""
                Nom: {name} 
                Naissance: {birth}
                Genre: {gender}
                Pays: {country}
                Région: {sub_area}
                Postes: {positions}
                """
                texts.append(text_content)  # Ajoute le texte à la liste
            except:
                continue  # Ignore les lignes mal formées
    return texts

# === Fonction pour créer ou charger un index Chroma ===
def build_or_load_chroma(texts, embeddings, chroma_dir):
    # Si le dossier Chroma existe et contient des fichiers, on le charge
    if os.path.exists(chroma_dir) and os.listdir(chroma_dir):
        return Chroma(persist_directory=chroma_dir, embedding_function=embeddings)
    # Sinon, si des textes sont disponibles, crée un nouvel index
    elif texts:
        return Chroma.from_texts(texts, embeddings, persist_directory=chroma_dir)
    return None  # Retourne None si aucune donnée

# === Fonction pour charger le LLM (Ollama Mistral) ===
def load_llm():
    return ChatOllama(model=OLLAMA_MODEL, temperature=TEMPERATURE)

# === Fonction pour créer la chaîne RAG conversationnelle ===
def create_rag_chain(llm, chroma_index):
    # Création d'un PromptTemplate avec le template personnalisé
    QA_PROMPT = PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    
    # Création d'une chaîne RAG qui récupère les documents pertinents et interroge le LLM
    return ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=chroma_index.as_retriever(search_kwargs={"k": TOP_K}),
        return_source_documents=False,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT}
    )

# === Fonction pour générer la réponse RAG à partir d'une question et de l'historique ===
def generate_rag_response(rag_chain, prompt, chat_history):
    """
    - prompt: question de l'utilisateur
    - chat_history: historique de la conversation
    - Retourne: (réponse texte, documents sources)
    """
    try:
        # Appel de la chaîne RAG
        result = rag_chain.invoke({
            "question": prompt,
            "chat_history": chat_history
        })
        # Retourne la réponse et les documents sources (si disponibles)
        return result["answer"], result.get("source_documents", [])
    except Exception as e:
        # En cas d'erreur, retourne un message d'erreur
        return f"Erreur lors du traitement: {str(e)}", []
