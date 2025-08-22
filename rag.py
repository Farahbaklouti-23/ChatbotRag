# ===============================
# rag.py - Système RAG avec Chroma depuis Hugging Face Hub
# ===============================

import os
import shutil
import zipfile
from huggingface_hub import hf_hub_download

# --- Imports LangChain / Ollama ---
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain

# ===============================
# Configuration des modèles et paramètres
# ===============================
HF_REPO_ID = "Farahbaklouti-2002/political-rag-index"  # ⚠️ à remplacer par ton repo Hugging Face
HF_FILENAME = "chroma.zip"            # Le fichier que tu as uploadé
CHROMA_DIR = "chroma_index"           # Répertoire local où sera extrait l’index

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "mistral"
TEMPERATURE = 0.1
TOP_K = 10

# ===============================
# Prompt personnalisé
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
# Téléchargement & extraction de l’index Chroma
# ===============================
def download_and_extract_chroma(repo_id=HF_REPO_ID, filename=HF_FILENAME, local_dir=CHROMA_DIR):
    """
    Télécharge chroma.zip depuis Hugging Face Hub et l'extrait dans local_dir.
    """
    if not os.path.exists(local_dir) or not os.listdir(local_dir):
        print("📥 Téléchargement de l'index Chroma depuis Hugging Face...")
        downloaded_file = hf_hub_download(repo_id=repo_id, filename=filename)

        # Nettoyer l'ancien dossier si besoin
        if os.path.exists(local_dir):
            shutil.rmtree(local_dir)

        os.makedirs(local_dir, exist_ok=True)

        # Décompression
        with zipfile.ZipFile(downloaded_file, "r") as zip_ref:
            zip_ref.extractall(local_dir)

    return local_dir

# ===============================
# Initialisation du système RAG
# ===============================
def initialize_rag_system():
    """
    Initialise la chaîne RAG en utilisant l’index Chroma hébergé sur Hugging Face Hub.
    """
    # 1. Charger embeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # 2. Télécharger / extraire Chroma index
    local_chroma_dir = download_and_extract_chroma()

    # 3. Charger Chroma
    chroma_index = Chroma(
        persist_directory=local_chroma_dir,
        embedding_function=embeddings
    )

    # 4. Charger LLM
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=TEMPERATURE)

    # 5. Construire chaîne RAG
    return create_rag_chain(llm, chroma_index)

# ===============================
# Chaîne RAG conversationnelle
# ===============================
def create_rag_chain(llm, chroma_index):
    QA_PROMPT = PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=chroma_index.as_retriever(search_kwargs={"k": TOP_K}),
        return_source_documents=True,  # garde les sources si besoin debug
        combine_docs_chain_kwargs={"prompt": QA_PROMPT}
    )

# ===============================
# Génération de réponse
# ===============================
def generate_rag_response(rag_chain, prompt, chat_history):
    try:
        result = rag_chain.invoke({
            "question": prompt,
            "chat_history": chat_history
        })
        return result["answer"], result.get("source_documents", [])
    except Exception as e:
        return f"Erreur lors du traitement: {str(e)}", []
