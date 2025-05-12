import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
import config
from config import OPENAI_API_KEY, RAG_EMBEDDING_MODEL_NAME, CHROMA_DB_PATH, SAMPLE_DOCUMENTS_PATH, LOCAL_LLM_INSTANCE
from rlpt import Sanitizer, PlaceholderManager

        

def load_and_split_documents(file_path: str) -> List[Document]:
    """Loads documents from a file and splits them into chunks."""
    if not os.path.exists(file_path):
        print(f"Error: Sample documents file not found at {file_path}")
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # A simple way to create multiple documents if separated by double newlines and "Document X:"
    raw_docs = content.split("\n\nDocument")[1:] # Skip potential empty first element
    docs_to_process = []
    for i, raw_doc_content in enumerate(raw_docs):
        # Re-add "Document" prefix for context if needed, or just use the content
        doc_content = raw_doc_content.split(":", 1)[1].strip() if ":" in raw_doc_content else raw_doc_content.strip()
        docs_to_process.append(Document(page_content=doc_content, metadata={"source": f"sample_doc_{i+1}"}))

    if not docs_to_process: # Fallback if the above split doesn't work
         docs_to_process = [Document(page_content=content, metadata={"source": "sample_documents_file"})]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs_to_process)
    print(f"Loaded and split {len(docs_to_process)} documents into {len(splits)} chunks.")
    return splits

def ingest_data(sanitize_during_ingestion: bool = True):
    """
    Loads data, (optionally) sanitizes it, creates embeddings, and stores them in ChromaDB.
    Args:
        sanitize_during_ingestion (bool): If True, sanitizes documents before embedding.
    """
    print("Starting data ingestion process...")

    if not OPENAI_API_KEY:
        print("OpenAI API key not configured. Cannot create embeddings.")
        return

    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(
        model=RAG_EMBEDDING_MODEL_NAME,
        openai_api_key=OPENAI_API_KEY
    )

    # Load and split documents
    document_chunks = load_and_split_documents(SAMPLE_DOCUMENTS_PATH)
    if not document_chunks:
        print("No document chunks to process. Exiting ingestion.")
        return

    sanitized_chunks_for_embedding = []

    if sanitize_during_ingestion:
        print("Sanitizing documents before embedding...")
        # This requires the local LLM to be initialized and passed to the Sanitizer
        # For this script, we'd need to handle LLM loading or assume it's passed
        # Let's assume we need to load it if config.LOCAL_LLM_INSTANCE is None
        if config.LOCAL_LLM_INSTANCE is None:
            try:
                from langchain_community.llms import Ollama # Local import
                config.LOCAL_LLM_INSTANCE = Ollama(
                    base_url=config.OLLAMA_BASE_URL,
                    model=config.LOCAL_NER_MODEL_NAME
                )
                print("Local LLM for sanitization initialized in ingest_data.")
            except ImportError:
                print("Error: Ollama not installed. Cannot sanitize during ingestion. `pip install ollama`")
                print("Proceeding without sanitization for ingestion.")
                sanitize_during_ingestion = False # Override
            except Exception as e:
                print(f"Error initializing local LLM for sanitization: {e}")
                print("Proceeding without sanitization for ingestion.")
                sanitize_during_ingestion = False # Override

        if sanitize_during_ingestion and config.LOCAL_LLM_INSTANCE:
            placeholder_manager = PlaceholderManager() # Use default path
            sanitizer = Sanitizer(local_llm=config.LOCAL_LLM_INSTANCE, placeholder_manager=placeholder_manager)

            for i, chunk in enumerate(document_chunks):
                print(f"Sanitizing chunk {i+1}/{len(document_chunks)}...")
                sanitized_content, _ = sanitizer.sanitize_text(chunk.page_content)
                sanitized_chunks_for_embedding.append(
                    Document(page_content=sanitized_content, metadata=chunk.metadata)
                )
                if i < 3 : # Log first few sanitizations for review
                    print(f"Original: {chunk.page_content[:100]}...")
                    print(f"Sanitized: {sanitized_content[:100]}...")
            print("Document sanitization complete.")
        else: # If sanitization was skipped or failed
             sanitized_chunks_for_embedding = document_chunks
             print("Skipping sanitization during ingestion or LLM not available.")
    else:
        sanitized_chunks_for_embedding = document_chunks
        print("Skipping sanitization during ingestion as per configuration.")

    if not sanitized_chunks_for_embedding:
        print("No chunks available for embedding after potential sanitization. Exiting.")
        return

    # Create ChromaDB vector store
    print(f"Creating vector store at {CHROMA_DB_PATH} with {len(sanitized_chunks_for_embedding)} chunks...")
    vector_store = Chroma.from_documents(
        documents=sanitized_chunks_for_embedding,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH
    )
    vector_store.persist()
    print(f"Data ingestion complete. Vector store persisted at {CHROMA_DB_PATH}")

if __name__ == "__main__":
    # This allows running the script directly
    # For the script to run standalone and initialize the LLM, we might need to
    # explicitly load it here if not handled by a global app context.
    # For now, we rely on config.LOCAL_LLM_INSTANCE potentially being set
    # or the script handling its own LLM init for sanitization.

    # Example: Initialize necessary components from config if running standalone
    import config # To load .env and other settings

    # Decide whether to sanitize during this standalone ingestion run
    # Set to False if you want to test ingestion without the NER LLM overhead for speed,
    # or if the NER LLM setup is complex for a standalone script.
    # The main app (app.py) will handle proper global LLM initialization.
    ingest_data(sanitize_during_ingestion=True) 