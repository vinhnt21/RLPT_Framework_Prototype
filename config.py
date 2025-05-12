import os
from dotenv import load_dotenv

load_dotenv() # Load variables from .env file

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Ollama Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LOCAL_NER_MODEL_NAME = os.getenv("LOCAL_NER_MODEL_NAME", "llama3:instruct")

# RAG Model Names
RAG_EMBEDDING_MODEL_NAME = os.getenv("RAG_EMBEDDING_MODEL_NAME", "text-embedding-3-small")
RAG_GENERATIVE_MODEL_NAME = os.getenv("RAG_GENERATIVE_MODEL_NAME", "gpt-4.1-mini-2025-04-14")

# Database and File Paths 
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./vector_store")
PLACEHOLDER_MAPPING_FILE = os.getenv("PLACEHOLDER_MAPPING_FILE", "./placeholder_mapping.json")
SAMPLE_DOCUMENTS_PATH = os.getenv("SAMPLE_DOCUMENTS_PATH", "./data/sample_documents.txt")



# Placeholder for global LLM instance (to be initialized in app.py or a dedicated module)
LOCAL_LLM_INSTANCE = None

# PII Entity Types (Consistent with placeholder_mapping.json structure)
PII_ENTITY_TYPES = ["PERSON_NAME", "EMAIL", "PHONE_NUMBER", "ADDRESS", "URL_PERSONAL", "ID_NUM", "USERNAME"]

# Ensure API key is available
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")

print("Configuration loaded.") 