# RLPT Framework Prototype

This project is a prototype implementation of the RLPT (RAG Local Placeholder Tagging) framework. It demonstrates how to protect sensitive data in Retrieval Augmented Generation (RAG) workflows by identifying PII using a local LLM, sanitizing inputs by replacing PII with placeholders, and recovering the original PII in the final output.

## Core Features

* **PII Sanitization:** Uses a local LLM (Ollama with DebertaV3_For_NER) for Named Entity Recognition (NER) to detect PII.
* **Placeholder Management:** Stores PII-to-placeholder mappings in a `placeholder_mapping.json` file.
* **Response Recovery:** Restores original PII from placeholders in the RAG system's output.
* **RAG Integration:** Integrates with a Langchain-based RAG system using OpenAI for embeddings and generation, and ChromaDB as a vector store.
* **Flask API:** Provides endpoints for querying the RAG system with RLPT protection and for embedding new (sanitized) text.

## Project Structure

```
rlpt_prototype/
├── app.py                   # Main Flask application
├── config.py                # Configuration loader
├── .env                     # Environment variables (OpenAI API key, etc.)
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── data/                    # Sample data for vector DB
│   └── sample_documents.txt
├── vector_store/            # ChromaDB persistence directory
├── placeholder_mapping.json # Stores PII placeholders and original values
├── ingest_data.py           # Script to populate ChromaDB
├── models/
│   └── input/               # Input data for the model
│   └── util.py              # Utility functions for the model
│   └── model.py             # Model
└── rlpt/                    # Core RLPT logic
    ├── __init__.py
    ├── prompts.py           # Prompts for local LLM (NER)
    ├── placeholder_manager.py # Manages placeholder_mapping.json
    ├── sanitization.py      # PII detection and sanitization
    └── response_recovery.py # PII recovery
```

## Setup Instructions

1. **Prerequisites:**
   * Python 3.8+
   * Ollama installed and running.
      * Create a model from our model directory
      * Run the model with Ollama
   * An OpenAI API key.

2. **Create and Activate Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**
   * Create a `.env` file in the project root with your OpenAI API key and other settings:
     ```env
     OPENAI_API_KEY="your_openai_api_key_here"
     OLLAMA_BASE_URL="http://localhost:11434"
     LOCAL_NER_MODEL_NAME="DebertaV3_For_NER"
     RAG_EMBEDDING_MODEL_NAME="text-embedding-3-small"
     RAG_GENERATIVE_MODEL_NAME="gpt-4.1-mini-2025-04-14"
     CHROMA_DB_PATH="./vector_store"
     PLACEHOLDER_MAPPING_FILE="./placeholder_mapping.json"
     SAMPLE_DOCUMENTS_PATH="./data/sample_documents.txt"
     ```

5. **Initialize Placeholder Mapping File:**
   Ensure `placeholder_mapping.json` exists in the root directory with the initial structure:
   ```json
   {
       "PERSON_NAME": {},
       "EMAIL": {},
       "PHONE_NUMBER": {},
       "ADDRESS": {},
       "URL_PERSONAL": {},
       "ID_NUM": {},
       "USERNAME": {},
   }
   ```

6. **Ingest Sample Data into Vector Database:**
   ```bash
   python ingest_data.py
   ```

7. **Run the Flask Application:**
   ```bash
   python app.py
   ```
   The application will start on `http://localhost:5001`.

## API Endpoints

The Flask application exposes the following endpoints:

* `GET /health`
  * Checks the status of initialized components.
* `POST /sanitize`
  * Sanitizes input text.
  * Request body: `{"text": "Your text with PII..."}`
  * Response: Sanitized text and PII mappings for that text.
* `POST /recover`
  * Recovers PII from text containing placeholders.
  * Request body: `{"text": "Text with <PLACEHOLDER_1>..."}`
  * Response: Text with PII restored.
* `POST /query`
  * Main endpoint to query the RAG system with RLPT protection.
  * Request body: `{"query": "Your question with potential PII..."}`
  * Response: Original query, sanitized query, sanitized RAG answer, and final recovered answer.
* `POST /embed_text`
  * Adds new text to the vector database after sanitization.
  * Request body: `{"text": "New document content with PII...", "metadata": {"source": "optional_source_info"}}`
  * Response: Confirmation message.

## Example Usage

1. **Query the System:**
   ```bash
   curl --location 'http://localhost:5001/query' \
   --header 'Content-Type: application/json' \
   --data '{
      "query": "What is Alice Wonderland'\''s email and where does she live?"
   }'
   ```
   * Response:
   ```json
   {
      "final_answer_to_user": "Alice Wonderland's email is alice.wonder@example.com and she lives at 123 Fantasy Lane, Dreamville.",
      "original_query": "What is Alice Wonderland's email and where does she live?",
      "sanitized_query_sent_to_rag": "What is <PERSON_NAME_3>'s email and where does she live?",
      "sanitized_rag_answer": "<PERSON_NAME_3>'s email is <EMAIL_3> and she lives at <ADDRESS_5>, <ADDRESS_6>.",
      "source_document_count": 3
   }
   ```

2. **Embed New Text:**
   ```bash
   curl --location 'http://localhost:5001/embed_text' \
   --header 'Content-Type: application/json' \
   --data-raw '{
      "text": "My name is Alice Wonderland and I live at 123 Fantasy Lane, Dreamville. My email is alice.wonder@example.com. You can call me at +1-555-0100.",
      "metadata": {"source": "new_employee_memo"}
   }'
   ```
   * Response:
   ```json
   {
    "message": "Text sanitized and added to vector database successfully.",
    "original_text_preview": "My name is Alice Wonderland and I live at 123 Fantasy Lane, Dreamville. My email is alice.wonder@exa...",
    "pii_detected_and_mapped": true,
    "sanitized_text_embedded_preview": "My name is <PERSON_NAME_3> and I live at <ADDRESS_5>, <ADDRESS_6>. My email is <EMAIL_3>. You can ca..."
   }  
   ```

## Notes on RLPT Logic

* **Sanitization during Ingestion:** If documents are sanitized before being stored in the vector DB, the placeholders become part of the stored documents. This means user queries must also be sanitized using the same placeholder scheme for effective retrieval. The PlaceholderManager ensures consistent mapping.
* **Sanitization at Query Time:** The user's query is sanitized. The RAG system then operates on this sanitized query and potentially sanitized documents.
* **Response Recovery:** The RAG system's response (which might contain placeholders from the query or the documents) is then processed by the ResponseRecoverer to restore original PII before sending it to the user.
* **Local LLM for NER:** The quality of PII detection heavily depends on the chosen local LLM and the prompt used (rlpt/prompts.py). The current prompt aims for JSON output. Fine-tuning or more sophisticated prompting might be needed for higher accuracy.
* **Placeholder Uniqueness:** Placeholders are generated like `<PII_TYPE_INDEX>` (e.g., `<PERSON_NAME_1>`). The PlaceholderManager ensures these are unique per PII type and map consistently to the original values.

## Future Improvements

* More robust error handling
* Batch processing for sanitization and embedding
* Support for more PII types and context-aware PII detection
* Option to use different local LLMs for NER
* More sophisticated RAG prompting to handle placeholders explicitly
* UI for easier interaction 