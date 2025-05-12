from flask import Flask, request, jsonify
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.llms import Ollama # For local NER LLM
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

import config # Import all configurations
from rlpt import PlaceholderManager, Sanitizer, ResponseRecoverer

app = Flask(__name__)

# --- Global Initializations ---
# Initialize components that are expensive to load, once at startup.

# 1. Local LLM for NER (Ollama)
try:
    config.LOCAL_LLM_INSTANCE = Ollama(
        base_url=config.OLLAMA_BASE_URL,
        model=config.LOCAL_NER_MODEL_NAME,
        # You might need to add temperature=0 or other params for consistent JSON output
        temperature=0
    )
    print(f"Successfully initialized Local NER LLM: {config.LOCAL_NER_MODEL_NAME} from {config.OLLAMA_BASE_URL}")
except Exception as e:
    print(f"Error initializing Local NER LLM ({config.LOCAL_NER_MODEL_NAME}): {e}")
    print("Please ensure Ollama is running and the model is pulled (e.g., 'ollama pull llama3:instruct').")
    config.LOCAL_LLM_INSTANCE = None # Set to None if failed

# 2. RLPT Components
placeholder_manager = PlaceholderManager(mapping_file_path=config.PLACEHOLDER_MAPPING_FILE)

if config.LOCAL_LLM_INSTANCE:
    sanitizer = Sanitizer(local_llm=config.LOCAL_LLM_INSTANCE, placeholder_manager=placeholder_manager)
else:
    sanitizer = None # Sanitization will not be available
    print("WARNING: Sanitizer not initialized because Local NER LLM failed to load.")

response_recoverer = ResponseRecoverer(placeholder_manager=placeholder_manager)

# 3. RAG Components
# OpenAI Embeddings for RAG
try:
    rag_embeddings = OpenAIEmbeddings(
        model=config.RAG_EMBEDDING_MODEL_NAME,
        openai_api_key=config.OPENAI_API_KEY
    )
    print(f"Successfully initialized RAG Embedding Model: {config.RAG_EMBEDDING_MODEL_NAME}")
except Exception as e:
    print(f"Error initializing RAG Embedding Model: {e}")
    rag_embeddings = None

# ChromaDB Vector Store
try:
    if rag_embeddings: # Only try to load if embeddings are available
        vector_store = Chroma(
            persist_directory=config.CHROMA_DB_PATH,
            embedding_function=rag_embeddings
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 chunks
        print(f"Successfully loaded Vector Store from: {config.CHROMA_DB_PATH}")
    else:
        vector_store = None
        retriever = None
        print("Vector Store not loaded due to embedding model initialization failure.")
except Exception as e:
    print(f"Error loading Vector Store from {config.CHROMA_DB_PATH}: {e}")
    print("Ensure you have run 'python scripts/ingest_data.py' to create the vector store.")
    vector_store = None
    retriever = None

# OpenAI Generative LLM for RAG
try:
    rag_llm = ChatOpenAI(
        model_name=config.RAG_GENERATIVE_MODEL_NAME,
        openai_api_key=config.OPENAI_API_KEY,
        temperature=0.7 # Adjust as needed
    )
    print(f"Successfully initialized RAG Generative LLM: {config.RAG_GENERATIVE_MODEL_NAME}")
except Exception as e:
    print(f"Error initializing RAG Generative LLM: {e}")
    rag_llm = None

# RAG Chain (QA Chain)
rag_chain = None
if retriever and rag_llm and sanitizer is not None:
    QA_TEMPLATE = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    IMPORTANT: If the context or the question contains bracketed placeholder tags (e.g., [PERSON_NAME_1]), you MUST preserve these tags exactly as they appear in your answer. Do not attempt to modify or explain them.

    Context: {context}
    Question: {question}
    Helpful Answer:"""
    qa_prompt = PromptTemplate(
        template=QA_TEMPLATE, input_variables=["context", "question"]
    )
    rag_chain = RetrievalQA.from_chain_type(
        llm=rag_llm,
        chain_type="stuff", # Uses all retrieved text in the prompt
        retriever=retriever,
        return_source_documents=True, # Useful for debugging/transparency
        chain_type_kwargs={"prompt": qa_prompt}
    )
    print("RAG QA Chain initialized.")
   
else:
    print("RAG QA Chain not initialized due to missing components (retriever or RAG LLM).")

# --- API Endpoints ---
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    status = {
        "status": "ok",
        "local_ner_llm_initialized": config.LOCAL_LLM_INSTANCE is not None,
        "sanitizer_initialized": sanitizer is not None,
        "rag_embeddings_initialized": rag_embeddings is not None,
        "vector_store_loaded": vector_store is not None,
        "rag_llm_initialized": rag_llm is not None,
        "rag_chain_initialized": rag_chain is not None
    }
    return jsonify(status)

@app.route('/sanitize', methods=['POST'])
def sanitize_endpoint():
    """Endpoint to sanitize a given text."""
    if not sanitizer:
        return jsonify({"error": "Sanitizer not available. Check Local NER LLM initialization."}), 500

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' in request body"}), 400

    text_to_sanitize = data['text']
    sanitized_text, pii_mappings = sanitizer.sanitize_text(text_to_sanitize)

    return jsonify({
        "original_text": text_to_sanitize,
        "sanitized_text": sanitized_text,
        "pii_mappings_for_this_text": pii_mappings # PII found in this specific text
    })

@app.route('/recover', methods=['POST'])
def recover_endpoint():
    """Endpoint to recover PII in a text containing placeholders."""
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' in request body"}), 400

    text_to_recover = data['text']
    # Optionally, allow passing specific_mappings if the client has them from a /sanitize call
    specific_mappings = data.get('specific_mappings') 

    recovered_text = response_recoverer.recover_text(text_to_recover, specific_mappings=specific_mappings)

    return jsonify({
        "text_with_placeholders": text_to_recover,
        "recovered_text": recovered_text
    })

@app.route('/query', methods=['POST'])
def query_rag_system():
    """
    Handles a user query:
    1. Sanitizes the query using RLPT.
    2. Sends the sanitized query to the RAG system.
    3. Gets a sanitized response from the RAG system.
    4. Recovers PII in the RAG response using RLPT.
    5. Returns the final, PII-restored answer to the user.
    """
    if not rag_chain:
        return jsonify({"error": "RAG system not initialized. Check component statuses via /health."}), 500
    if not sanitizer:
        return jsonify({"error": "Sanitizer not available for query processing."}), 500

    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' in request body"}), 400

    original_query = data['query']
    print(f"\n--- New Query Received ---\nOriginal Query: {original_query}")

    # 1. Sanitize the user query
    sanitized_query, query_pii_map = sanitizer.sanitize_text(original_query)
    print(f"Sanitized Query: {sanitized_query}")
    if query_pii_map:
        print(f"PII map for query: {query_pii_map}")

    # 2. Send sanitized query to RAG system
    # The RAG chain itself might use a prompt that needs to be aware of placeholders.
    # For this prototype, we assume the RAG LLM can handle placeholders if instructed.
    # A more advanced setup might involve specific instructions in the RAG prompt about placeholders.
    try:
        rag_response = rag_chain.invoke({"query": sanitized_query})
        # rag_response is a dict, e.g., {'query': ..., 'result': ..., 'source_documents': ...}
        sanitized_answer = rag_response.get('result', "Error: Could not get result from RAG chain.")
        source_documents = rag_response.get('source_documents', [])

        print(f"Sanitized RAG Answer: {sanitized_answer}")
        if source_documents:
            print(f"Retrieved {len(source_documents)} source documents.")
            # for i, doc in enumerate(source_documents):
            #     print(f"  Source {i+1}: {doc.page_content[:100]}...") # Print snippet
    except Exception as e:
        print(f"Error during RAG chain invocation: {e}")
        return jsonify({"error": f"RAG system error: {str(e)}"}), 500

    # 3. Recover PII in the RAG response
    # We use all known mappings (global placeholder_manager) for recovery,
    # as the RAG response might contain placeholders from the documents, not just the query.
    final_answer = response_recoverer.recover_text(sanitized_answer) # No specific_mappings here, use global
    print(f"Final Recovered Answer: {final_answer}")

    return jsonify({
        "original_query": original_query,
        "sanitized_query_sent_to_rag": sanitized_query,
        "sanitized_rag_answer": sanitized_answer,
        "final_answer_to_user": final_answer,
        "source_document_count": len(source_documents)
        # "source_documents_content": [doc.page_content for doc in source_documents] # Optional
    })

@app.route('/embed_text', methods=['POST'])
def embed_new_text_endpoint():
    """
    Endpoint to demonstrate embedding new text into the vector database.
    1. Receives text.
    2. Sanitizes the text using RLPT (detects PII, replaces with existing or new placeholders).
    3. Embeds the sanitized text using the RAG embedding model.
    4. Adds the sanitized text and its embedding to ChromaDB.
    """
    if not vector_store or not rag_embeddings:
        return jsonify({"error": "Vector store or embedding model not initialized."}), 500
    if not sanitizer:
        return jsonify({"error": "Sanitizer not available for text embedding."}), 500

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' in request body"}), 400

    original_text = data['text']
    doc_metadata = data.get('metadata', {"source": "api_submission"}) # Optional metadata

    print(f"\n--- New Text for Embedding Received ---\nOriginal Text: {original_text[:200]}...")

    # 1. Sanitize the text
    sanitized_text_for_embedding, text_pii_map = sanitizer.sanitize_text(original_text)
    print(f"Sanitized Text for Embedding: {sanitized_text_for_embedding[:200]}...")
    if text_pii_map:
        print(f"PII map for this text: {text_pii_map}")
        # Note: placeholder_manager is updated globally by sanitizer.

    # 2. Create Langchain Document
    doc_to_embed = [Document(page_content=sanitized_text_for_embedding, metadata=doc_metadata)]

    # 3. Add to ChromaDB (Chroma.add_documents handles embedding)
    try:
        vector_store.add_documents(documents=doc_to_embed, embedding=rag_embeddings) # Pass embedding func
        vector_store.persist() # Persist changes
        print(f"Successfully added sanitized text to vector store. New document ID(s) created by Chroma.")
        # ChromaDB typically generates IDs if not provided.
        # If you need to return the ID, you'd need to manage IDs explicitly or retrieve the last added.
    except Exception as e:
        print(f"Error adding document to vector store: {e}")
        return jsonify({"error": f"Failed to add document to vector store: {str(e)}"}), 500

    return jsonify({
        "message": "Text sanitized and added to vector database successfully.",
        "original_text_preview": original_text[:100] + "...",
        "sanitized_text_embedded_preview": sanitized_text_for_embedding[:100] + "...",
        "pii_detected_and_mapped": bool(text_pii_map)
    })

if __name__ == '__main__':
    # Ensure Ollama is running and the model is pulled:
    # `ollama pull llama3:instruct` (or your chosen model)
    # Run the data ingestion script first if the vector store is empty:
    # `python scripts/ingest_data.py`
    app.run(debug=True, port=5001) # Use a different port if 5000 is common 