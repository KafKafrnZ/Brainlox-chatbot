import os
import logging
from functools import lru_cache
from typing import List
from dotenv import load_dotenv
from flask import Flask, request, send_from_directory
from flask_restful import Resource, Api
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory

# Load environment variables from .env
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set USER_AGENT explicitly
os.environ["USER_AGENT"] = "Brainlox-Chatbot/1.0"

# Flask app setup
app = Flask(__name__)
api = Api(app)

# Rate limiting
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"  # Change to "redis://localhost:6379" for production
)

# Configuration using environment variable for HF token
CONFIG = {
    "WEB_URLS": ["https://brainlox.com/courses/category/technical"],
    "HF_API_TOKEN": os.getenv("HF_API_TOKEN"),  # Fetch from .env
    "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
    "LLM_MODEL": "HuggingFaceH4/zephyr-7b-beta",
    "CHUNK_SIZE": 1000,
    "CHUNK_OVERLAP": 200,
    "DB_DIR": "./chroma_db",
    "MAX_QUERY_LENGTH": 500,
}

def load_web_data(urls: List[str] = CONFIG["WEB_URLS"]) -> List:
    try:
        loader = WebBaseLoader(urls)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG["CHUNK_SIZE"],
            chunk_overlap=CONFIG["CHUNK_OVERLAP"]
        )
        docs = text_splitter.split_documents(documents)
        logger.info(f"Loaded and split {len(docs)} documents from {urls}")
        return docs
    except Exception as e:
        logger.error(f"Failed to load web data: {e}")
        return []

def create_vector_store(docs: List) -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=CONFIG["EMBEDDING_MODEL"])
    if os.path.exists(CONFIG["DB_DIR"]):
        logger.info("Loading existing vector store")
        return Chroma(persist_directory=CONFIG["DB_DIR"], embedding_function=embeddings)
    logger.info("Creating new vector store")
    vector_store = Chroma.from_documents(docs, embeddings, persist_directory=CONFIG["DB_DIR"])
    return vector_store

def initialize_chatbot():
    # Check if token is loaded
    if not CONFIG["HF_API_TOKEN"]:
        raise ValueError("HF_API_TOKEN not found in .env file")
    
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = CONFIG["HF_API_TOKEN"]
    
    docs = load_web_data()
    if not docs:
        raise RuntimeError("No documents loaded, cannot initialize chatbot")
    
    vector_store = create_vector_store(docs)
    
    llm = HuggingFaceEndpoint(
        repo_id=CONFIG["LLM_MODEL"],
        huggingfacehub_api_token=CONFIG["HF_API_TOKEN"],
        temperature=0.7,
        max_length=500
    )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="result",
        return_messages=True
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        memory=memory,
        return_source_documents=True
    )
    return qa_chain

# Initialize chatbot globally at startup
try:
    QA_CHAIN = initialize_chatbot()
    logger.info("Chatbot initialized successfully")
except Exception as e:
    logger.critical(f"Failed to initialize chatbot at startup: {e}")
    raise

@lru_cache(maxsize=100)
def cached_query(query: str) -> dict:
    return QA_CHAIN.invoke({"query": query})

# Flask Routes
@app.route('/')
def home():
    return "Welcome to Brainlox Chatbot API. Visit /ui to chat."

@app.route('/ui', methods=['GET'])
def ui():
    # Serve chat.html from the same directory as chatbot.py
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'chat.html')

# Chatbot API endpoint
class ChatbotAPI(Resource):
    @limiter.limit("5 per minute")
    def post(self):
        data = request.get_json()
        query = data.get("query", "").strip()
        if not query or len(query) > CONFIG["MAX_QUERY_LENGTH"]:
            logger.warning(f"Invalid query received: {query}")
            return {"error": "Invalid or too long query"}, 400
        
        try:
            logger.info(f"Processing query: {query}")
            result = cached_query(query)
            return {
                "response": result["result"],
                "sources": [doc.page_content[:200] for doc in result["source_documents"]]
            }, 200
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {"error": "Internal server error"}, 500

# Register API resource
api.add_resource(ChatbotAPI, '/chat')

if __name__ == "__main__":
    try:
        logger.info("Starting Flask application")
        app.run(debug=False, host='0.0.0.0', port=5000)
    except Exception as e:
        logger.critical(f"Application failed to start: {e}")