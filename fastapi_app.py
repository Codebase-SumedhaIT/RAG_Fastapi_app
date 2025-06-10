from fastapi import FastAPI, UploadFile, File, Form, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
from pydantic import BaseModel
from typing import List, Optional
import os
import shutil
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import re
import json
import streamlit as st
import logging
import requests
from faiss_engine import FaissQueryEngine
from utils.pdf_to_text import extract_text_from_pdf
from utils.embed_texts import create_embeddings_from_texts
import pytesseract

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configure Tesseract command path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str
    domain: str

class MCQRequest(BaseModel):
    domain: str
    num_questions: int
    difficulty: str
    model_name: str
    topic: Optional[str] = ""

class FileUploadRequest(BaseModel):
    domain: str
    subject: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]

# FastAPI app initialization
app = FastAPI(title="RAG Based Learning Assistant")

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React development server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configurations
MODEL_PATHS = {
    "Mistral-7B": r"c:\Users\kisha\.lmstudio\models\lmstudio-community\Mistral-7B-Instruct-v0.3-GGUF\Mistral-7B-Instruct-v0.3-IQ3_M.gguf",
    "Llama-2": r"c:\Users\kisha\.lmstudio\models\hugging-quants\Llama-3.2-1B-Instruct-Q8_0-GGUF\llama-3.2-1b-instruct-q8_0.gguf",
    "DeepSeek": r"C:\Users\kisha\.lmstudio\models\lmstudio-community\DeepSeek-R1-Distill-Qwen-7B-GGUF\DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf"
}

# Model cache
model_cache = {}
embedding_model = None

def get_llm(model_name: str):
    if model_name not in model_cache:
        model_cache[model_name] = Llama(
            model_path=MODEL_PATHS[model_name],
            n_ctx=8192,
            n_batch=512,
            n_threads=8,
            n_gpu_layers=-1,
            verbose=False
        )
    return model_cache[model_name]

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
    return embedding_model

# Update the root endpoint with health check
@app.get("/")
async def root():
    try:
        # Test FAISS directory
        faiss_dir = os.path.join("Embeddor", "faiss_index")
        faiss_exists = os.path.exists(faiss_dir)
        
        return {
            "status": "healthy",
            "message": "RAG Based Learning Assistant API is running",
            "faiss_index_available": faiss_exists
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/api/models")
async def get_available_models():
    logger.debug("Fetching available models")
    models = list(MODEL_PATHS.keys())
    logger.debug(f"Found models: {models}")
    return {"models": models}

@app.get("/api/domains")
async def get_available_domains():
    logger.debug("Fetching available domains")
    try:
        domains = [d for d in os.listdir(os.path.join("Embeddor", "faiss_index")) 
                  if os.path.isdir(os.path.join("Embeddor", "faiss_index", d))]
        logger.debug(f"Found domains: {domains}")
        return {"domains": domains}
    except Exception as e:
        logger.error(f"Error fetching domains: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_files(
    files: List[UploadFile],
    domain: str = Form(...),
    subject: str = Form(...)
):
    try:
        # 1. Create domain folder
        base_dir = os.path.join("Embeddor", "faiss_index", domain)
        os.makedirs(base_dir, exist_ok=True)
        text_files = []
        for file in files:
            # Save PDF to domain folder
            pdf_path = os.path.join(base_dir, file.filename)
            with open(pdf_path, "wb") as f:
                f.write(await file.read())
            # Extract text to .txt in domain folder
            text_path = os.path.splitext(pdf_path)[0] + ".txt"
            extract_text_from_pdf(pdf_path, text_path)
            text_files.append(text_path)
        # 2. Create embeddings and save in domain folder with domain name
        create_embeddings_from_texts(
            text_files,
            base_dir,
            domain  # This will name the files {domain}_embeddings.index and {domain}_metadata.pkl
        )
        return {"message": f"Files uploaded, embeddings created for domain '{domain}'."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query", response_model=QueryResponse)
async def query_response(request: QueryRequest):
    try:
        logger.debug(f"Received query: {request.query} for domain: {request.domain}")
        llm = get_llm("DeepSeek")
        logger.debug("Loaded DeepSeek model")
        engine = FaissQueryEngine(
            f'Embeddor/faiss_index/{request.domain}/{request.domain}_embeddings.index',
            f'Embeddor/faiss_index/{request.domain}/{request.domain}_metadata.pkl'
        )
        logger.debug("Loaded FAISS engine")
        query_vector = engine.create_query_embedding(request.query)
        logger.debug("Created query embedding")
        results = engine.search(query_vector, k=3)
        logger.debug(f"Search results: {results}")

        # Process contexts
        contexts = [f"Context {i+1}:\n{r['sentence'][:2000]}" for i, r in enumerate(results)]
        combined_context = "\n\n".join(contexts)

        # Generate prompt and response
        prompt = f"""<s>[INST] You are a knowledgeable AI assistant. Using the following contexts, provide a clear and accurate answer to the question.

        Contexts:
        {combined_context}

        Question: {request.query} [/INST]"""

        response = llm(
            prompt,
            max_tokens=2048,
            temperature=0.7,
            top_p=0.95,
            repeat_penalty=1.2,
            top_k=50,
            echo=False,
            stop=["</s>", "[INST]"]
        )

        # Format sources
        sources = [{"file": r['file'], "confidence": 1 - r['distance']} for r in results]

        cleaned_answer = clean_model_output(response['choices'][0]['text'])
        return QueryResponse(
            answer=cleaned_answer,
            sources=sources
        )

    except Exception as e:
        logger.error(f"Query endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def similarity_search(query: str, metadata_path, faiss_index_path, top_k: int = 5) -> list:
    import faiss
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    faiss_index = faiss.read_index(faiss_index_path)
    model = SentenceTransformer('BAAI/bge-large-en-v1.5')
    query_vec = model.encode([query], normalize_embeddings=True)
    D, I = faiss_index.search(np.array(query_vec).astype('float32'), top_k)
    results = []
    for idx in I[0]:
        results.append(metadata['texts'][idx]['text'])
    return results

def get_all_content(metadata_path):
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    all_text = []
    for item in metadata['texts']:
        all_text.append(item['text'])
    return " ".join(all_text)

def clean_json_response(text: str) -> str:
    match = re.search(r"\[\s*{.*?}\s*\]", text, re.DOTALL)
    if match:
        text = match.group(0)
    else:
        text = text.strip()
    text = re.sub(r'}\s*{', '},{', text)
    text = re.sub(r',\s*([\]}])', r'\1', text)
    text = text.replace("'", '"')
    text = re.sub(r'(\w+)\s*:', r'"\1":', text)
    return text

def safe_json_loads(cleaned_json):
    try:
        return json.loads(cleaned_json)
    except Exception:
        try:
            import json_repair
            repaired = json_repair.repair_json(cleaned_json)
            return json.loads(repaired)
        except Exception:
            return []

def validate_mcqs(mcqs, num_questions):
    validated = []
    for item in mcqs:
        if all(k in item for k in ["question", "options", "correct_answer", "explanation"]):
            if isinstance(item["options"], dict) and len(item["options"]) == 4 and item["correct_answer"] in item["options"]:
                validated.append(item)
        if len(validated) >= num_questions:
            break
    return validated[:num_questions]

def generate_mcq_prompt(num_questions: int, topic: str, difficulty: str, content: str) -> str:
    return f"""<s>[INST] 
Generate exactly {num_questions} MCQs as a JSON array ONLY, with no explanation or commentary before or after.
Each MCQ object must have:
- "question": string
- "options": object with keys "A", "B", "C", "D"
- "correct_answer": one of "A", "B", "C", "D"
- "explanation": string

Format:
[
  {{
    "question": "...",
    "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
    "correct_answer": "A",
    "explanation": "..."
  }},
  ...
]

Use ONLY this content for reference:
{content[:2000]}

Do NOT invent facts.
Do NOT output anything except the JSON array.
[/INST]"""

@app.post("/api/mcq")
async def generate_mcqs(request: MCQRequest):
    try:
        metadata_path = f'Embeddor/faiss_index/{request.domain}/{request.domain}_metadata.pkl'
        faiss_index_path = f'Embeddor/faiss_index/{request.domain}/{request.domain}_embeddings.index'
        # Use similarity search if topic is provided
        if hasattr(request, "topic") and request.topic and request.topic.strip():
            top_chunks = similarity_search(request.topic, metadata_path, faiss_index_path, top_k=5)
            content = "\n".join(top_chunks)
        else:
            content = get_all_content(metadata_path)
        topic_str = f" on {request.topic}" if hasattr(request, "topic") and request.topic else ""
        prompt = generate_mcq_prompt(request.num_questions, topic_str, request.difficulty, content)
        llm = get_llm(request.model_name)
        response = llm(
            prompt,
            max_tokens=2048,
            temperature=0.6,
            top_p=0.7,
            repeat_penalty=1.1,
            top_k=8,
            stop=["</s>", "[INST]"]
        )
        generated_text = response['choices'][0]['text']
        cleaned_json = clean_json_response(generated_text)
        mcqs = safe_json_loads(cleaned_json)
        validated_mcqs = validate_mcqs(mcqs, request.num_questions)
        return {"mcqs": validated_mcqs}
    except Exception as e:
        logger.error(f"Error generating MCQs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def clean_model_output(text: str) -> str:
    # Remove </think> and similar tags
    text = re.sub(r'</think>', '', text, flags=re.IGNORECASE)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text

# Create static directory for favicon if it doesn't exist
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)

# Add favicon endpoint
@app.get("/favicon.ico")
async def get_favicon():
    # Return a default favicon or your custom one
    return FileResponse("static/favicon.ico", media_type="image/x-icon")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
