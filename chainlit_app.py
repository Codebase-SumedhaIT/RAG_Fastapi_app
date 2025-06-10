import chainlit as cl
import asyncio
from concurrent.futures import ThreadPoolExecutor
from faiss_engine import FaissQueryEngine
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

# Configuration
INDEX_PATH = 'Embeddor/faiss_index/PNR/PNR_embeddings.index'
METADATA_PATH = 'Embeddor/faiss_index/PNR/PNR_metadata.pkl'
MODEL_PATH = r"C:\Users\kisha\.lmstudio\models\lmstudio-community\DeepSeek-R1-Distill-Qwen-7B-GGUF\DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf"

# Helper function to clean thinking patterns
def clean_thinking(text):
    if "<think>" in text:
        return text.split("</think>")[-1].strip()
    return text

@cl.cache
def load_llm():
    return Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,
        n_batch=128,
        n_threads=6,
        n_gpu_layers=18,
        use_mlock=True,
        use_mmap=True,
        verbose=False
    )

@cl.cache 
def load_embedding_model():
    return SentenceTransformer('BAAI/bge-large-en-v1.5')

@cl.cache
def load_engine():
    return FaissQueryEngine(INDEX_PATH, METADATA_PATH)

@cl.on_chat_start
async def start():
    cl.user_session.set("llm", load_llm())
    cl.user_session.set("embedder", load_embedding_model())
    cl.user_session.set("engine", load_engine())
    await cl.Message(content="Document analyzer ready. Ask anything!").send()

@cl.on_message
async def main(message: cl.Message):
    try:
        # Get components from session
        llm = cl.user_session.get("llm")
        embedder = cl.user_session.get("embedder")
        engine = cl.user_session.get("engine")
        
        # RAG retrieval
        query_embedding = embedder.encode(
            [message.content], 
            convert_to_tensor=True,
            normalize_embeddings=True
        ).cpu().numpy().astype('float32')
        
        results = engine.search(query_embedding, k=3)
        
        # Build context from retrieved docs
        context = "\n".join(
            f"Document {i+1}: {res['sentence'][:500]}..." 
            for i, res in enumerate(results)
        )
        
        # DeepSeek optimized prompt format
        prompt = f"""[INST] Answer the following question using only the context provided:

Context:
{context}

Question: {message.content}

Answer: [/INST]"""
        
        # Create message for streaming
        msg = cl.Message(content="")
        await msg.send()
        
        # Set up callback handler - THIS IS THE KEY FIX
        cb = cl.AsyncLangchainCallbackHandler(
            stream_final_answer=True,
            answer_prefix_tokens=["Answer"]
        )
        cb.answer_reached = True  # Force answer mode to prevent thinking output
        
        # Run the LLM with streaming
        def run_llm():
            return llm(
                prompt,
                max_tokens=512,
                temperature=0.5,
                stream=True,
                stop=["</s>", "[INST]", "<think>"]
            )
        
        # Execute in thread to avoid blocking
        response_gen = await asyncio.get_event_loop().run_in_executor(
            ThreadPoolExecutor(), run_llm
        )
        
        # Process tokens with thinking removal
        response = ""
        for chunk in response_gen:
            if 'choices' in chunk and chunk['choices']:
                token = chunk['choices'][0].get('text', '')
                if token:
                    # Skip thinking tags
                    if "<think>" in token or "</think>" in token:
                        continue
                    # Add token to response and stream
                    response += token
                    await msg.stream_token(token)
        
        # Update message with cleaned response
        await msg.update()
        
        # Show sources
        sources = "\n".join(
            f"📄 {res['file']} (Confidence: {1 - res['distance']:.0%})"
            for res in results
        )
        await cl.Message(content=f"**Reference Sources:**\n{sources}").send()
        
    except Exception as e:
        await cl.Message(content=f"⚠️ Error: {str(e)}").send()
