import os
import openai
import numpy as np
import glob
import pickle
import faiss
from tqdm import tqdm
import json

# Get API key from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

def load_text_files(domain_folder):
    """Load all text files from a domain folder."""
    text_files = glob.glob(os.path.join(domain_folder, "*.txt"))
    documents = []
    
    for file_path in text_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            documents.append({
                'content': content,
                'file': os.path.basename(file_path),
                'path': file_path
            })
    return documents

def create_embeddings(documents, batch_size=100):
    """Create embeddings using OpenAI's ada-002 model."""
    all_embeddings = []
    all_texts = []
    
    for doc in tqdm(documents, desc="Creating embeddings"):
        # Split content into chunks (max 8k tokens)
        chunks = [doc['content'][i:i+8000] for i in range(0, len(doc['content']), 8000)]
        
        for chunk in chunks:
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=chunk
            )
            all_embeddings.append(response['data'][0]['embedding'])
            all_texts.append({
                'text': chunk,
                'file': doc['file'],
                'path': doc['path']
            })
    
    return np.array(all_embeddings), all_texts

def save_faiss_index(embeddings, texts, domain_name):
    """Save FAISS index and metadata."""
    # Create output directory
    output_dir = os.path.join('faiss_index', domain_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create and save FAISS index
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    
    index_path = os.path.join(output_dir, f'{domain_name}_embeddings.index')
    faiss.write_index(index, index_path)
    
    # Save metadata
    metadata = {
        'texts': texts,
        'dimension': dimension,
        'model': 'text-embedding-ada-002'
    }
    metadata_path = os.path.join(output_dir, f'{domain_name}_metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    return index_path, metadata_path

def process_domain(domain_name):
    """Process all files in a domain folder."""
    print(f"\nProcessing domain: {domain_name}")
    
    # Set paths
    domain_folder = os.path.join("../Extracted_Text", domain_name)
    
    # Load documents
    documents = load_text_files(domain_folder)
    if not documents:
        print(f"No text files found in {domain_folder}")
        return
    
    # Create embeddings
    print("Creating embeddings using OpenAI ada-002...")
    embeddings, texts = create_embeddings(documents)
    
    # Save index and metadata
    index_path, metadata_path = save_faiss_index(embeddings, texts, domain_name)
    print(f"\nSaved FAISS index to: {index_path}")
    print(f"Saved metadata to: {metadata_path}")
    
    # Prepare training data for Llama fine-tuning
    training_data = []
    for text in texts:
        training_data.append({
            "input": text['text'],  # Using the entire text as input
            "file": text['file'],
            "metadata": {
                "path": text['path']
            }
        })
    
    # Save training data
    train_path = os.path.join('faiss_index', domain_name, f'{domain_name}_train.json')
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2)
    
    print(f"Saved training data to: {train_path}")

if __name__ == "__main__":
    # Process each domain folder
    base_dir = os.path.join("..", "Extracted_Text")
    domains = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    print(f"base dir: {base_dir}")
    for domain in domains:
        print(f"domain: {domain}")
        process_domain(domain)