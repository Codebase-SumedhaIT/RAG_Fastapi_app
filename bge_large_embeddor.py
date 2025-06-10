import os
import numpy as np
import glob
import pickle
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def create_embeddings(folder_path, output_dir):
    """Create and save embeddings using BGE-large-en-v1.5"""
    # Initialize the model
    model = SentenceTransformer('BAAI/bge-large-en-v1.5')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all text files
    text_files = glob.glob(os.path.join(folder_path, "*.txt"))
    texts = []
    embeddings = []
    
    # Process each file
    for file_path in tqdm(text_files, desc="Creating embeddings"):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Create embedding
        embedding = model.encode(content, normalize_embeddings=True)
        
        # Store text and embedding
        texts.append({
            'text': content,
            'file': os.path.basename(file_path),
            'path': file_path
        })
        embeddings.append(embedding)
    
    # Convert embeddings to numpy array
    embeddings_array = np.array(embeddings).astype('float32')
    
    # Create FAISS index
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)
    
    # Save index
    index_path = os.path.join(output_dir, 'Synthesis_embeddings.index')
    faiss.write_index(index, index_path)
    
    # Save metadata
    metadata = {
        'texts': texts,
        'dimension': dimension,
        'model': 'BAAI/bge-large-en-v1.5'
    }
    metadata_path = os.path.join(output_dir, 'Synthesis_metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"Index saved to: {index_path}")
    print(f"Metadata saved to: {metadata_path}")

if __name__ == "__main__":
    # Update these paths to match your setup
    input_folder = "Extracted_Text/Synthesis"
    output_dir = "Embeddor/faiss_index/Synthesis"
    create_embeddings(input_folder, output_dir)