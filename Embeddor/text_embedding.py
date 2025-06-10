from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import json
import pickle

def load_text_file(file_path):
    """Load text from file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def create_embeddings(text, model):
    """Create embeddings for text using the specified model."""
    sentences = text.split('\n')
    sentences = [s.strip() for s in sentences if s.strip()]
    embeddings = model.encode(sentences)
    return embeddings, sentences

def create_faiss_index(embeddings, index_path):
    """Create and save FAISS index."""
    if len(embeddings) == 0:
        raise ValueError("No embeddings to index!")
        
    # Convert to numpy array if not already
    embeddings_array = np.array(embeddings).astype('float32')
    vector_dimension = embeddings_array.shape[1]
    
    print(f"Creating FAISS index with {len(embeddings)} vectors of dimension {vector_dimension}")
    
    index = faiss.IndexFlatL2(vector_dimension)
    index.add(embeddings_array)
    faiss.write_index(index, index_path)
    return index

def process_text_files(input_dir, output_dir, model_name='all-MiniLM-L6-v2'):
    """Process text files and create FAISS index."""
    model = SentenceTransformer(model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    all_embeddings = []
    all_sentences = []
    file_metadata = {}
    
    # Check if input directory exists and has text files
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    text_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    if not text_files:
        raise ValueError(f"No .txt files found in {input_dir}")
    
    # Process each text file
    for filename in text_files:
        print(f"Processing: {filename}")
        input_path = os.path.join(input_dir, filename)
        text = load_text_file(input_path)
        
        if not text.strip():
            print(f"Warning: Empty file {filename}, skipping...")
            continue
            
        # Create embeddings
        embeddings, sentences = create_embeddings(text, model)
        print(f"Created {len(embeddings)} embeddings for {filename}")
        
        # Store metadata
        start_idx = len(all_sentences)
        all_embeddings.extend(embeddings)
        all_sentences.extend(sentences)
        
        file_metadata[filename] = {
            'start_idx': start_idx,
            'end_idx': start_idx + len(sentences)
        }
    
    if not all_embeddings:
        raise ValueError("No embeddings were created from any of the files!")
    
    # Convert to numpy array
    all_embeddings = np.array(all_embeddings).astype('float32')
    print(f"\nTotal embeddings created: {len(all_embeddings)}")
    
    # Create and save FAISS index
    index_path = os.path.join(output_dir, 'embeddings.index')
    create_faiss_index(all_embeddings, index_path)
    
    # Save metadata and sentences
    metadata_path = os.path.join(output_dir, 'metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump({'sentences': all_sentences, 'files': file_metadata}, f)
    
    print(f"\nFAISS index saved to: {index_path}")
    print(f"Metadata saved to: {metadata_path}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_directory = os.path.join(current_dir, "..", "Extracted_Text","Synthesis")
    output_directory = os.path.join(current_dir, "faiss_index")
    
    process_text_files(input_directory, output_directory)