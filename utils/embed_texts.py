import os
import numpy as np
import pickle
import faiss
from sentence_transformers import SentenceTransformer

def create_embeddings_from_texts(text_files, output_dir, domain):
    model = SentenceTransformer('BAAI/bge-large-en-v1.5')
    os.makedirs(output_dir, exist_ok=True)
    texts = []
    embeddings = []
    for file_path in text_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        embedding = model.encode(content, normalize_embeddings=True)
        texts.append({
            'text': content,
            'file': os.path.basename(file_path),
            'path': file_path
        })
        embeddings.append(embedding)
    embeddings_array = np.array(embeddings).astype('float32')
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)
    # Save index and metadata
    index_path = os.path.join(output_dir, f'{domain}_embeddings.index')
    faiss.write_index(index, index_path)
    metadata = {
        'texts': texts,
        'dimension': dimension,
        'model': 'BAAI/bge-large-en-v1.5'
    }
    metadata_path = os.path.join(output_dir, f'{domain}_metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    return index_path, metadata_path