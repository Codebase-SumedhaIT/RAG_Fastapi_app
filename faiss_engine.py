import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

class FaissQueryEngine:
    def __init__(self, index_path, metadata_path):
        """Initialize FAISS query engine"""
        self.index = faiss.read_index(index_path)
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.texts = metadata['texts']
        self.dimension = metadata['dimension']
        self.model_name = metadata.get('model', 'unknown')
        
        # Initialize the sentence transformer model
        self.embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
    
    def create_query_embedding(self, query):
        """Create an embedding for the query"""
        embedding = self.embedding_model.encode([query], 
                                             convert_to_tensor=True, 
                                             normalize_embeddings=True)
        return embedding.cpu().numpy().astype('float32')
        
    def search(self, query_vector, k=5):
        """Search for similar vectors"""
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.texts):
                result = {
                    'sentence': self.texts[idx]['text'],
                    'file': self.texts[idx]['file'],
                    'distance': float(dist)
                }
                results.append(result)
        
        return results