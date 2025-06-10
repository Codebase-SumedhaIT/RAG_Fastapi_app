import streamlit as st
import os
from faiss_query_engine import FaissQueryEngine

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("RAG Based Chatbot")
st.markdown("---")

# Set paths
index_path = 'Embeddor/faiss_index/PNR/PNR_embeddings.index'
metadata_path = 'Embeddor/faiss_index/PNR/PNR_metadata.pkl'

# Initialize engine (do it only once)
@st.cache_resource
def load_query_engine():
    return FaissQueryEngine(index_path, metadata_path)

try:
    engine = load_query_engine()
    
    # Query input
    query = st.text_input("Enter your question:", placeholder="Type your question here...")
    k = st.slider("Number of results:", min_value=1, max_value=10, value=5)
    
    if st.button("Search") or query:
        if query:
            with st.spinner('Searching...'):
                query_vector = engine.create_query_embedding(query)
                results = engine.search(query_vector, k=k)
            
            # Display results
            for result in results:
                with st.container():
                    st.markdown(f"""
                    **Response:** {result['sentence']}  
                    **Source:** {result['file']}  
                    **Similarity Score:** {1 - result['distance']:.4f}
                    ---
                    """)
        else:
            st.warning("Please enter a question.")
            
except Exception as e:
    st.error(f"Error loading index or metadata: {str(e)}")
    st.error(f"Make sure the following files exist:\n{index_path}\n{metadata_path}")