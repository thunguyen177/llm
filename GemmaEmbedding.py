# pip install ollama faiss-cpu numpy
# ollama run embeddinggemma
# ollama pull gemma:2b

import ollama
import faiss
import numpy as np

# 1. In-memory "Knowledge Base" 📚
# In a real-world scenario, this would be a collection of documents.
documents = [
    "The capital of France is Paris.",
    "The Eiffel Tower is a famous landmark in Paris.",
    "Mars is known as the Red Planet.",
    "The solar system has eight planets.",
    "The main component of Earth's atmosphere is nitrogen.",
    "Water is composed of hydrogen and oxygen atoms.",
]

# 2. Use EmbeddingGemma to create embeddings for our documents
print(  "Embedding documents...")
embeddings = []
for doc in documents:
    response = ollama.embeddings(model="embeddinggemma", prompt=doc)
    embeddings.append(response["embedding"])

# Convert embeddings to a NumPy array
embeddings_np = np.array(embeddings).astype('float32')

# 3. Create and populate a FAISS vector store 🧠
# FAISS (Facebook AI Similarity Search) is a library for efficient similarity search.
print("Creating FAISS index...")
dimension = embeddings_np.shape[1]  # Get the dimension of the embeddings
index = faiss.IndexFlatL2(dimension) # Using a simple L2 distance index
index.add(embeddings_np)
print(f"FAISS index created with {index.ntotal} vectors.")

# 4. The RAG Query Function
def query_rag(query: str, k: int = 2): 
    """
    Queries the RAG system.
    1. Embeds the query.
    2. Searches the vector store for the top-k most similar documents.
    3. Creates a prompt with the retrieved context.
    4. Calls the generative model to get an answer.
    """
    # Step 1: Embed the query
    query_embedding_response = ollama.embeddings(model="embeddinggemma", prompt=query)
    query_embedding = np.array([query_embedding_response["embedding"]]).astype('float32')

    # Step 2: Search the vector store
    print(f"\nSearching for the top {k} most relevant documents...")
    distances, indices = index.search(query_embedding, k)
    
    retrieved_chunks = [documents[i] for i in indices[0]]
    print("Retrieved context:", retrieved_chunks)

    # Step 3: Create a prompt with the context
    prompt_template = f"""
    Based ONLY on the following context, answer the user's question.
    If the context does not contain the answer, state that you don't have enough information.
    Context:
    - {"\n- ".join(retrieved_chunks)}
    Question: {query}
    """

    # Step 4: Call the generative model
    print("Generating answer...")
    response = ollama.chat(
        model='gemma:2b',
        messages=[{'role': 'user', 'content': prompt_template}]
    )
    return response['message']['content']

# 5. Run the RAG pipeline with a sample query
user_query = "What is the capital of France and what is a famous landmark there?"
final_answer = query_rag(user_query)

print("\n--- Final Answer ---")
print(final_answer)

user_query_2 = "What is Mars known as?"
final_answer_2 = query_rag(user_query_2)

print("\n--- Final Answer ---")
print(final_answer_2)