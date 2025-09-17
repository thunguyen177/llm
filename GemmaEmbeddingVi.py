import ollama
import faiss
import numpy as np

# 1. "Cơ sở dữ liệu" trong bộ nhớ (để ví dụ đơn giản)
documents = [
    "The capital of France is Paris.",
    "The Eiffel Tower is a famous landmark in Paris.",
    "Mars is known as the Red Planet.",
    "The solar system has eight planets.",
    "The main component of Earth's atmosphere is nitrogen.",
    "Water is composed of hydrogen and oxygen atoms.",
]

# 2. Tạo embedding cho các tài liệu bằng EmbeddingGemma
print("Embedding documents...")
embeddings = []
for doc in documents:
    response = ollama.embeddings(model="embeddinggemma", prompt=doc)
    embeddings.append(response["embedding"])

# Chuyển embeddings thành mảng NumPy
embeddings_np = np.array(embeddings).astype('float32')

# 3. Tạo và nạp dữ liệu vào FAISS vector store
print("Creating FAISS index...")
dimension = embeddings_np.shape[1]          # chiều của embedding
index = faiss.IndexFlatL2(dimension)        # chỉ mục L2 đơn giản
index.add(embeddings_np)
print(f"FAISS index created with {index.ntotal} vectors.")

# 4. Hàm truy vấn RAG
def query_rag(query: str, k: int = 2):
    """
    Truy vấn hệ thống RAG.
    1. Embedding câu hỏi.
    2. Tìm kiếm vector store để lấy top‑k tài liệu liên quan.
    3. Tạo prompt với ngữ cảnh thu thập.
    4. Gọi mô hình sinh văn bản để trả lời.
    """
    # Bước 1: Embedding câu hỏi
    query_embedding_response = ollama.embeddings(model="embeddinggemma", prompt=query)
    query_embedding = np.array([query_embedding_response["embedding"]]).astype('float32')

    # Bước 2: Tìm kiếm
    print(f"\nSearching for the top {k} most relevant documents...")
    distances, indices = index.search(query_embedding, k)

    retrieved_chunks = [documents[i] for i in indices[0]]
    print("Retrieved context:", retrieved_chunks)

    # Bước 3: Tạo prompt
    prompt_template = f"""
    Based ONLY on the following context, answer the user's question.
    If the context does not contain the answer, state that you don't have enough information.
    Context:
    - {"\n- ".join(retrieved_chunks)}
    Question: {query}
    """

    # Bước 4: Gọi mô hình sinh
    print("Generating answer...")
    response = ollama.chat(
        model='gemma:2b',
        messages=[{'role': 'user', 'content': prompt_template}]
    )
    return response['message']['content']

# 5. Kiểm thử pipeline với các câu hỏi mẫu
user_query = "What is the capital of France and what is a famous landmark there?"
final_answer = query_rag(user_query)

print("\n--- Final Answer ---")
print(final_answer)

user_query_2 = "What is Mars known as?"
final_answer_2 = query_rag(user_query_2)

print("\n--- Final Answer (Mars) ---")
print(final_answer_2)
