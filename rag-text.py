import os
import json
import numpy as np
import faiss
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# === Config ===
embedding_model = "models/gemini-embedding-exp-03-07"
vec_path = "public/text/doc_vectors.npy"
index_path = "public/text/faiss.index"
meta_path = "public/text/doc_metadata.json"

# === Sample documents (used only on first run) ===
documents = [
    """Once upon a time in a quiet village nestled between the mountains, 
    there lived a curious young girl named Elara. One day, she wandered 
    into the forest and discovered a hidden cave. Inside, she found an 
    ancient dragon guarding a glowing crystal that granted eternal wisdom.""",

    """Long ago, in a distant kingdom ruled by a just king, a knight named Thorne 
    was tasked with recovering a stolen crown from a band of clever thieves. 
    He journeyed across rivers and deserts to find the hidden fortress where 
    the crown was kept.""",

    """On a snowy night, an old inventor built a clock that could turn back time. 
    But the more he used it, the more the timeline fractured. Eventually, 
    he had to choose between saving his past or preserving the future.""",

    """A spaceship carrying the last survivors of Earth encountered a strange 
    signal in deep space. It led them to a mysterious planet filled with 
    structures resembling ancient Earth civilizations."""
]

# === Step 1: Generate or Load Embeddings and FAISS Index ===
if os.path.exists(vec_path) and os.path.exists(index_path) and os.path.exists(meta_path):
    print("Loading vectors, index, and document metadata...")
    doc_vectors = np.load(vec_path)
    index = faiss.read_index(index_path)
    with open(meta_path, "r") as f:
        doc_texts = json.load(f)
else:
    print("Generating vectors and building FAISS index...")
    embedder = GoogleGenerativeAIEmbeddings(
        model=embedding_model, task_type="RETRIEVAL_DOCUMENT"
    )
    doc_vectors = np.array(embedder.embed_documents(documents)).astype("float32")

    # Save vectors and metadata
    np.save(vec_path, doc_vectors)
    with open(meta_path, "w") as f:
        json.dump(documents, f)

    # Create and save FAISS index
    index = faiss.IndexFlatL2(doc_vectors.shape[1])
    index.add(doc_vectors)
    faiss.write_index(index, index_path)
    doc_texts = documents

# === Step 2: Embed the Query ===
query = "What was the dragon protecting?"
query_embedder = GoogleGenerativeAIEmbeddings(
    model=embedding_model, task_type="RETRIEVAL_QUERY"
)
q_embed = np.array(query_embedder.embed_query(query)).astype("float32")

# === Step 3: Search FAISS for Top Matches ===
top_k = 2
D, I = index.search(np.array([q_embed]), top_k)
retrieved_docs = [doc_texts[i] for i in I[0]]

# === Step 4: Construct Prompt and Query Gemini ===
context = "\n\n".join(retrieved_docs)
final_prompt = f"""Answer the question based on the context below.

Context:
{context}

Question:
{query}

Answer:"""

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
response = llm.invoke(final_prompt)

# === Step 5: Print Response ===
print("Generated Answer:")
print(response.content)
