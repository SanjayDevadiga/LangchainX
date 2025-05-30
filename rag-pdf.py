import os
import json
import numpy as np
import faiss
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

from dotenv import load_dotenv
load_dotenv()

# === Config ===
embedding_model = "models/gemini-embedding-exp-03-07"
pdf_path = "public/pdf/resume.pdf"
vec_path = "public/pdf/doc_vectors.npy"
index_path = "public/pdf/faiss.index"
meta_path = "public/pdf/doc_metadata.json"

# === Step 1: Read and Split PDF ===
def load_pdf_chunks(pdf_path):
    reader = PdfReader(pdf_path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)

    if len(text) == 0:
        print("Not able to read the pdf")
        exit(0)

    # print(text)
    print(len(text))
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)

    result = splitter.split_text(text)
    print(len(result))
    return result

def batched_embed(embedder, texts, batch_size=10):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            batch_embeddings = embedder.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"Batch {i // batch_size + 1} failed: {e}")
            continue
    return np.array(all_embeddings).astype("float32")

# === Step 2: Generate or Load Embeddings and Index ===
if os.path.exists(vec_path) and os.path.exists(index_path) and os.path.exists(meta_path):
    print("Loading FAISS index and metadata...")
    doc_vectors = np.load(vec_path)
    index = faiss.read_index(index_path)
    with open(meta_path, "r") as f:
        doc_chunks = json.load(f)
else:
    print("Reading PDF and generating embeddings...")
    doc_chunks = load_pdf_chunks(pdf_path)


    embedder = GoogleGenerativeAIEmbeddings(
        model=embedding_model, task_type="RETRIEVAL_DOCUMENT"
    )
    doc_vectors = np.array(embedder.embed_documents(doc_chunks)).astype("float32")

    # Save
    np.save(vec_path, doc_vectors)
    with open(meta_path, "w") as f:
        json.dump(doc_chunks, f)

    index = faiss.IndexFlatL2(doc_vectors.shape[1])
    index.add(doc_vectors)
    faiss.write_index(index, index_path)

# === Step 3: Embed Query and Search ===
query = "Explain Sanjay's experirnce in AI field?"
query_embedder = GoogleGenerativeAIEmbeddings(
    model=embedding_model, task_type="RETRIEVAL_QUERY"
)
q_embed = np.array(query_embedder.embed_query(query)).astype("float32")

top_k = 3
D, I = index.search(np.array([q_embed]), top_k)
retrieved_docs = [doc_chunks[i] for i in I[0]]

# === Step 4: Prompt Gemini ===
context = "\n\n".join(retrieved_docs)
prompt = f"""Answer the question based on the context below.

Context:
{context}

Question:
{query}

Answer:"""

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
response = llm.invoke(prompt)

# === Step 5: Output ===
print("Generated Answer:")
print(response.content)
