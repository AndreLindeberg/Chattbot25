import os
import re
import numpy as np
import polars as pl
from pypdf import PdfReader
from dotenv import load_dotenv
import google.generativeai as genai

# API-key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Läs PDF:er
pdf_folder = "data/"
pdf_files = [file for file in os.listdir(pdf_folder) if file.endswith(".pdf")]

all_text = []
for filename in pdf_files:
    reader = PdfReader(os.path.join(pdf_folder, filename))
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    all_text.append(text)

# Dela i meningar
def split_into_sentences(text):
    return [s.strip() for s in re.split(r'(?<=\.)\s+', text) if s.strip()]

all_sentences = []
for doc in all_text:
    all_sentences.extend(split_into_sentences(doc))

print(f" {len(all_sentences)} meningar extraherade")

# Skapa sentence embeddings
def embed_sentences(sentences):
    embeddings = []
    for i, sentence in enumerate(sentences):
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=sentence,
                task_type="SEMANTIC_SIMILARITY"
            )
            embeddings.append(result["embedding"])
        except Exception as e:
            print(f"Fel i mening {i}: {e}")
            embeddings.append([0.0] * 768)
    return embeddings

sentence_embeddings = embed_sentences(all_sentences)

# Semantic chunking
def semantic_chunking(sentences, embeddings, threshold=0.8):
    chunks = []
    current_chunk = [sentences[0]]
    current_vector = embeddings[0]

    for i in range(1, len(sentences)):
        sim = np.dot(current_vector, embeddings[i]) / (
            np.linalg.norm(current_vector) * np.linalg.norm(embeddings[i])
        )
        if sim >= threshold:
            current_chunk.append(sentences[i])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
        current_vector = embeddings[i]
    chunks.append(" ".join(current_chunk))
    return chunks

semantic_chunks = semantic_chunking(all_sentences, sentence_embeddings)

print(f" {len(semantic_chunks)} chunks skapade")

# Skapa chunk-embeddings
def embed_chunks(chunks):
    embeddings = []
    for i, chunk in enumerate(chunks):
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=chunk,
                task_type="SEMANTIC_SIMILARITY"
            )
            embeddings.append(result["embedding"])
        except Exception as e:
            print(f"Fel i chunk {i}: {e}")
            embeddings.append([0.0] * 768)
    return embeddings

chunk_embeddings = embed_chunks(semantic_chunks)

# Spara vector store
df = pl.DataFrame({
    "chunk": semantic_chunks,
    "embedding": chunk_embeddings
})
df.write_parquet("vector_store.parquet")

print("Klar! Vector store sparad som vector_store.parquet")
