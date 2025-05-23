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



# Functions --------------------------------------------------------------------------------------------------------

# Function - Cosine Similarity
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) *
        np.linalg.norm(vec2))



# Function - Split into sentences
def split_into_sentences(text):
    return [s.strip() for s in re.split(r'(?<=\.)\s+', text) if s.strip()]




# Function - sentence embeddings
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




# Function - Semantic search 
def semantic_search(query, sentences, sentence_embeddings, k=5):
    # Create embedding for question
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=query,
        task_type="SEMANTIC_SIMILARITY"
    )
    query_embedding = result["embedding"]

    # Cosine Similarity between question and all sentences
    similarity_scores = []
    for i, sentence_embedding in enumerate(sentence_embeddings): 
        sim = cosine_similarity(query_embedding, sentence_embedding) 
        similarity_scores.append((i, sim)) 

    # Sort by similarity
    similarity_scores.sort(key=lambda x: x[1], reverse=True)

    # Return top sentences
    top_sentences = [i for i, _ in similarity_scores[:k]] 
    return [sentences[i] for i in top_sentences]




# Function - Semantic chunking
def semantic_chunking(sentences, embeddings, threshold=0.8, max_chunk_words=200):
    chunks = []
    current_chunk = [sentences[0]]
    current_vector = embeddings[0]

    for i in range(1, len(sentences)):
        sim = np.dot(current_vector, embeddings[i]) / (
            np.linalg.norm(current_vector) * np.linalg.norm(embeddings[i])
        )
        current_text = " ".join(current_chunk)
        if sim >= threshold and len(current_text.split()) < max_chunk_words:
            current_chunk.append(sentences[i])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
        current_vector = embeddings[i]
    chunks.append(" ".join(current_chunk))
    return chunks






# Function - chunk-embeddings
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




#----------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    print("prepare_data.py körs som huvudprogram")

    #  Ladda PDF-filer
    pdf_folder = "Data/"
    pdf_files = [file for file in os.listdir(pdf_folder) if file.endswith(".pdf")]

    all_text = []
    for filename in pdf_files:
        path = os.path.join(pdf_folder, filename)
        reader = PdfReader(path)

        text = ""
        for page in reader.pages:
            text += page.extract_text()

        all_text.append({"filename": filename, "text": text})



    # Divide into sentences
    all_sentences = []
    for doc in all_text:
        all_sentences.extend(split_into_sentences(doc["text"]))

    print(f" {len(all_sentences)} meningar extraherade")


    # sentence embeddings
    sentence_embeddings = embed_sentences(all_sentences)


    # Semantic chunking
    semantic_chunks = semantic_chunking(all_sentences, sentence_embeddings)
    print(f" {len(semantic_chunks)} chunks skapade")


    # chunk-embeddings
    chunk_embeddings = embed_chunks(semantic_chunks)



    # Save vector store
    df = pl.DataFrame({
        "chunk": semantic_chunks,
        "embedding": chunk_embeddings
    })
    df.write_parquet("vector_store.parquet")

    print("Klar! Vector store sparad som vector_store.parquet")
