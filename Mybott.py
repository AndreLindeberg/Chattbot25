import os
from dotenv import load_dotenv
import google.generativeai as genai
import streamlit as st
import numpy as np
import polars as pl


#st.write("Den nya versionen av Mybott.py körs") # Testkod

# API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Gemini model
model = genai.GenerativeModel("models/gemini-1.5-flash-latest")



# Function - Cosine Similarity
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) *
        np.linalg.norm(vec2))




# Function - semantic search chunks / Used to find the most relevant chunks based on a question.
def semantic_search_chunks(query, chunks, chunk_embeddings, k=3):
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=query,
        task_type="SEMANTIC_SIMILARITY"
    )
    query_embedding = result["embedding"]

    similarity_scores = []
    for i, chunk_embedding in enumerate(chunk_embeddings):
        sim = cosine_similarity(query_embedding, chunk_embedding)
        similarity_scores.append((i, sim))

    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    top_indices = [i for i, _ in similarity_scores[:k]]

    return [chunks[i] for i in top_indices]






# Vector Store ------------------------------------------------------------------------------------------


VECTOR_STORE_PATH = "vector_store.parquet"

if os.path.exists(VECTOR_STORE_PATH):
    df = pl.read_parquet(VECTOR_STORE_PATH)
    semantic_chunks = df["chunk"].to_list()
    chunk_embeddings = df["embedding"].to_list()
    st.sidebar.success("✅ Vector store laddad från disk")
else:
    st.error("vector_store.parquet saknas. Skapa den först med prepare_data.py och ladda upp till GitHub.")
    st.stop()



# Streamlit -----------------------------------------------------------------------------------------
st.title("🤖 Chattbot om Typ 1-diabetes")
st.markdown(
    "<div style='color: salmon; '>⚠️ Den här chattboten är ett skolprojekt och ger inte medicinska råd. Konsultera alltid vårdpersonal för frågor om din hälsa.</div>",
    unsafe_allow_html=True
)

query = st.text_input("Vad vill du fråga?")

if query:
    with st.spinner("🔍 Letar efter svar i kontexten..."):

        # Search for most relevant chunks
        top_chunks = semantic_search_chunks(query, semantic_chunks, chunk_embeddings, k=2)
        context = "\n\n".join(top_chunks)

        # System prompt - limits model
        system_prompt = """Du är en expert på typ 1-diabetes.
Svara endast utifrån den information som ges i kontexten och ingen annan information.
Om du inte hittar svaret, skriv: 'Det vet jag inte.'
Var tydlig och använd stycken eller punktlistor om det passar."""

        # Prompt for modellen
        user_prompt = f"Fråga: {query}\n\nHär är kontexten:\n{context}"

        # # Generate answer from Gemini
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        response = model.generate_content(
            contents=full_prompt,
            generation_config={"temperature": 0.3, "max_output_tokens": 300}
        )


        # Show answer
        st.markdown("**💬 Svar från Gemini:**")
        st.write(response.text)

        # Show context
        with st.expander("🔍 Visa använd kontext"):
            st.write(context)




# Redogörelse för hur modellen potentiellt skulle kunna  användas i verkligheten och vilka potentiella utmaningar och möjligheter som finns.

# Denna app har potential att utvecklas vidare genom att inkludera fler dokument och utökad kontext, med målet att stötta både personer som lever 
# med typ 1-diabetes och deras anhöriga. Det råder fortfarande stor okunskap om sjukdomen, vilket kan skapa oro och missförstånd i vardagen.
# Genom att erbjuda ett tillgängligt stöd i form av tydliga, faktabaserade svar direkt i appen, kan användare få hjälp i stunden när frågor uppstår.
# Ett problem som ofta uppstår för folk med diabetes eller anhöriga är just att man har en fråga för stunden och inte har tillgång
# till läkare eller sköterska. Ett problem som som appen ska lösa.
# Appen skulle på sikt kunna erbjudas som ett digitalt stödverktyg inom vården, eller lanseras som en fristående app tillgänglig för allmänheten.

# Begränsningar och juridiska överväganden:
# Ett potentiellt problem med denna typ av chattbot är att svaren som ges tenderar att vara generella. 
# Eftersom varje individ reagerar olika på behandling och har unika behov, kan ett generiskt svar 
# aldrig ersätta individuell medicinsk rådgivning. 

# En möjlig lösning vore att individanpassa modellen genom att koppla den till ett personligt konto, 
# där användarens egen data (t.ex. blodsockernivåer, livsstil, behandling) används för att anpassa svaren. 
# Detta skulle kräva omfattande resurser, säker datalagring och eventuellt kontinuerlig finjustering av modellen. 
# Därför skulle en sådan lösning troligen innebära att tjänsten blir avgiftsbelagd.

# Innan appen kan lanseras brett behöver juridiska och etiska aspekter beaktas, särskilt kring ansvarsfrågan 
# vid felaktiga eller missförstådda råd. Det krävs tydlig ansvarsfriskrivning samt eventuell samverkan med vårdgivare 
# för att säkerställa att appen används på ett säkert och korrekt sätt.
