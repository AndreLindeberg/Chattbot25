import os
import polars as pl
from dotenv import load_dotenv
import google.generativeai as genai
import numpy as np
from Mybott import semantic_search_chunks, cosine_similarity


load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("models/gemini-1.5-flash-latest")


# Vector store
df = pl.read_parquet("vector_store.parquet")
semantic_chunks = df["chunk"].to_list()
chunk_embeddings = df["embedding"].to_list()



# Validation data
validation_data = [
    {
        "question": "Vad är typ-1 diabetes?",
        "ideal_answer": """Typ 1-diabetes är en autoimmun sjukdom där kroppens immunsystem attackerar 
        och förstör de insulinproducerande cellerna i bukspottkörteln. Utan insulin kan kroppen inte 
        reglera blodsockret, vilket gör att glukos (socker) ansamlas i blodet istället för att tas upp 
        av kroppens celler."""
    },
    {
        "question": "Vilka är de vanligaste symptomen vid typ 1-diabetes?",
        "ideal_answer": """Vanliga symptom inkluderar ökad törst, stora urinmängder, trötthet, 
        viktnedgång samt synrubbningar. Dessa symptom uppträder ofta snabbt, särskilt hos barn och unga."""
    },
    {
        "question": "Hur behandlas typ 1-diabetes?",
        "ideal_answer": """Behandlingen består av livslång insulinbehandling, som ges genom injektioner 
        eller insulinpump. Behandlingen kombineras med regelbundna blodsockermätningar, anpassad kost 
        och fysisk aktivitet."""
    },
    {
        "question": "Vad är hypoglykemi och vilka är symtomen?",
        "ideal_answer": """Hypoglykemi innebär att blodsockret är för lågt. Vanliga symtom är svettningar, 
        darrningar, hunger, oro, irritation och i allvarliga fall medvetslöshet."""
    },
    {
        "question": "Hur påverkar fysisk aktivitet blodsockret vid typ 1-diabetes?",
        "ideal_answer": """Fysisk aktivitet sänker blodsockret genom att öka kroppens känslighet för insulin. 
        Det är viktigt att anpassa insulin- och kolhydratintag före, under och efter träning för att undvika blodsockerfall."""
    },
    {
        "question": "Vad är HbA1c och varför är det viktigt?",
        "ideal_answer": """HbA1c är ett blodprov som visar medelblodsockret över de senaste 2–3 månaderna. Det används för att bedöma 
        hur väl diabetesbehandlingen fungerar över tid."""
    },
    {
        "question": "Vad ska man göra om en person med diabetes får lågt blodsocker och får svårt att ta hand om sig själv?",
    "ideal_answer": """Om en person med diabetes visar tecken på lågt blodsocker (hypoglykemi) och har svårt att själv hantera det:
- Ge personen snabba kolhydrater, till exempel druvsocker, juice eller något sött att äta.
- Övervaka personen noggrant tills symtomen förbättras.
- Ge aldrig insulin vid misstänkt lågt blodsocker, det sänker blodsockret ytterligare och kan förvärra tillståndet.
- Om personen inte förbättras eller blir medvetslös, ring 112 omedelbart."""
    },
    {
        "question": "'Hur många bultar finns det i Ölandsbron?", 
        "ideal_answer": """Det vet jag inte."""
    }
]



system_prompt = """Du är en expert på typ 1-diabetes.
Svara bara utifrån den kontext du får.
Om du inte hittar svaret, skriv: 'Det vet jag inte.'
Svara tydligt och gärna med punktlistor."""



evaluation_system_prompt = """Du är ett intelligent utvärderingssystem vars uppgift är att utvärdera en chattbots svar angående diabetes typ-1.
Om svaret är väldigt nära det önskade svaret, sätt poängen 1. Om svaret är felaktigt eller inte bra nog, sätt poängen 0.
Om svaret är delvis i linje med det önskade svaret, sätt poängen 0.5. Motivera kort varför du sätter den poäng du gör. """



# Gemini-svarsfunktion
def generate_response(prompt_text, user_prompt_text):
    full_prompt = f"{prompt_text}\n\n{user_prompt_text}"
    response = model.generate_content(
        contents=full_prompt,
        generation_config={"temperature": 0.3}
    )
    return response.text.strip()

# Evaluation
for i, entry in enumerate(validation_data):
    query = entry["question"]
    ideal = entry["ideal_answer"]

    top_chunks = semantic_search_chunks(query, semantic_chunks, chunk_embeddings, k=3)
    context = "\n".join(top_chunks)

    user_prompt = f"Fråga: {query}\n\nHär är kontexten:\n{context}"
    ai_answer = generate_response(system_prompt, user_prompt)

    evaluation_prompt = f"""Fråga: {query}
AI-svar: {ai_answer}
Facit: {ideal}"""

    evaluation = generate_response(evaluation_system_prompt, evaluation_prompt)

    print(f"\n--- Fråga {i+1} ---")
    print("?", query)
    print("AI-svar:", ai_answer)
    print("Facit:", ideal)
    print("Bedömning:", evaluation)
