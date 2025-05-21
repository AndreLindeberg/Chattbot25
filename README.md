# 🤖 Chattbot om Typ 1-diabetes

Detta är ett skolprojekt som demonstrerar hur man bygger en faktabaserad chattbot med hjälp av RAG-teknik (Retrieval-Augmented Generation), Google Gemini och Streamlit.

---

## Syfte

Att skapa en interaktiv chattbot som svarar på frågor om typ 1-diabetes baserat på faktagranskade PDF-källor.

---

## Paket som används

- **Streamlit** – gränssnitt
- **Google Gemini API** – språkmodell
- **Polars** – vektorlagring
- **Semantic chunking** – för smart informationshämtning
- **pypdf** – för att läsa in PDF-filer
- **dotenv** – hantering av API-nyckel

---

## Projektstruktur

```
Chattbot/
├── data/                      # PDF-källor
├── vector_store.parquet       # Embeddings + chunks (cache)
├── prepare_data.py            # Bygger och sparar vector store
├── Mybott.py                  # Streamlit-app
├── evaluate_bot.py            # Automatiserad utvärdering
├── README.md                  # Dokumentation
```

---

## Begränsningar

- Alla svar är generella och får inte tolkas som medicinsk rådgivning.
- Chattboten är beroende av innehållet i inlästa dokument.
- Varje individ är unik – svar kan inte ersätta personlig bedömning.

---

## Juridiska överväganden

För en verklig lansering krävs:
- Ansvarsfriskrivning i appen
- Dataskyddsregler (GDPR)
- Juridisk granskning av användning inom vården

---

## Vidareutveckling

- Stöd för fler dokument
- Automatisk sammanfattning av svar: I stället för att visa ett helt chunk med 4 meningar kan appen sammanfatta det     till 1–2 meningar vilket gör svaret mer förståeligt.
- Möjlighet till individanpassning via konto + egen data
- Utveckla chattboten som en mobilapp eller webbtjänst kopplad till vårdsystem eller patientdata.

---

## Validering

Utvärdering sker med 7 kontrollfrågor som jämförs mot fördefinierade ideala svar. Resultatet loggas och poängsätts automatiskt i `evaluate_bot.py`.
För automatisk utvärdering, kör evaluate_bot.py.
Facitsvar och betygssystem finns i koden.

---

## Av: André Lindeberg

Skolprojekt inom ramen för kursen Deep Learning.
