# llmragPOC

This is a small RAG app built with Python and Streamlit. Use a few article URLs, the app pulls the text, finds the most relevant bits, and then tries to answer question from that content.

## What it is doing

It’s basically a mini version of a document QA:

1. Pull content from the web
   - Using Python with requests and BeautifulSoup, it grabs the page HTML and cleans it up.
   - That part removes junk like scripts, styles, and other messy bits so it's left with actual readable content.

2. Break the text into chunks
   - The app splits the article text into smaller chunks.
   - That helps with retrieval and makes it easier to find the right section later.

3. Turn text into embeddings
   - With scikit-learn, it converts each chunk into a numeric representation called an embedding.
   - That part lets the app do semantic search instead of just matching exact words.

4. Database part
    - This app doesn’t use a full database like Postgres or MongoDB. For this demo, the database part is just a local pickle file named news_index.pkl.
    - Python saves the chunked article text and the matching embeddings to a local file.
    - scikit-learn handles the embedding/vector math.
    - When I ask a question later, the app reloads that saved index and does the search again.
    - the database part here is basically a lightweight local index, not a big production vector DB. If I wanted to make it more serious later, could swap this out for something like Chroma, Pinecone, or Weaviate.

5. Search for relevant content
   - When you ask a question, the app compares your question against the stored chunks.
   - It looks for the chunks that are most similar in meaning.
   - That’s the semantic search part.

6. Use an LLM to answer
   - If an OpenAI API key is available, the app sends the retrieved context to an LLM using the OpenAI Python SDK.
   - The model uses that context to answer question.
   - That’s the RAG part: the answer comes from retrieved documents, not just the model’s memory.

## Some of the terms

- LLM: the language model that writes the answer.
- RAG: retrieval-augmented generation, where the model answers using retrieved text.
- Embedding: a numeric version of text that captures meaning.
- Vector database: a simple local index that stores embeddings so similar text can be found quickly.
- Semantic search: finding content by meaning, not just literal keywords.
- Prompt: the instruction and context sent to the model.
- Temperature: a setting that controls how creative or strict the answer should be.

## How to run it

```bash
cd llmragPOC
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Then start it with:

```bash
streamlit run main.py
```