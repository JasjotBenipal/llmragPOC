import os
import pickle
import re
from typing import List, Tuple

import numpy as np
import requests
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

st.set_page_config(page_title="llmragPOC", layout="wide")

# App title shown in the browser and in the Streamlit UI.
st.title("llmragPOC: News Research Tool")
st.sidebar.title("News Article URLs")
st.sidebar.caption("Paste up to 3 article URLs, then ask a question about them.")

# Collect the URLs the user wants to analyze.
urls = [st.sidebar.text_input(f"URL {index + 1}", value="") for index in range(3)]
process_url_clicked = st.sidebar.button("Process URLs", use_container_width=True)

# File used to store the local search index so it can be reused between runs.
INDEX_PATH = "news_index.pkl"


def extract_text_from_html(html: str, source_url: str) -> str:
    # Parse the HTML and remove non-content elements such as scripts and styles.
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text_blocks: List[str] = []
    for element in soup.find_all(["article", "main", "section", "p", "li", "h1", "h2", "h3", "div"]):
        text = " ".join(element.get_text(" ", strip=True).split())
        if text:
            text_blocks.append(text)

    if text_blocks:
        return "\n\n".join(text_blocks)

    return " ".join(soup.get_text(" ", strip=True).split())


def fetch_article_text(url: str) -> str:
    # Download the article HTML and convert it into clean text content.
    try:
        response = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise ValueError(f"Unable to fetch article: {exc}") from exc

    return extract_text_from_html(response.text, url)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    # Split the article content into smaller passages so retrieval is faster and more precise.
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    if not paragraphs:
        return []

    chunks: List[str] = []
    current_chunk = ""
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) <= chunk_size:
            current_chunk = f"{current_chunk}\n\n{paragraph}".strip() if current_chunk else paragraph
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk)

    final_chunks: List[str] = []
    for chunk in chunks:
        if len(chunk) <= chunk_size:
            final_chunks.append(chunk)
            continue

        parts = re.split(r"(?<=[.?!])\s+", chunk)
        current = ""
        for part in parts:
            if len(current) + len(part) <= chunk_size:
                current = f"{current} {part}".strip() if current else part
            else:
                if current:
                    final_chunks.append(current)
                current = part
        if current:
            final_chunks.append(current)

    if overlap > 0 and len(final_chunks) > 1:
        overlapped: List[str] = []
        for index, chunk in enumerate(final_chunks):
            if index == 0:
                overlapped.append(chunk)
                continue
            previous = overlapped[-1]
            if len(previous) < overlap:
                overlapped.append(chunk)
                continue
            overlap_text = previous[-overlap:]
            overlapped.append(f"{overlap_text}\n\n{chunk}")
        return overlapped

    return final_chunks


def build_index(chunks: List[str], sources: List[str] | None = None) -> dict:
    # Create a lightweight vector space model for semantic retrieval.
    # Each chunk becomes a numeric representation that can be compared to the user's question.
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    embeddings = vectorizer.fit_transform(chunks)
    return {
        "chunks": chunks,
        "vectorizer": vectorizer,
        "embeddings": embeddings,
        "sources": sources or ["Unknown source"] * len(chunks),
    }


def save_index(index: dict, file_path: str) -> None:
    # Persist the index to disk so the app can reuse it without reprocessing every time.
    with open(file_path, "wb") as handle:
        pickle.dump(index, handle)


def normalize_index(index: dict) -> dict:
    # Backward compatibility for older saved indexes that do not have a sources field.
    if "sources" not in index:
        index["sources"] = ["Unknown source"] * len(index.get("chunks", []))
    return index


def load_index(file_path: str) -> dict:
    # Load the saved index from disk.
    with open(file_path, "rb") as handle:
        return normalize_index(pickle.load(handle))


def get_answer(query: str, index: dict) -> Tuple[str, List[str]]:
    # Convert the user's question into the same vector space as the article chunks.
    query_vector = index["vectorizer"].transform([query])

    # Measure semantic relevance between the question and each chunk.
    similarities = cosine_similarity(query_vector, index["embeddings"]).ravel()
    top_indices = np.argsort(similarities)[::-1][:4]
    context_parts = [index["chunks"][idx] for idx in top_indices if similarities[idx] > 0.0]
    context = "\n\n".join(context_parts) if context_parts else "No supporting context was found."

    # If an API key is available, pass the retrieved context to an LLM for a polished answer.
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        client = OpenAI(api_key=api_key)
        # The prompt tells the language model to answer only from the retrieved context.
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": "You answer questions using the provided article context. If the context is insufficient, say so plainly.",
                },
                {"role": "user", "content": f"Question: {query}\n\nContext:\n{context}"},
            ],
        )
        answer = response.choices[0].message.content.strip()
        sources = [index["sources"][idx] for idx in top_indices[:3] if idx < len(index["sources"])]
        return answer, sources

    answer = (
        "OpenAI API key not found, so I’m answering from the closest retrieved snippets.\n\n"
        + "\n\n".join(context_parts[:3])
    )
    return answer, [index["sources"][idx] for idx in top_indices[:3] if idx < len(index["sources"])]


if process_url_clicked:
    # Only process URLs that the user actually entered.
    selected_urls = [url.strip() for url in urls if url and url.strip()]
    if not selected_urls:
        st.warning("Please enter at least one URL before processing.")
    else:
        with st.spinner("Fetching and indexing articles..."):
            all_chunks: List[str] = []
            all_sources: List[str] = []
            for url in selected_urls:
                try:
                    article_text = fetch_article_text(url)
                    chunks = chunk_text(article_text)
                    all_chunks.extend(chunks)
                    all_sources.extend([url] * len(chunks))
                except Exception as exc:  # noqa: BLE001
                    st.sidebar.warning(f"Skipping {url}: {exc}")

            if all_chunks:
                index = build_index(all_chunks, all_sources)
                save_index(index, INDEX_PATH)
                st.success(f"Indexed {len(all_chunks)} text chunks from {len(selected_urls)} article(s).")
            else:
                st.error("No readable text was extracted from the provided URLs.")

query = st.text_input("Question")
if query:
    if os.path.exists(INDEX_PATH):
        index = load_index(INDEX_PATH)
        answer, sources = get_answer(query, index)
        st.header("Answer")
        st.write(answer)
        if sources:
            st.subheader("Sources")
            for source in sources:
                if source.startswith("http"):
                    st.markdown(f"- [{source}]({source})")
                else:
                    st.write(source)
    else:
        st.info("Process at least one URL first so the knowledge base exists.")