# What Was Learned

This project served as a hands-on exploration of LLM-based retrieval systems. The following concepts were covered during development.

## LangChain

- How to chain together loaders, splitters, embeddings, and retrievers into a single pipeline.
- Using `RetrievalQAWithSourcesChain` to return answers alongside cited sources.
- The difference between document loaders and text splitters, and when each is applied.

## FAISS

- What a vector store is and why similarity search scales better than keyword search.
- How to persist and reload a FAISS index from disk across sessions.
- The role of embeddings in converting raw text into searchable high-dimensional vectors.

## OpenAI API

- How text embeddings differ from chat completions in purpose and cost.
- Controlling generation behavior through `temperature` and `max_tokens`.

## Retrieval-Augmented Generation (RAG)

- The full RAG pipeline: load, split, embed, store, retrieve, generate.
- Why chunk size and overlap affect retrieval quality and answer accuracy.
- How source attribution works in a QA chain when documents carry metadata.

## Streamlit

- Building an interactive UI with sidebar inputs and placeholder-based dynamic content.
- Handling state between button clicks and text inputs within a single-page app.
