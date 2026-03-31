import os
import shutil
import streamlit as st
import time
from dotenv import load_dotenv
from langchain_community.llms import OpenAI
from langchain_classic.chains import RetrievalQAWithSourcesChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import NewsURLLoader, UnstructuredURLLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

st.set_page_config(page_title="News Researcher", layout="centered")
st.title("News Researcher")
st.markdown("Analyze multiple news sources to answer your questions with relevant citations.")

st.sidebar.title("News URLs")
st.sidebar.markdown("Paste up to 3 news URLs to extract information from:")
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}", key=f"url_{i}")
    if url:
        urls.append(url)

analyse_btn = st.sidebar.button("Analyze Sources")
faiss_index_path = "faiss_index"

main_placeholder = st.empty()

llm = OpenAI(temperature=0.1, max_tokens=500)

query = main_placeholder.text_input("Ask a question about the news articles", placeholder="Type your question here...")

if analyse_btn:
    if urls:
        with st.spinner("Loading and analyzing news articles..."):
            clean_urls = [u.strip() for u in urls if u.strip()]
            clean_urls = list(dict.fromkeys(clean_urls))

            # Prefer news-aware extraction first, then fallback to generic extraction.
            try:
                loader = NewsURLLoader(urls=clean_urls, text_mode=True, nlp=False, continue_on_failure=True)
                data = loader.load()
            except ImportError:
                data = []
            if not data:
                fallback_loader = UnstructuredURLLoader(urls=clean_urls)
                data = fallback_loader.load()

            # Keep only meaningful article-like content.
            filtered_docs = []
            ignored_sources = []
            for i, doc in enumerate(data):
                # QA-with-sources requires every document to include a `source` key.
                if "source" not in doc.metadata or not doc.metadata.get("source"):
                    fallback_source = clean_urls[i] if i < len(clean_urls) else f"source_{i + 1}"
                    doc.metadata["source"] = fallback_source

                text = (doc.page_content or "").strip()
                if len(text) >= 300:
                    filtered_docs.append(doc)
                else:
                    ignored_sources.append(doc.metadata.get("source", "Unknown source"))

            if not filtered_docs:
                st.error("Could not extract enough content from the provided URLs. Try different article links.")
                st.stop()

            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1200,
                chunk_overlap=200
            )
            docs = text_splitter.split_documents(filtered_docs)

            embeddings = OpenAIEmbeddings()
            vector_data = FAISS.from_documents(docs, embeddings)
            time.sleep(2)

            if os.path.exists(faiss_index_path):
                shutil.rmtree(faiss_index_path)
            vector_data.save_local(faiss_index_path)

        st.success(f"Articles analyzed from {len(filtered_docs)} source(s). You can now ask your question.")
        if ignored_sources:
            st.info("Some URLs had very little extractable content and were ignored.")
    else:
        st.warning("Please provide at least one valid URL.")

if query:
    if os.path.exists(faiss_index_path):
        with st.spinner("Fetching the answer..."):
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
            chain = RetrievalQAWithSourcesChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(search_kwargs={"k": 4})
            )
            result = chain({"question": query.strip()}, return_only_outputs=True)
            answer = result.get("answer", "No answer found.")
            sources = result.get("sources", "")

        st.header("Answer")
        st.write(answer)

        if sources:
            st.subheader("Sources")
            for source in sources.split("\n"):
                st.write(source)
        else:
            st.caption("No sources were returned. Try re-analyzing with direct article URLs.")
    else:
        st.warning("Please analyze the URLs first before asking a question.")
