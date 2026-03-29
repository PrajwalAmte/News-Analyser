import os
import streamlit as st
import time
from dotenv import load_dotenv
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
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

llm = OpenAI(temperature=0.9, max_tokens=500)

query = main_placeholder.text_input("Ask a question about the news articles", placeholder="Type your question here...")

if analyse_btn:
    if urls:
        with st.spinner("Loading and analyzing news articles..."):
            loader = UnstructuredURLLoader(urls=urls)
            data = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000
            )
            docs = text_splitter.split_documents(data)

            embeddings = OpenAIEmbeddings()
            vector_data = FAISS.from_documents(docs, embeddings)
            time.sleep(2)
            vector_data.save_local(faiss_index_path)

        st.success("Articles analyzed. You can now ask your question.")
    else:
        st.warning("Please provide at least one valid URL.")

if query:
    if os.path.exists(faiss_index_path):
        with st.spinner("Fetching the answer..."):
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            answer = result.get("answer", "No answer found.")
            sources = result.get("sources", "")

        st.header("Answer")
        st.write(answer)

        if sources:
            st.subheader("Sources")
            for source in sources.split("\n"):
                st.write(source)
    else:
        st.warning("Please analyze the URLs first before asking a question.")
