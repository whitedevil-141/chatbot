from langchain.llms import HuggingFaceHub
import os
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# Set Hugging Face API key (you can get one free at https://huggingface.co/settings/tokens)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "HUGGINGFACEHUB_API_TOKEN"

# Streamlit UI
st.set_page_config(page_title="AI Business Chatbot", page_icon="🤖")
st.title("🤖 Business AI Chatbot")

docs = [
    "Brew&Bean is a coffee shop open from 8am to 8pm every day.",
    "We serve coffee, tea, pastries, and sandwiches.",
    "Customers can book tables online or via WhatsApp.",
    "Refunds are available only for online pre-orders."
]

# Embeddings + Vector DB
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma.from_texts(docs, embedding)

# Hugging Face model (free small one)
llm = HuggingFaceHub(repo_id="google/flan-t5-small")

retriever = db.as_retriever()
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Type your question..."):
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    answer = qa.run(prompt)
    st.chat_message("assistant").write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
