import os
import streamlit as st
from dotenv import load_dotenv
from langchain.llms import Ollama
# --------------------------
# Load environment variables
# --------------------------
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HF_TOKEN:
    st.error("❌ Hugging Face API token not found. Add it in .env or Streamlit Secrets.")
    st.stop()

# --------------------------
# Load company data
# --------------------------
def load_company_data(folder="data"):
    texts = []
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                texts.append(f.read())
    return "\n\n".join(texts)

company_data = load_company_data()

# --------------------------
# Build strict prompt
# --------------------------
def build_prompt(question, context):
    return f"""
You are a company assistant. Answer the question ONLY using the information provided below.

If the answer is not in the information, reply exactly: "Sorry, I don’t know."

Information:
{context}

Question: {question}
Answer:
"""

# --------------------------
# Hugging Face Model Client
# --------------------------
llm = Ollama(model="mistral")

def ask_company_bot(question):
    prompt = build_prompt(question, company_data)
    return llm(prompt)


# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Company AI Chatbot", page_icon="🤖")
st.title("🤖 Company Knowledge Chatbot")
st.write("Ask me anything about our company. I will only answer from the company data.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Chat box
if prompt := st.chat_input("Ask a question..."):
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    answer = ask_company_bot(prompt)
    st.chat_message("assistant").write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
