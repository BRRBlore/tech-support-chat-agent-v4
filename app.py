import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
import pandas as pd
import os

# --- PAGE SETUP ---
st.set_page_config(page_title="Tech Support Chat Agent BR v4.0", page_icon="🤖")
st.title("🤖 Tech Support Chat Agent BR v4.0")
st.markdown("Ask your computer, server, or hardware questions below.")

# --- SIDEBAR ---
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

uploaded_file = st.sidebar.file_uploader("Upload Tech Support CSV", type=["csv"])

if not uploaded_file:
    st.info("Please upload a CSV file to get started.")
    st.stop()

if not api_key:
    st.warning("Please enter your OpenAI API key.")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key

# --- LOAD AND EMBED DATA ---
df = pd.read_csv(uploaded_file)
docs = [
    Document(page_content=f"Q: {row['question']}\nA: {row['answer']}")
    for _, row in df.iterrows()
]

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embedding_model)

# --- GPT SETUP ---
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

# --- SESSION STATE FOR MEMORY ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- CHAT FUNCTION ---
def get_bot_response(user_input):
    k = 3
    memory_limit = 3
    retrieved_docs = vectorstore.similarity_search(user_input, k=k)
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    recent_history = st.session_state.chat_history[-memory_limit:]
    formatted_history = ""
    for q, a in recent_history:
        formatted_history += f"User: {q}\nAI: {a}\n"

    prompt = (
        f"You are a smart, helpful tech support assistant.\n"
        f"Use only the provided context and past conversation to answer.\n\n"
        f"Conversation History:\n{formatted_history}\n"
        f"Context:\n{context}\n"
        f"User: {user_input}\nAI:"
    )

    response = llm.predict(prompt)
    st.session_state.chat_history.append((user_input, response))
    return response

# --- CHAT UI ---
for i, (q, a) in enumerate(st.session_state.chat_history):
    st.markdown(f"**You:** {q}")
    st.markdown(f"**AI:** {a}")

user_input = st.text_input("Ask a question:", key="input")

if st.button("Send") and user_input:
    with st.spinner("Thinking..."):
        response = get_bot_response(user_input)
        st.experimental_rerun()

# --- RESET OPTION ---
if st.sidebar.button("🔁 Reset Chat"):
    st.session_state.chat_history = []
    st.experimental_rerun()
