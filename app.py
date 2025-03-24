import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
import pandas as pd
import os

# --- PAGE SETUP ---
st.set_page_config(page_title="Tech Support Chat Agent BR v4.0", page_icon="🤖")
st.markdown("""
    <style>
    .send-button > button {
        background-color: #2E8B57 !important;
        color: white !important;
        font-weight: bold;
        border-radius: 5px;
    }
    .stTextInput>div>div>input {
        font-size: 16px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 Tech Support Chat Agent BR v4.0")
st.markdown("Ask your computer, server, or hardware questions below.")

# --- LOAD DATA & SETUP ---
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
DEFAULT_CSV_PATH = "tech_support_sample_QA.csv"
df = pd.read_csv(DEFAULT_CSV_PATH)
docs = [Document(page_content=f"Q: {row['question']}\nA: {row['answer']}") for _, row in df.iterrows()]

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embedding_model)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

# --- SESSION STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "input_buffer" not in st.session_state:
    st.session_state.input_buffer = ""

# --- CHAT FUNCTION ---
def get_bot_response(user_input):
    k = 3
    memory_limit = 3
    retrieved_docs = vectorstore.similarity_search(user_input, k=k)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    recent_history = st.session_state.chat_history[-memory_limit:]
    formatted_history = "".join([f"User: {q}\nAI: {a}\n" for q, a in recent_history])
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
for q, a in st.session_state.chat_history:
    st.markdown(f"**🧑 You:** {q}")
    st.markdown(f"**🤖 AI:** {a}")

# --- Input + Send ---
st.session_state.input_buffer = st.text_input("Ask a question:", value=st.session_state.input_buffer, key="text_input")

col1, col2 = st.columns([1, 8])
with col1:
    send_clicked = st.button("Send", key="send_button", use_container_width=True)

# --- Handle Send ---
if send_clicked or st.session_state.input_buffer:
    user_input = st.session_state.input_buffer.strip()
    if user_input:
        with st.spinner("Thinking..."):
            get_bot_response(user_input)
        st.session_state.input_buffer = ""
        st.rerun()

# --- RESET OPTION ---
if st.button("🔁 Reset Chat"):
    st.session_state.chat_history = []
    st.session_state.input_buffer = ""
    st.rerun()
