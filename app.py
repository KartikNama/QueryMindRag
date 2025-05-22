#Imports and Setup
import os
import streamlit as st
from streamlit_chat import message

st.set_page_config(
    page_title="Zzapkart Chatbot",
    page_icon="🛒",
    layout="centered",
    initial_sidebar_state="collapsed"
)

custom_css = """
<style>
    body {
        background-color: #121212;
        color: white;
    }
    .stTextInput > div > div > input {
        background-color: #1f1f1f;
        color: white;
    }
    .stButton>button {
        background-color: #ff6f00;
        color: white;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


#LangChain and Utilities
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

#Fix SQLite for LangChain
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

#API Key Setup
os.environ["COHERE_API_KEY"] = 'Ox97SolGnL68xrDjbNAMiVaWCqZ5Fny3d7hYAub6'

#Document Preprocessing
@st.cache_data
def doc_preprocessing():
    loader = PyPDFLoader("Zzapkart.pdf")
    docs = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(docs)

#Embeddings Store
@st.cache_resource
def embeddings_store():
    embedding = CohereEmbeddings(model="embed-english-v3.0")
    texts = doc_preprocessing()
    vectordb = FAISS.from_documents(documents=texts, embedding=embedding)
    return vectordb.as_retriever()

#Conversational RAG Chain
@st.cache_resource
def conversational_qa():
    retriever = embeddings_store()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    return ConversationalRetrievalChain.from_llm(
        llm=ChatCohere(),
        memory=memory,
        retriever=retriever
    )

#Simple Retrieval QA Chain (No memory)
@st.cache_resource
def rag_qa_chain():
    retriever = embeddings_store()
    return RetrievalQA.from_chain_type(
        llm=ChatCohere(),
        chain_type="stuff",
        retriever=retriever
    )

#Chat Display
def display_conversation(history):
    for i in reversed(range(len(history["generated"]))):
        message(history["past"][i], is_user=True, key=f"{i}_user")
        message(history["generated"][i], key=str(i))

        # Feedback buttons
        feedback_key = f"feedback_{i}"
        if feedback_key not in st.session_state:
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("👍", key=f"{feedback_key}_up"):
                    st.session_state[feedback_key] = "positive"
                    st.success("Thanks for your feedback! 😊")
            with col2:
                if st.button("👎", key=f"{feedback_key}_down"):
                    st.session_state[feedback_key] = "negative"
                    st.warning("We’ll use your feedback to improve!")
                    

# ========================
# ✅ Zzapkart Custom Logic
# ========================

# Instructions for fallback LLM
zzapkart_instructions = (
    "You are a helpful support assistant for Zzapkart, an e-commerce company. "
    "You assist with order tracking, cancellations (only if not yet shipped), and replacements (within 7 days of delivery). "
    "If a user gives an invalid Order ID or doesn't meet the policy, refer them to support@zzapkart.com. "
    "Speak clearly and politely, following Zzapkart’s official policies from the company handbook."
)

# import pandas as pd

# # Load order data from Excel into a dictionary
# @st.cache_data
# def load_order_data():
#     df = pd.read_excel("data.xlsx")
#     orders = {
#         row["order_id"]: {
#             "status": row["status"],
#             "delivered": bool(row["delivered"]),
#             "date": str(row["date"].date() if hasattr(row["date"], 'date') else row["date"])
#         }
#         for _, row in df.iterrows()
#     }
#     return orders

# orders = load_order_data()

# Simulated Order DB
orders = {
    "ZZP123": {"status": "Processing", "delivered": False, "date": "2025-05-10"},
    "ZZP456": {"status": "Shipped", "delivered": False, "date": "2025-05-08"},
    "ZZP789": {"status": "Delivered", "delivered": True, "date": "2025-05-03"}
}

# Detect intent from message
def detect_intent(msg):
    msg = msg.lower()
    if "cancel" in msg:
        return "cancel"
    if "replace" in msg or "damaged" in msg:
        return "replace"
    if "track" in msg:
        return "track"
    return "fallback"

# Rule-based logic override
def rule_based_response(user_msg):
    intent = detect_intent(user_msg)
    order_id = None
    parts = user_msg.upper().split()
    for p in parts:
        if p.startswith("ZZP"):
            order_id = p
            break

    if intent == "cancel":
        if not order_id:
            return "Please provide your Order ID to proceed with cancellation."
        order = orders.get(order_id)
        if not order:
            return "I couldn't find that Order ID. Please check or contact support@zzapkart.com."
        if order["status"] == "Processing":
            return f"Your order {order_id} has been cancelled. You’ll receive a confirmation email shortly."
        else:
            return f"Sorry, your order {order_id} has already shipped. Please request a return after delivery."

    elif intent == "replace":
        if not order_id:
            return "Please provide your Order ID and describe the issue with your product."
        order = orders.get(order_id)
        if not order:
            return "Invalid Order ID. Please double-check or contact support@zzapkart.com."
        if order["delivered"]:
            return (
                f"Order {order_id} was delivered on {order['date']}. "
                "You are eligible for a replacement if it’s within 7 days. Please upload a product image for verification."
            )
        else:
            return f"Replacements are only available after delivery. Your order {order_id} hasn’t been delivered yet."

    elif intent == "track":
        if not order_id:
            return "Sure! Please provide your Order ID to check tracking."
        order = orders.get(order_id)
        if not order:
            return "I couldn't find that Order ID. Please re-check or contact support@zzapkart.com."
        return f"Your order {order_id} is currently '{order['status']}' (placed on {order['date']})."

    return None  # fallback to LLM

#Main App Logic
def main_f():
    st.title("🛒 Zzapkart Support Assistant")
    st.markdown("### 🤖 Meet **Zzappy**, your Zzapkart assistant!")

    #Chain Initialization
    rag_chain = rag_qa_chain()
    convo_chain = conversational_qa()

    #Session Initialization
    if "generated" not in st.session_state:
      st.session_state["generated"] = ["I’m Zzappy, your virtual assistant 🤖. How can I help you today?"]
    if "past" not in st.session_state:
      st.session_state["past"] = ["Welcome to Zzapkart Support 👋"]


    #Take input
    user_query = st.text_input("Ask your question:")

    #Generate Response
    if user_query:
      with st.spinner("Thinking..."):
          if st.checkbox("Use Document-based QA (Simple RAG)", key="rag_toggle"):
              output = rag_chain.run(user_query)
          else:
              output = rule_based_response(user_query)
              if not output:
                  output = convo_chain({"question": user_query})["answer"]

          st.session_state.past.append(user_query)
          st.session_state.generated.append(output)

    #Show Chat history
    if st.session_state["generated"]:
        display_conversation(st.session_state)

#Execution of logic
if __name__ == "__main__":
    main_f()
